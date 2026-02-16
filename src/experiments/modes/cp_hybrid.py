"""
CPHybridExperiment -- GA + CP-SAT Distributed Constraint Repair (Mode G).

A **pure genetic algorithm** where the **only** repair mechanism is the
CP-SAT solver using a distributed constraint programming approach.

Architecture::

    For each generation:
      1. Tournament selection
      2. Crossover  (course-aware attribute swap)
      3. Mutation   (constraint-aware gene mutation)
      4. CP-SAT repair on each offspring  (the SOLE repair)
         -- decomposes into clusters, solves bridge genes globally,
            then solves each cluster independently
         -- hard constraints enforced as hard CP constraints
         -- soft constraints (compactness) as objective terms
      5. Evaluate fitness
      6. NSGA-II survivor selection

No heuristic repairs, no deterministic repair, no gene-level local
search, no RepairEngine.  All constraint satisfaction is delegated
to OR-Tools CP-SAT.

Usage::

    from src.experiments.modes.cp_hybrid import CPHybridExperiment

    exp = CPHybridExperiment(
        seed=42, pop_size=20, ngen=50,
        fitness_weights=(-1.0, -1.0),
        cp_timeout=10,
        cp_full_interval=10,
    )
    exp.run()
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from src.experiments.base import BaseExperiment
from src.ga.run_helpers import EvolutionStats

logger = logging.getLogger(__name__)


class CPHybridExperiment(BaseExperiment):
    """GA + CP-SAT Hybrid: Pure GA with distributed CP-SAT as sole repair.

    Parameters (on top of BaseExperiment)
    -------------------------------------
    cp_timeout : float
        Max seconds for a quick per-individual CP-SAT repair (violated
        genes only).  Default 10.
    cp_timeout_full : float
        Max seconds for a full pipeline CP-SAT repair (global + cluster
        decomposition).  Default 60.
    cp_num_workers : int
        Number of CP-SAT solver workers (threads per model).  Default 8.
    cp_min_shared_courses : int
        Minimum shared courses to merge programmes into one cluster.
        Default 2.
    cp_full_interval : int
        Run the full decomposed CP pipeline on the best individual every
        N generations.  Default 10.
    cp_soft_objective : bool
        When True, the CP solver minimises soft-constraint proxies
        (group/instructor compactness) alongside deviation.  Default True.
    tournament_size : int
        Tournament selection size.  Default 3.
    """

    def __init__(
        self,
        *,
        # CP-SAT parameters
        cp_timeout: float = 10.0,
        cp_timeout_full: float = 60.0,
        cp_num_workers: int = 8,
        cp_min_shared_courses: int = 2,
        cp_full_interval: int = 10,
        cp_soft_objective: bool = True,
        tournament_size: int = 3,
        # Parent
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cp_timeout = cp_timeout
        self.cp_timeout_full = cp_timeout_full
        self.cp_num_workers = cp_num_workers
        self.cp_min_shared_courses = cp_min_shared_courses
        self.cp_full_interval = cp_full_interval
        self.cp_soft_objective = cp_soft_objective
        self.tournament_size = tournament_size

        # Tracking
        self._cp_quick_repairs: int = 0
        self._cp_full_repairs: int = 0
        self._cp_quick_success: int = 0
        self._cp_full_success: int = 0
        self._best_hard_history: list[float] = []
        self._best_soft_history: list[float] = []

    def _get_experiment_name(self) -> str:
        return "ga_07_cp_hybrid"

    def _get_extra_config(self) -> dict[str, Any]:
        return {
            "cp_timeout": self.cp_timeout,
            "cp_timeout_full": self.cp_timeout_full,
            "cp_num_workers": self.cp_num_workers,
            "cp_min_shared_courses": self.cp_min_shared_courses,
            "cp_full_interval": self.cp_full_interval,
            "cp_soft_objective": self.cp_soft_objective,
            "tournament_size": self.tournament_size,
        }

    def _get_extra_results(self) -> dict[str, Any]:
        return {
            "cp_quick_repairs": self._cp_quick_repairs,
            "cp_full_repairs": self._cp_full_repairs,
            "cp_quick_success": self._cp_quick_success,
            "cp_full_success": self._cp_full_success,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_hard_breakdown(self, ind: list) -> dict[str, int]:
        """Return per-constraint hard violation counts for *ind*."""
        from src.constraints.evaluator import Evaluator
        from src.domain.timetable import Timetable

        tt = Timetable(genes=ind, context=self.data.context)
        ev = Evaluator()
        return {c.name: int(c.weight * c.evaluate(tt)) for c in ev.hard}

    def _get_soft_breakdown(self, ind: list) -> dict[str, float]:
        """Return per-constraint soft penalty for *ind*."""
        from src.constraints.evaluator import Evaluator
        from src.domain.timetable import Timetable

        tt = Timetable(genes=ind, context=self.data.context)
        ev = Evaluator()
        return {c.name: c.weight * c.evaluate(tt) for c in ev.soft}

    # ------------------------------------------------------------------
    # CP-SAT repair: quick (violated genes only)
    # ------------------------------------------------------------------

    def _cp_repair_quick(
        self,
        ind: list,
        family_map: dict[str, set[str]],
    ) -> list:
        """Run a quick CP-SAT repair on violated genes only.

        Non-violated genes are frozen so the solver only fixes conflicts.
        """
        from src.ga.repair.cp.solver import CPSATSolver, FrozenAssignment
        from src.ga.repair.detector import detect_violated_genes

        violations = detect_violated_genes(ind, self.data.context, strategy="hybrid")
        if not violations:
            return ind  # already feasible

        violated_indices = sorted(violations.keys())
        violated_set = set(violated_indices)

        # Freeze all non-violated genes
        frozen = [
            FrozenAssignment.from_gene(i, g)
            for i, g in enumerate(ind)
            if i not in violated_set
        ]

        solver = CPSATSolver(
            self.data.context,
            family_map,
            timeout_seconds=self.cp_timeout,
            num_workers=self.cp_num_workers,
            soft_objective=self.cp_soft_objective,
        )

        result = solver.solve(
            ind,
            violated_indices,
            frozen=frozen,
            warm_start=True,
        )

        self._cp_quick_repairs += 1

        if result.success:
            self._cp_quick_success += 1
            repaired = list(ind)
            for gi, (iid, rid, sq) in result.assignments.items():
                repaired[gi] = copy.deepcopy(ind[gi])
                repaired[gi].instructor_id = iid
                repaired[gi].room_id = rid
                repaired[gi].start_quanta = sq
            return repaired

        return ind  # solver failed, return original

    # ------------------------------------------------------------------
    # CP-SAT repair: full pipeline (decomposed)
    # ------------------------------------------------------------------

    def _cp_repair_full(
        self,
        ind: list,
        family_map: dict[str, set[str]],
    ) -> tuple[list, float, float]:
        """Run the full decomposed CP-SAT pipeline on an individual.

        Returns (repaired_ind, hard, soft).
        """
        from src.ga.repair.cp.pipeline import CPRepairPipeline

        pipeline = CPRepairPipeline(
            timeout_global=self.cp_timeout_full,
            timeout_cluster=self.cp_timeout,
            num_workers=self.cp_num_workers,
            min_shared_courses=self.cp_min_shared_courses,
            soft_objective=self.cp_soft_objective,
        )

        repaired, cp_stats = pipeline.repair(
            ind,
            self.data.context,
            family_map,
        )

        self._cp_full_repairs += 1
        if cp_stats.success:
            self._cp_full_success += 1

        h, s = self.evaluate(repaired)
        return repaired, h, s

    # ------------------------------------------------------------------
    # Core evolution loop
    # ------------------------------------------------------------------

    def _run_evolution(self) -> tuple[list[Any], EvolutionStats]:
        """GA loop: select -> crossover -> mutate -> CP-SAT repair -> evaluate."""
        import random

        from deap import base, creator, tools

        from src.ga.core.population import get_family_map_from_json

        start_time = time.time()
        stats = EvolutionStats()

        # Build family map for CP solver
        groups_json = str(self.data_dir / "Groups.json")
        family_map = get_family_map_from_json(groups_json)

        # DEAP setup
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=self.fitness_weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        # ============================================================
        # Phase 1: Create and CP-repair initial population
        # ============================================================
        self.logger.info(
            "Phase 1: Creating initial population (%d individuals)...",
            self.pop_size,
        )

        pop = self.create_initial_population()

        # CP-repair every individual in the initial population
        self.logger.info("Phase 1: CP-SAT repairing initial population...")
        for i, ind in enumerate(pop):
            repaired = self._cp_repair_quick(list(ind), family_map)
            ind[:] = repaired
            ind.fitness.values = self.evaluate(ind)
            if (i + 1) % max(1, self.pop_size // 5) == 0:
                best_h = min(p.fitness.values[0] for p in pop[: i + 1])
                best_s = min(
                    p.fitness.values[1]
                    for p in pop[: i + 1]
                    if p.fitness.values[0] == best_h
                )
                self.logger.info(
                    "  Repaired %d/%d  (best so far: Hard=%.0f Soft=%.0f)",
                    i + 1,
                    self.pop_size,
                    best_h,
                    best_s,
                )

        best = min(pop, key=lambda p: (p.fitness.values[0], p.fitness.values[1]))
        self.logger.info(
            "Phase 1 complete: Best Hard=%.0f Soft=%.0f (%.1fs)",
            best.fitness.values[0],
            best.fitness.values[1],
            time.time() - start_time,
        )

        # Log initial constraint breakdown
        bd = self._get_hard_breakdown(list(best))
        for name, cnt in sorted(bd.items(), key=lambda x: -x[1]):
            if cnt > 0:
                self.logger.info("  %s: %d", name, cnt)

        # ============================================================
        # Phase 2: GA evolution with CP-SAT as sole repair
        # ============================================================
        self.logger.info(
            "Phase 2: GA evolution (%d generations, pop=%d)",
            self.ngen,
            self.pop_size,
        )

        random.seed(self.seed)

        for gen in range(self.ngen):
            gen_start = time.time()

            # -- Selection (tournament) --
            offspring = tools.selTournament(
                pop, len(pop), tournsize=self.tournament_size
            )
            offspring = [
                creator.Individual(copy.deepcopy(list(ind))) for ind in offspring
            ]

            # -- Crossover --
            self.apply_crossover(offspring)

            # -- Mutation --
            self.apply_mutation(offspring)

            # -- CP-SAT repair (the ONLY repair) --
            for ind in offspring:
                if not ind.fitness.valid:
                    repaired = self._cp_repair_quick(list(ind), family_map)
                    ind[:] = repaired

            # -- Evaluate --
            self.evaluate_offspring(offspring)

            # -- Survivor selection (NSGA-II: mu+lambda) --
            combined = pop + offspring
            pop = tools.selNSGA2(combined, self.pop_size)

            # -- Periodic full CP pipeline on best individual --
            if (gen + 1) % self.cp_full_interval == 0:
                best_idx = min(
                    range(len(pop)),
                    key=lambda i: (
                        pop[i].fitness.values[0],
                        pop[i].fitness.values[1],
                    ),
                )
                best_ind = pop[best_idx]
                if best_ind.fitness.values[0] > 0:
                    self.logger.info(
                        "  Full CP pipeline on best (Hard=%.0f)...",
                        best_ind.fitness.values[0],
                    )
                    prev_h = best_ind.fitness.values[0]
                    prev_s = best_ind.fitness.values[1]
                    repaired, new_h, new_s = self._cp_repair_full(
                        list(best_ind), family_map
                    )
                    if (new_h, new_s) <= (prev_h, prev_s):
                        pop[best_idx][:] = repaired
                        pop[best_idx].fitness.values = (new_h, new_s)
                        self.logger.info(
                            "  Full CP: Hard %.0f->%.0f, Soft %.0f->%.0f",
                            prev_h,
                            new_h,
                            prev_s,
                            new_s,
                        )

            # -- Stats & logging --
            best = min(pop, key=lambda p: (p.fitness.values[0], p.fitness.values[1]))
            self._best_hard_history.append(float(best.fitness.values[0]))
            self._best_soft_history.append(float(best.fitness.values[1]))

            self.record_generation_stats(pop, stats, gen, gen_start)

            elapsed = time.time() - start_time
            if gen == 0 or (gen + 1) % self.log_interval == 0 or gen == self.ngen - 1:
                self.logger.info(
                    "Gen %3d/%d: Best Hard=%.0f Soft=%.0f  " "Feasible=%d/%d  (%.1fs)",
                    gen + 1,
                    self.ngen,
                    best.fitness.values[0],
                    best.fitness.values[1],
                    sum(1 for p in pop if p.fitness.values[0] == 0),
                    self.pop_size,
                    elapsed,
                )

            # Early stopping if all hard constraints satisfied
            if best.fitness.values[0] == 0:
                self.logger.info("Reached 0 hard violations at gen %d!", gen + 1)
                break

        # ============================================================
        # Phase 3: Final full CP-SAT polish on best individual
        # ============================================================
        best = min(pop, key=lambda p: (p.fitness.values[0], p.fitness.values[1]))
        if best.fitness.values[0] > 0:
            self.logger.info(
                "Phase 3: Final full CP pipeline (Hard=%.0f)...",
                best.fitness.values[0],
            )
            repaired, new_h, new_s = self._cp_repair_full(list(best), family_map)
            if (new_h, new_s) <= (best.fitness.values[0], best.fitness.values[1]):
                best_idx = pop.index(best)
                pop[best_idx][:] = repaired
                pop[best_idx].fitness.values = (new_h, new_s)
                self.logger.info(
                    "Phase 3: Hard %.0f->%.0f, Soft %.0f->%.0f",
                    best.fitness.values[0],
                    new_h,
                    best.fitness.values[1],
                    new_s,
                )

        stats.elapsed_time = time.time() - start_time

        # Final summary
        final_best = min(pop, key=lambda p: (p.fitness.values[0], p.fitness.values[1]))
        self.logger.info(
            "DONE: Hard=%.0f Soft=%.0f  time=%.1fs  "
            "CP repairs: %d quick (%d ok), %d full (%d ok)",
            final_best.fitness.values[0],
            final_best.fitness.values[1],
            stats.elapsed_time,
            self._cp_quick_repairs,
            self._cp_quick_success,
            self._cp_full_repairs,
            self._cp_full_success,
        )

        # Constraint breakdowns
        bd = self._get_hard_breakdown(list(final_best))
        if any(v > 0 for v in bd.values()):
            self.logger.info("Final hard constraint breakdown:")
            for name, cnt in sorted(bd.items(), key=lambda x: -x[1]):
                if cnt > 0:
                    self.logger.info("  %s: %d", name, cnt)

        sbd = self._get_soft_breakdown(list(final_best))
        self.logger.info("Final soft constraint breakdown:")
        for name, val in sorted(sbd.items(), key=lambda x: -x[1]):
            if val > 0:
                self.logger.info("  %s: %.1f", name, val)

        return pop, stats
