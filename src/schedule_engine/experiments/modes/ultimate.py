"""
UltimateExperiment — ILS + Full Repair Arsenal (Mode F).

Combines ALL available repair infrastructure into an Iterated Local Search
pipeline designed to push hard constraint violations as close to zero as
possible:

  Phase 1: Multi-start initialisation
           Generate ``n_starts`` individuals with conflict-aware init, apply
           iterative deterministic-repair + gene-level local-search, keep the
           best.

  Phase 2: Iterated Local Search (ILS)
           Perturb → Repair → Gene-LS → RepairEngine → Accept if improved.
           Greedy acceptance (only accept strict improvements).
           Periodic group & instructor rescheduling (ruin-and-recreate) on
           stagnation every 10 non-improving iterations.
           Warm + fresh diversification restart on prolonged stagnation.

Repair arsenal used:
  • ``repair_individual_unified`` — 11 registered deterministic operators
  • ``optimize_gene_greedy`` — per-gene greedy hill climbing with
    time + room + instructor + combined neighbourhoods
  • ``RepairEngine`` — evaluation-guided MoveTime / SwapRoom /
    ReassignInstructor with epsilon-greedy policy
  • ``group_reschedule_pass`` — ruin-and-recreate for worst conflicted groups
  • ``instructor_reschedule_pass`` — ruin-and-recreate for worst instructors

Usage::

    from schedule_engine.experiments.modes.ultimate import UltimateExperiment

    exp = UltimateExperiment(
        seed=42, pop_size=100, ngen=500,
        fitness_weights=(-1.0, -1.0),
    )
    exp.run()
"""

from __future__ import annotations

import collections
import copy
import logging
import math
import random
import time
from typing import Any

from schedule_engine.experiments.base import BaseExperiment
from schedule_engine.ga.run_helpers import EvolutionStats

logger = logging.getLogger(__name__)


class UltimateExperiment(BaseExperiment):
    """
    ILS with the full repair arsenal (Mode F).

    Additional Parameters
    ---------------------
    n_starts : int
        Number of multi-start individuals for initial solution (default 5).
    repair_ls_rounds : int
        Number of iterative deterministic-repair + gene-LS rounds per start
        individual (default 5).
    ils_iterations : int
        Number of ILS perturb-repair-accept iterations (default 200).
    perturb_frac : float
        Fraction of current best Hard to use as perturbation size (default 0.15).
    perturb_min : int
        Minimum number of genes to perturb (default 10).
    engine_max_steps : int
        Max steps for the evaluation-guided RepairEngine (default 20).
    engine_budget_ms : float
        Time budget (ms) for RepairEngine per ILS iteration (default 500).
    engine_max_candidates : int
        Max candidate moves per RepairEngine step (default 40).
    engine_policy : str
        RepairEngine policy: ``"round_robin"`` or ``"epsilon_greedy"``.
    engine_epsilon : float
        Exploration rate for epsilon-greedy policy (default 0.15).
    deterministic_max_iters : int
        Max iterations for deterministic repair pass (default 2).
    ls_max_iters : int
        Max greedy iterations per gene in local search (default 10).
    stagnation_restart : int
        Trigger diversification restart after this many iterations without
        improvement (default 50).
    """

    def __init__(
        self,
        *,
        # Multi-start
        n_starts: int = 5,
        repair_ls_rounds: int = 5,
        # ILS
        ils_iterations: int = 300,
        perturb_frac: float = 0.15,
        perturb_min: int = 10,
        # RepairEngine
        engine_max_steps: int = 20,
        engine_budget_ms: float = 500.0,
        engine_max_candidates: int = 50,
        engine_policy: str = "epsilon_greedy",
        engine_epsilon: float = 0.15,
        # Deterministic repair
        deterministic_max_iters: int = 3,
        # Gene-level local search
        ls_max_iters: int = 12,
        # Diversification
        stagnation_restart: int = 30,
        # Parent
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_starts = n_starts
        self.repair_ls_rounds = repair_ls_rounds
        self.ils_iterations = ils_iterations
        self.perturb_frac = perturb_frac
        self.perturb_min = perturb_min
        self.engine_max_steps = engine_max_steps
        self.engine_budget_ms = engine_budget_ms
        self.engine_max_candidates = engine_max_candidates
        self.engine_policy = engine_policy
        self.engine_epsilon = engine_epsilon
        self.deterministic_max_iters = deterministic_max_iters
        self.ls_max_iters = ls_max_iters
        self.stagnation_restart = stagnation_restart

        # Tracking — aggregates
        self._total_deterministic_fixes: int = 0
        self._total_engine_fixes: int = 0
        self._total_ls_fixes: int = 0
        self._ils_improvements: int = 0
        self._restarts: int = 0

        # Tracking — per-iteration (for ILS plots)
        self._iter_ids: list[int] = []
        self._iter_best_hard: list[float] = []
        self._iter_best_soft: list[float] = []
        self._iter_cand_hard: list[float] = []
        self._iter_det_fixes: list[int] = []
        self._iter_ls_delta: list[int] = []
        self._iter_engine_steps: list[int] = []
        self._iter_times: list[float] = []
        self._iter_perturb_sizes: list[int] = []
        self._iter_constraint_history: dict[str, list[float]] = {}
        self._improvement_iters: list[int] = []
        self._restart_iters: list[int] = []
        self._reschedule_events: list[dict[str, Any]] = []
        self._improvement_events: list[dict[str, Any]] = []
        self._phase1_hard: float = 0.0

    def _get_experiment_name(self) -> str:
        return "ga_06_ultimate"

    def _get_extra_config(self) -> dict[str, Any]:
        return {
            "n_starts": self.n_starts,
            "repair_ls_rounds": self.repair_ls_rounds,
            "ils_iterations": self.ils_iterations,
            "perturb_frac": self.perturb_frac,
            "perturb_min": self.perturb_min,
            "engine_max_steps": self.engine_max_steps,
            "engine_budget_ms": self.engine_budget_ms,
            "engine_max_candidates": self.engine_max_candidates,
            "engine_policy": self.engine_policy,
            "engine_epsilon": self.engine_epsilon,
            "deterministic_max_iters": self.deterministic_max_iters,
            "ls_max_iters": self.ls_max_iters,
            "stagnation_restart": self.stagnation_restart,
        }

    def _get_extra_results(self) -> dict[str, Any]:
        return {
            "total_deterministic_fixes": self._total_deterministic_fixes,
            "total_engine_fixes": self._total_engine_fixes,
            "total_ls_fixes": self._total_ls_fixes,
            "ils_improvements": self._ils_improvements,
            "restarts": self._restarts,
        }

    def _create_exporter(self) -> Any:
        """Override to use ILS-aware exporter that generates ILS plots."""
        from schedule_engine.experiments.output.ils import ILSExporter

        return ILSExporter(
            output_dir=self.output_dir,
            data=self.data,
            logger=self.logger,
            ils_data=self._get_ils_plot_data(),
        )

    def _get_ils_plot_data(self) -> dict[str, Any]:
        """Collect all ILS tracking data for the exporter."""
        return {
            "iterations": self._iter_ids,
            "best_hard": self._iter_best_hard,
            "best_soft": self._iter_best_soft,
            "candidate_hard": self._iter_cand_hard,
            "det_fixes": self._iter_det_fixes,
            "ls_delta": self._iter_ls_delta,
            "engine_steps": self._iter_engine_steps,
            "iter_times": self._iter_times,
            "perturb_sizes": self._iter_perturb_sizes,
            "constraint_history": self._iter_constraint_history,
            "improvement_iters": self._improvement_iters,
            "restart_iters": self._restart_iters,
            "reschedule_events": self._reschedule_events,
            "improvement_events": self._improvement_events,
            "phase1_hard": self._phase1_hard,
        }

    # ------------------------------------------------------------------
    # Constraint breakdown helper
    # ------------------------------------------------------------------

    def _get_hard_breakdown(self, ind: list) -> dict[str, int]:
        """Return per-constraint hard violation counts for *ind*."""
        from schedule_engine.constraints.evaluator import Evaluator
        from schedule_engine.domain.timetable import Timetable

        tt = Timetable(genes=ind, context=self.data.context)
        ev = Evaluator()
        return {c.name: int(c.weight * c.evaluate(tt)) for c in ev.hard}

    # ------------------------------------------------------------------

    def _run_evolution(self) -> tuple[list[Any], EvolutionStats]:
        from schedule_engine.ga.operators.local_search import optimize_gene_greedy
        from schedule_engine.ga.repair.basic import repair_individual_unified
        from schedule_engine.ga.repair.engine import (
            RepairEngine,
            _build_counts,
            _gene_violation_score,
        )
        from schedule_engine.utils.room_compatibility import is_room_suitable_for_course

        start_time = time.time()
        rng = random.Random(self.seed)
        stats = EvolutionStats()

        # ── helper: gene-level LS pass ────────────────────────────────
        def gene_ls_pass(ind: list) -> int:
            """Single-pass gene LS — fast iteration for ILS throughput."""
            inst_c, room_c, group_c = _build_counts(ind)
            violated = []
            for i, g in enumerate(ind):
                score = _gene_violation_score(
                    g,
                    self.data.context,
                    inst_c,
                    room_c,
                    group_c,
                )
                if score > 0:
                    violated.append((i, score))
            violated.sort(key=lambda x: -x[1])
            total_delta = 0
            for idx, _ in violated:
                improved, delta = optimize_gene_greedy(
                    ind[idx],
                    ind,
                    idx,
                    self.data.context,
                    max_iterations=self.ls_max_iters,
                )
                if delta > 0:
                    ind[idx] = improved
                    total_delta += delta
            return total_delta

        # ── helper: smart perturbation ────────────────────────────────
        def smart_perturb(ind: list, n_perturb: int) -> None:
            inst_c, room_c, group_c = _build_counts(ind)
            violated = []
            for i, g in enumerate(ind):
                score = _gene_violation_score(
                    g,
                    self.data.context,
                    inst_c,
                    room_c,
                    group_c,
                )
                if score > 0:
                    violated.append((i, score))
            if not violated:
                return
            violated.sort(key=lambda x: -x[1])
            targets = violated[:n_perturb]
            avail = sorted(self.data.context.available_quanta)
            max_avail = max(avail)

            # Track already-perturbed times to avoid intra-perturbation conflicts
            perturbed_times: dict[str, set[int]] = collections.defaultdict(set)

            for idx, _ in targets:
                g = ind[idx]
                dur = g.num_quanta
                course_key = (g.course_id, g.course_type)

                # Build blocked times for this gene's groups
                # Include both existing schedule AND already-perturbed genes
                group_blocked: set[int] = set()
                for other_i, other in enumerate(ind):
                    if other_i == idx:
                        continue
                    if set(g.group_ids) & set(other.group_ids):
                        for q in range(other.start_quanta, other.end_quanta):
                            group_blocked.add(q)
                # Also add times from already-perturbed genes of same groups
                for gid in g.group_ids:
                    group_blocked |= perturbed_times.get(gid, set())

                # Find valid starts avoiding group conflicts
                valid_starts = []
                for start_q in avail:
                    if start_q + dur > max_avail + 1:
                        continue
                    slot = set(range(start_q, start_q + dur))
                    if not slot & group_blocked:
                        valid_starts.append(start_q)

                if valid_starts:
                    g.start_quanta = rng.choice(valid_starts)
                else:
                    candidates = [q for q in avail if q + dur - 1 <= max_avail]
                    if candidates:
                        g.start_quanta = min(
                            candidates,
                            key=lambda sq: len(
                                set(range(sq, sq + dur)) & group_blocked
                            ),
                        )

                # Record perturbed time for intra-perturbation tracking
                new_slot = set(range(g.start_quanta, g.start_quanta + dur))
                for gid in g.group_ids:
                    perturbed_times[gid] |= new_slot

                # Random suitable room
                course = self.data.context.courses.get(course_key)
                if course:
                    req = (
                        str(getattr(course, "required_room_features", "lecture"))
                        .lower()
                        .strip()
                    )
                    suitable = [
                        r.room_id
                        for r in self.data.context.rooms.values()
                        if is_room_suitable_for_course(
                            req,
                            str(getattr(r, "room_features", "lecture")).lower().strip(),
                            getattr(course, "specific_lab_features", None),
                            getattr(r, "specific_features", None),
                        )
                    ]
                    if suitable:
                        g.room_id = rng.choice(suitable)

                # Fix instructor availability at new time
                instructor = self.data.context.instructors.get(g.instructor_id)
                if instructor and not instructor.is_full_time:
                    avail_ok = all(
                        q in instructor.available_quanta
                        for q in range(g.start_quanta, g.start_quanta + dur)
                    )
                    if not avail_ok:
                        for cand in self.data.context.instructors.values():
                            q_courses = getattr(cand, "qualified_courses", set())
                            if (
                                course_key not in q_courses
                                and g.course_id not in q_courses
                            ):
                                continue
                            if cand.is_full_time or all(
                                q in cand.available_quanta
                                for q in range(g.start_quanta, g.start_quanta + dur)
                            ):
                                g.instructor_id = cand.instructor_id
                                break

        # ── helper: group-cluster scatter ─────────────────────────────
        def group_scatter_perturb(ind: list) -> None:
            """Find the most-conflicted group and reschedule all its sessions."""
            # Count per-group time-overlap conflicts
            group_genes: dict[str, list[int]] = collections.defaultdict(list)
            for i, g in enumerate(ind):
                for gid in g.group_ids:
                    group_genes[gid].append(i)

            # Find group with most time-overlapping violations
            worst_group = None
            worst_conflicts = 0
            for gid, gene_idxs in group_genes.items():
                conflicts = 0
                # Check pairwise time overlaps
                for a_pos in range(len(gene_idxs)):
                    for b_pos in range(a_pos + 1, len(gene_idxs)):
                        ga = ind[gene_idxs[a_pos]]
                        gb = ind[gene_idxs[b_pos]]
                        if (
                            ga.start_quanta < gb.end_quanta
                            and gb.start_quanta < ga.end_quanta
                        ):
                            conflicts += 1
                if conflicts > worst_conflicts:
                    worst_conflicts = conflicts
                    worst_group = gid

            if worst_group is None or worst_conflicts == 0:
                return

            target_idxs = group_genes[worst_group]
            avail = sorted(self.data.context.available_quanta)
            max_avail = max(avail)

            # Collect OTHER sessions that share groups with these genes
            # (beyond the worst_group itself)
            other_blocked: dict[int, set[int]] = {}
            for idx in target_idxs:
                blocked = set()
                g = ind[idx]
                for other_i, other in enumerate(ind):
                    if other_i in target_idxs:
                        continue
                    if set(g.group_ids) & set(other.group_ids):
                        for q in range(other.start_quanta, other.end_quanta):
                            blocked.add(q)
                other_blocked[idx] = blocked

            # Greedily assign non-overlapping times
            rng.shuffle(target_idxs)
            assigned_times: set[int] = set()

            for idx in target_idxs:
                g = ind[idx]
                dur = g.num_quanta
                blocked = assigned_times | other_blocked.get(idx, set())

                valid = []
                for sq in avail:
                    if sq + dur > max_avail + 1:
                        continue
                    slot = set(range(sq, sq + dur))
                    if not slot & blocked:
                        valid.append(sq)

                if valid:
                    chosen = rng.choice(valid)
                else:
                    candidates = [q for q in avail if q + dur - 1 <= max_avail]
                    if candidates:
                        chosen = min(
                            candidates,
                            key=lambda sq: len(set(range(sq, sq + dur)) & blocked),
                        )
                    else:
                        continue

                g.start_quanta = chosen
                assigned_times |= set(range(chosen, chosen + dur))

        # ── helper: init a fresh individual ───────────────────────────
        def make_fresh_individual(seed_offset: int) -> tuple[list, float, float]:
            """Generate and repair a new individual from scratch."""
            random.seed(self.seed + seed_offset + 1000)
            pop = self.create_initial_population()
            ind = list(pop[0])
            for _ in range(self.repair_ls_rounds):
                det_stats = repair_individual_unified(
                    ind,
                    self.data.context,
                    selective=True,
                    max_iterations=self.deterministic_max_iters,
                )
                fixes = (
                    det_stats.get("total_fixes", 0)
                    if isinstance(det_stats, dict)
                    else 0
                )
                self._total_deterministic_fixes += fixes
                ls_delta = gene_ls_pass(ind)
                self._total_ls_fixes += ls_delta
            h, s = self.evaluate(ind)
            return ind, h, s

        # ── helper: group rescheduling pass ───────────────────────────
        def group_reschedule_pass(ind: list, n_groups: int = 3) -> int:
            """Ruin-and-recreate: remove and re-insert sessions for
            the most-conflicted groups.

            1. For each of the ``n_groups`` worst groups, collect all
               gene indices.
            2. For each gene in that group, "blank" it (save + remove
               from occupancy).
            3. Greedily re-insert each gene at the time with the fewest
               total conflicts, considering already-placed genes.

            Returns the total number of genes that were moved.
            """
            avail = sorted(self.data.context.available_quanta)
            if not avail:
                return 0
            max_avail = max(avail)

            # Build group → gene-index map
            group_genes: dict[str, list[int]] = collections.defaultdict(list)
            for i, g in enumerate(ind):
                for gid in g.group_ids:
                    group_genes[gid].append(i)

            # Count per-group time-overlap violations
            group_conflicts: dict[str, int] = {}
            for gid, gene_idxs in group_genes.items():
                conflicts = 0
                for a_pos in range(len(gene_idxs)):
                    for b_pos in range(a_pos + 1, len(gene_idxs)):
                        ga = ind[gene_idxs[a_pos]]
                        gb = ind[gene_idxs[b_pos]]
                        if (
                            ga.start_quanta < gb.end_quanta
                            and gb.start_quanta < ga.end_quanta
                        ):
                            conflicts += 1
                if conflicts > 0:
                    group_conflicts[gid] = conflicts

            if not group_conflicts:
                return 0

            # Pick the n_groups worst
            worst = sorted(group_conflicts.items(), key=lambda x: -x[1])[:n_groups]
            total_moves = 0

            for gid, _ in worst:
                target_idxs = list(set(group_genes[gid]))  # deduplicate
                rng.shuffle(target_idxs)

                # Build occupancy of NON-target genes for this group
                # (what's "fixed" while we re-insert)
                target_set = set(target_idxs)

                # Save original positions
                saved = {idx: ind[idx].start_quanta for idx in target_idxs}

                # Greedy re-insertion: for each gene, pick time with
                # minimum conflict score (group + instructor + room)
                placed_times: dict[str, set[int]] = collections.defaultdict(set)

                for idx in target_idxs:
                    g = ind[idx]
                    dur = g.num_quanta

                    # Build blocked times from other genes AND already-placed
                    group_blocked: set[int] = set()
                    for other_i, other in enumerate(ind):
                        if other_i in target_set:
                            continue
                        if set(g.group_ids) & set(other.group_ids):
                            for q in range(other.start_quanta, other.end_quanta):
                                group_blocked.add(q)
                    # Add already-placed targets
                    for ggid in g.group_ids:
                        group_blocked |= placed_times.get(ggid, set())

                    instr_blocked: set[int] = set()
                    for other_i, other in enumerate(ind):
                        if other_i in target_set:
                            continue
                        if other.instructor_id == g.instructor_id:
                            for q in range(other.start_quanta, other.end_quanta):
                                instr_blocked.add(q)

                    # Score each candidate start
                    best_sq = None
                    best_score = float("inf")
                    for sq in avail:
                        if sq + dur > max_avail + 1:
                            continue
                        slot = set(range(sq, sq + dur))
                        score = len(slot & group_blocked) * 3 + len(
                            slot & instr_blocked
                        )
                        if score < best_score:
                            best_score = score
                            best_sq = sq
                            if score == 0:
                                break  # Perfect slot

                    if best_sq is not None:
                        g.start_quanta = best_sq
                        new_slot = set(range(best_sq, best_sq + dur))
                        for ggid in g.group_ids:
                            placed_times[ggid] |= new_slot
                        if g.start_quanta != saved[idx]:
                            total_moves += 1

            return total_moves

        # ── helper: instructor rescheduling pass ──────────────────────
        def instructor_reschedule_pass(ind: list, n_instr: int = 3) -> int:
            """Ruin-and-recreate for the most-conflicted instructors.

            For each of the ``n_instr`` worst instructors:
            1. Collect all gene indices taught by this instructor.
            2. Greedily re-assign each gene to a time slot that:
               - Falls within instructor availability (for part-time)
               - Doesn't overlap with other sessions of this instructor
               - Minimises group overlap conflicts
            3. Optionally swap instructor if no good slot exists.

            Returns the number of genes moved.
            """
            avail = sorted(self.data.context.available_quanta)
            if not avail:
                return 0
            max_avail = max(avail)

            # Build instructor → gene-index map
            instr_genes: dict[str, list[int]] = collections.defaultdict(list)
            for i, g in enumerate(ind):
                instr_genes[g.instructor_id].append(i)

            # Count per-instructor violations:
            # (a) time-overlap (exclusivity) and (b) availability
            instr_conflicts: dict[str, int] = {}
            for iid, gene_idxs in instr_genes.items():
                conflicts = 0
                instructor = self.data.context.instructors.get(iid)
                for a_pos in range(len(gene_idxs)):
                    ga = ind[gene_idxs[a_pos]]
                    # Check availability
                    if instructor and not instructor.is_full_time:
                        for q in range(ga.start_quanta, ga.end_quanta):
                            if q not in instructor.available_quanta:
                                conflicts += 1
                    # Check pairwise time overlaps
                    for b_pos in range(a_pos + 1, len(gene_idxs)):
                        gb = ind[gene_idxs[b_pos]]
                        if (
                            ga.start_quanta < gb.end_quanta
                            and gb.start_quanta < ga.end_quanta
                        ):
                            conflicts += 1
                if conflicts > 0:
                    instr_conflicts[iid] = conflicts

            if not instr_conflicts:
                return 0

            worst = sorted(instr_conflicts.items(), key=lambda x: -x[1])[:n_instr]
            total_moves = 0

            for iid, _ in worst:
                target_idxs = list(set(instr_genes[iid]))
                rng.shuffle(target_idxs)
                target_set = set(target_idxs)
                instructor = self.data.context.instructors.get(iid)

                # Build instructor availability set
                if instructor and not instructor.is_full_time:
                    instr_avail = set(instructor.available_quanta)
                else:
                    instr_avail = set(avail)  # full-time = always available

                saved = {idx: ind[idx].start_quanta for idx in target_idxs}

                # Track placed instructor times to avoid self-overlap
                placed_instr_times: set[int] = set()
                # Collect times of this instructor's OTHER genes (not in target)
                for oi, og in enumerate(ind):
                    if oi not in target_set and og.instructor_id == iid:
                        for q in range(og.start_quanta, og.end_quanta):
                            placed_instr_times.add(q)

                for idx in target_idxs:
                    g = ind[idx]
                    dur = g.num_quanta

                    # Group blocked times (from non-target genes)
                    group_blocked: set[int] = set()
                    for oi, og in enumerate(ind):
                        if oi in target_set:
                            continue
                        if set(g.group_ids) & set(og.group_ids):
                            for q in range(og.start_quanta, og.end_quanta):
                                group_blocked.add(q)

                    # Score each candidate start
                    best_sq = None
                    best_score = float("inf")
                    for sq in avail:
                        if sq + dur > max_avail + 1:
                            continue
                        slot = set(range(sq, sq + dur))
                        # Hard: instructor must be available at all quanta
                        unavail_count = len(slot - instr_avail)
                        # Hard: no instructor self-overlap
                        self_overlap = len(slot & placed_instr_times)
                        # Soft: group overlap
                        grp_overlap = len(slot & group_blocked)
                        score = unavail_count * 5 + self_overlap * 4 + grp_overlap * 2
                        if score < best_score:
                            best_score = score
                            best_sq = sq
                            if score == 0:
                                break

                    if best_sq is not None and best_sq != saved[idx]:
                        g.start_quanta = best_sq
                        total_moves += 1

                    # Update placed times
                    new_slot = set(range(g.start_quanta, g.start_quanta + dur))
                    placed_instr_times |= new_slot

            return total_moves

        # ══════════════════════════════════════════════════════════════
        # PHASE 1: Multi-start initialisation
        # ══════════════════════════════════════════════════════════════
        self.logger.info(
            "Phase 1: Multi-start init (%d starts, %d repair+LS rounds each)",
            self.n_starts,
            self.repair_ls_rounds,
        )
        best_h = float("inf")
        best_s = float("inf")
        best_ind: list | None = None

        for start_idx in range(self.n_starts):
            random.seed(self.seed + start_idx)
            pop = self.create_initial_population()
            ind = list(pop[0])

            for _ in range(self.repair_ls_rounds):
                det_stats = repair_individual_unified(
                    ind,
                    self.data.context,
                    selective=True,
                    max_iterations=self.deterministic_max_iters,
                )
                fixes = (
                    det_stats.get("total_fixes", 0)
                    if isinstance(det_stats, dict)
                    else 0
                )
                self._total_deterministic_fixes += fixes
                ls_delta = gene_ls_pass(ind)
                self._total_ls_fixes += ls_delta

            h, s = self.evaluate(ind)
            self.logger.info(
                "Start %d/%d: Hard=%.0f Soft=%.0f (%.1fs)",
                start_idx + 1,
                self.n_starts,
                h,
                s,
                time.time() - start_time,
            )
            if (h, s) < (best_h, best_s):
                best_h, best_s = h, s
                best_ind = copy.deepcopy(ind)

        assert best_ind is not None
        self.logger.info(
            "Best initial: Hard=%.0f Soft=%.0f (%.1fs)",
            best_h,
            best_s,
            time.time() - start_time,
        )
        self._phase1_hard = best_h

        # Log initial constraint breakdown
        breakdown = self._get_hard_breakdown(best_ind)
        for name, cnt in sorted(breakdown.items(), key=lambda x: -x[1]):
            if cnt > 0:
                self.logger.info("  %s: %d", name, cnt)

        # ══════════════════════════════════════════════════════════════
        # PHASE 2: Greedy ILS with smart perturbation + group repair
        # ══════════════════════════════════════════════════════════════
        repair_engine = RepairEngine(
            context=self.data.context,
            evaluator=self.evaluate,
            policy=self.engine_policy,
            max_steps=self.engine_max_steps,
            max_candidates=self.engine_max_candidates,
            budget_ms=self.engine_budget_ms,
            epsilon=self.engine_epsilon,
            rng=rng,
            logger=self.logger,
            log_steps=False,
            log_candidates=False,
        )

        random.seed(self.seed)
        no_improve_count = 0

        for ils_iter in range(self.ils_iterations):
            iter_start = time.time()

            # ── perturb from best (greedy ILS) ────────────────────────
            candidate = copy.deepcopy(best_ind)
            n_perturb = max(self.perturb_min, int(best_h * self.perturb_frac))
            smart_perturb(candidate, n_perturb)

            # ── deterministic repair ──────────────────────────────────
            det_stats = repair_individual_unified(
                candidate,
                self.data.context,
                selective=True,
                max_iterations=self.deterministic_max_iters,
            )
            fixes = (
                det_stats.get("total_fixes", 0) if isinstance(det_stats, dict) else 0
            )
            self._total_deterministic_fixes += fixes

            # ── gene-level local search ───────────────────────────────
            ls_delta = gene_ls_pass(candidate)
            self._total_ls_fixes += ls_delta

            # ── RepairEngine ──────────────────────────────────────────
            r_stats = repair_engine.repair_individual(
                candidate, budget_ms=self.engine_budget_ms
            )
            self._total_engine_fixes += r_stats.applied_steps

            # ── greedy acceptance ─────────────────────────────────────
            ch, cs = self.evaluate(candidate)
            improved_tag = ""
            _prev_best_h = best_h

            if (ch, cs) < (best_h, best_s):
                best_h, best_s = ch, cs
                best_ind = copy.deepcopy(candidate)
                self._ils_improvements += 1
                no_improve_count = 0
                improved_tag = " *IMPROVED*"
                self._improvement_iters.append(ils_iter + 1)
                self._improvement_events.append(
                    {
                        "iter": ils_iter + 1,
                        "delta": _prev_best_h - best_h,
                        "source": "perturb+repair",
                    }
                )
            else:
                no_improve_count += 1

            # ── periodic rescheduling on best (every 10 stagnant) ──────
            if no_improve_count > 0 and no_improve_count % 10 == 0:
                trial = copy.deepcopy(best_ind)
                resc_before_h = best_h
                # Alternate: group(10,30,...) vs instructor(20,40,...)
                if (no_improve_count // 10) % 2 == 1:
                    n_gr = 3 if no_improve_count < 20 else 5
                    group_reschedule_pass(trial, n_groups=n_gr)
                    resc_type = "group"
                else:
                    instructor_reschedule_pass(trial, n_instr=5)
                    resc_type = "instructor"
                # Full repair chain after rescheduling
                repair_individual_unified(
                    trial,
                    self.data.context,
                    selective=True,
                    max_iterations=self.deterministic_max_iters,
                )
                gene_ls_pass(trial)
                repair_engine.repair_individual(
                    trial, budget_ms=self.engine_budget_ms * 3
                )
                new_h, new_s = self.evaluate(trial)
                self._reschedule_events.append(
                    {
                        "iter": ils_iter + 1,
                        "type": resc_type,
                        "before": resc_before_h,
                        "after": new_h,
                    }
                )
                if (new_h, new_s) < (best_h, best_s):
                    best_h, best_s = new_h, new_s
                    best_ind = trial
                    self._ils_improvements += 1
                    no_improve_count = 0
                    self._improvement_iters.append(ils_iter + 1)
                    self._improvement_events.append(
                        {
                            "iter": ils_iter + 1,
                            "delta": resc_before_h - best_h,
                            "source": "rescheduling",
                        }
                    )
                    self.logger.info("  rescheduling improved to Hard=%d", best_h)

            # ── diversification restart ───────────────────────────────
            if no_improve_count >= self.stagnation_restart:
                self._restarts += 1
                self._restart_iters.append(ils_iter + 1)
                restart_before_h = best_h
                self.logger.info(
                    "Diversification restart #%d at ILS %d (stagnation=%d)",
                    self._restarts,
                    ils_iter + 1,
                    no_improve_count,
                )
                # Strategy A: fresh from scratch
                fresh_ind, fresh_h, fresh_s = make_fresh_individual(self._restarts)
                # Strategy B: warm restart — moderate perturb + reschedule + repair
                warm = copy.deepcopy(best_ind)
                heavy_n = max(20, len(warm) // 10)  # ~10% of genes
                smart_perturb(warm, heavy_n)
                group_reschedule_pass(warm, n_groups=5)
                instructor_reschedule_pass(warm, n_instr=5)
                for _ in range(2):  # 2 rounds of repair+LS
                    repair_individual_unified(
                        warm,
                        self.data.context,
                        selective=True,
                        max_iterations=self.deterministic_max_iters,
                    )
                    gene_ls_pass(warm)
                repair_engine.repair_individual(
                    warm, budget_ms=self.engine_budget_ms * 3
                )
                warm_h, warm_s = self.evaluate(warm)
                # Pick the better restart
                if (warm_h, warm_s) < (fresh_h, fresh_s):
                    restart_ind, restart_h, restart_s = warm, warm_h, warm_s
                    tag = "warm"
                else:
                    restart_ind, restart_h, restart_s = (fresh_ind, fresh_h, fresh_s)
                    tag = "fresh"
                self.logger.info(
                    "  restart: %s=%.0f fresh=%.0f warm=%.0f",
                    tag,
                    restart_h,
                    fresh_h,
                    warm_h,
                )
                # Update best if the restart is better
                if (restart_h, restart_s) < (best_h, best_s):
                    best_h, best_s = restart_h, restart_s
                    best_ind = copy.deepcopy(restart_ind)
                    self._ils_improvements += 1
                    self._improvement_iters.append(ils_iter + 1)
                    self._improvement_events.append(
                        {
                            "iter": ils_iter + 1,
                            "delta": restart_before_h - best_h,
                            "source": "restart",
                        }
                    )
                no_improve_count = 0

            # ── per-iteration tracking (for ILS plots) ────────────────
            iter_elapsed = time.time() - iter_start
            self._iter_ids.append(ils_iter + 1)
            self._iter_best_hard.append(float(best_h))
            self._iter_best_soft.append(float(best_s))
            self._iter_cand_hard.append(float(ch))
            self._iter_det_fixes.append(int(fixes))
            self._iter_ls_delta.append(int(ls_delta))
            self._iter_engine_steps.append(int(r_stats.applied_steps))
            self._iter_times.append(iter_elapsed)
            self._iter_perturb_sizes.append(n_perturb)

            # Per-constraint breakdown for this iteration
            if (
                ils_iter == 0
                or (ils_iter + 1) % max(1, self.log_interval) == 0
                or improved_tag
            ):
                bd = self._get_hard_breakdown(best_ind)
                for cname, cnt in bd.items():
                    if cname not in self._iter_constraint_history:
                        self._iter_constraint_history[cname] = []
                    self._iter_constraint_history[cname].append(float(cnt))
                # Ensure consistent length — use iteration index as alignment key
                self._iter_constraint_history.setdefault("_sample_iters", []).append(
                    ils_iter + 1
                )

            # ── logging ───────────────────────────────────────────────
            elapsed = time.time() - start_time
            if (
                (ils_iter + 1) % self.log_interval == 0
                or ils_iter == 0
                or improved_tag
                or ils_iter == self.ils_iterations - 1
            ):
                self.logger.info(
                    "ILS %3d/%d: cand=%.0f best=%.0f (%.1fs)%s",
                    ils_iter + 1,
                    self.ils_iterations,
                    ch,
                    best_h,
                    elapsed,
                    improved_tag,
                )

            # ── record stats ──────────────────────────────────────────
            from deap import base, creator

            if not hasattr(creator, "FitnessMulti"):
                creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMulti)
            fake_ind = creator.Individual(best_ind)
            fake_ind.fitness.values = (best_h, best_s)
            self.record_generation_stats([fake_ind], stats, ils_iter, iter_start)

            # Early stopping
            if best_h <= 2:
                self.logger.info(
                    "Reached structural floor (Hard=%.0f), stopping.", best_h
                )
                break

        stats.elapsed_time = time.time() - start_time

        # Build final population for output
        from deap import base, creator

        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)
        final_ind = creator.Individual(best_ind)
        final_ind.fitness.values = (best_h, best_s)

        # Log final constraint breakdown
        self.logger.info(
            "DONE: Hard=%.0f Soft=%.0f improvements=%d restarts=%d time=%.1fs",
            best_h,
            best_s,
            self._ils_improvements,
            self._restarts,
            stats.elapsed_time,
        )
        breakdown = self._get_hard_breakdown(best_ind)
        self.logger.info("Final constraint breakdown:")
        for name, cnt in sorted(breakdown.items(), key=lambda x: -x[1]):
            if cnt > 0:
                self.logger.info("  %s: %d", name, cnt)

        return [final_ind], stats
