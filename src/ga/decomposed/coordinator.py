"""Decomposed Scheduler: Master coordinator for Supergroup-decomposed GA + CP-SAT.

This is the main entry point for the new architecture:

    1. Build cluster contexts (one-time setup)
    2. Generate initial population (full-chromosome, then partition)
    3. GA evolution loop:
       a. Standard NSGA-II operators (select, crossover, mutate)
       b. Evaluate fitness
       c. CP-SAT polish on elite individuals (per-cluster decomposed)
    4. Final CP-SAT full optimization pass
    5. Return best schedule

The key insight: GA handles diversity and global structure exploration,
while CP-SAT handles hard constraint satisfaction within each cluster.
This division of labor plays to each algorithm's strengths.
"""

from __future__ import annotations

import copy
import logging
import random
import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deap import base, tools
from rich.table import Table

from src.constraints import Evaluator
from src.domain.timetable import Timetable
from src.ga.core.creator_registry import get_creator
from src.ga.core.evaluator import evaluate
from src.ga.core.individual import create_individual
from src.ga.core.population import generate_course_group_aware_population
from src.ga.decomposed.cluster_context import (
    PartitionResult,
    build_cluster_contexts,
    partition_individual,
)
from src.ga.decomposed.cp_optimizer import ClusterCPOptimizer, CPOptimizeResult
from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import mutate_individual
from src.utils.console_service import get_console

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import Individual, SchedulingContext
    from src.io.time_system import QuantumTimeSystem

__all__ = ["DecomposedConfig", "DecomposedResult", "DecomposedScheduler"]

logger = logging.getLogger(__name__)
console = get_console()


# ── Configuration ─────────────────────────────────────────────────────────


@dataclass
class DecomposedConfig:
    """Configuration for the decomposed GA + CP-SAT scheduler.

    Attributes
    ----------
    pop_size : int
        GA population size.
    generations : int
        Number of GA generations.
    crossover_prob : float
        Crossover probability.
    mutation_prob : float
        Mutation probability.
    cp_interval : int
        Run CP-SAT polish every N generations.
    cp_elite_count : int
        Number of elite individuals to CP-optimize per interval.
    cp_timeout_bridge : float
        CP-SAT timeout for bridge genes (seconds).
    cp_timeout_cluster : float
        CP-SAT timeout per cluster (seconds).
    cp_num_workers : int
        CP-SAT internal parallelism.
    cp_soft_objective : bool
        Whether CP-SAT should also optimize soft constraints.
    cp_full_interval : int
        Run full (non-fix-only) CP optimization every N generations.
        Set to 0 to disable.
    seed : int
        Random seed for reproducibility.
    """

    pop_size: int = 30
    generations: int = 100
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    cp_interval: int = 5
    cp_elite_count: int = 3
    cp_timeout_bridge: float = 20.0
    cp_timeout_cluster: float = 15.0
    cp_timeout_chunk: float = 15.0
    cp_num_workers: int = 4
    cp_soft_objective: bool = False
    cp_full_interval: int = 20
    cp_max_chunk_size: int = 50
    seed: int = 42


@dataclass
class DecomposedResult:
    """Result of a decomposed scheduling run.

    Attributes
    ----------
    best_individual : list[SessionGene]
        The best schedule found.
    best_hard : float
        Hard violation penalty of best.
    best_soft : float
        Soft violation penalty of best.
    generations_run : int
        Number of generations actually executed.
    cluster_info : dict[str, dict]
        Per-cluster statistics.
    total_time : float
        Total wall time.
    convergence_history : list[dict]
        Per-generation metrics.
    """

    best_individual: list[SessionGene] = field(default_factory=list)
    best_hard: float = float("inf")
    best_soft: float = float("inf")
    generations_run: int = 0
    cluster_info: dict[str, dict] = field(default_factory=dict)
    total_time: float = 0.0
    convergence_history: list[dict] = field(default_factory=list)


# ── Main Scheduler ────────────────────────────────────────────────────────


class DecomposedScheduler:
    """Supergroup-decomposed GA + CP-SAT hybrid scheduler.

    Usage::

        scheduler = DecomposedScheduler(ctx, config=DecomposedConfig(
            pop_size=30, generations=100, cp_interval=5
        ))
        result = scheduler.run()

    Parameters
    ----------
    ctx : SchedulingContext
        Fully linked scheduling context.
    config : DecomposedConfig | None
        Configuration. Uses defaults if None.
    """

    def __init__(
        self,
        ctx: SchedulingContext,
        config: DecomposedConfig | None = None,
    ) -> None:
        self.ctx = ctx
        self.config = config or DecomposedConfig()
        self.evaluator = Evaluator()

        # Ensure DEAP creator is registered
        get_creator()

        # Build cluster partition (one-time)
        console.print("[cyan]Building cluster decomposition...[/cyan]")
        self.partition = build_cluster_contexts(ctx)

        # Log cluster info
        table = Table(title="Cluster Decomposition")
        table.add_column("Cluster", style="bold")
        table.add_column("Programmes")
        table.add_column("Groups", justify="right")
        table.add_column("Courses", justify="right")
        table.add_column("Instructors", justify="right")
        table.add_column("Shared Instr", justify="right")

        for cid, cc in self.partition.cluster_contexts.items():
            shared_count = len(
                cc.cluster.instructor_ids & self.partition.shared_instructor_ids
            )
            table.add_row(
                cid,
                ", ".join(sorted(cc.cluster.programmes)),
                str(len(cc.cluster.group_ids)),
                str(len(cc.sub_ctx.courses)),
                str(len(cc.sub_ctx.instructors)),
                str(shared_count),
            )
        console.print(table)

        # Build CP optimizer
        self.cp_optimizer = ClusterCPOptimizer(
            timeout_bridge=self.config.cp_timeout_bridge,
            timeout_cluster=self.config.cp_timeout_cluster,
            timeout_chunk=self.config.cp_timeout_chunk,
            num_workers=self.config.cp_num_workers,
            soft_objective=self.config.cp_soft_objective,
            max_chunk_size=self.config.cp_max_chunk_size,
        )

        # State
        self.population: list[Any] = []
        self.toolbox = base.Toolbox()

    def run(self) -> DecomposedResult:
        """Execute the full decomposed GA + CP-SAT pipeline.

        Returns
        -------
        DecomposedResult
            The best schedule and run diagnostics.
        """
        t0 = _time.monotonic()
        cfg = self.config
        random.seed(cfg.seed)

        result = DecomposedResult()

        # ── Phase 0: Setup DEAP toolbox ───────────────────────────
        self._setup_toolbox()

        # ── Phase 1: Initial population ───────────────────────────
        console.print(
            f"\n[cyan]Generating initial population (n={cfg.pop_size})...[/cyan]"
        )
        self.population = self.toolbox.population(n=cfg.pop_size)

        # Evaluate initial population
        self._evaluate_population(self.population)

        best = tools.selBest(self.population, 1)[0]
        console.print(
            f"  Initial best: hard={best.fitness.values[0]:.0f}, "
            f"soft={best.fitness.values[1]:.0f}"
        )

        # ── Phase 1.5: CP-SAT polish on initial population ───────
        console.print("[cyan]CP-SAT initial polish...[/cyan]")
        self._cp_polish_elite(
            n_elite=min(cfg.cp_elite_count, len(self.population)),
            fix_only=True,
        )

        best = tools.selBest(self.population, 1)[0]
        console.print(
            f"  After CP polish: hard={best.fitness.values[0]:.0f}, "
            f"soft={best.fitness.values[1]:.0f}"
        )

        # ── Phase 2: GA evolution loop ────────────────────────────
        console.print(
            f"\n[bold cyan]Starting GA evolution "
            f"({cfg.generations} generations)...[/bold cyan]\n"
        )

        stagnation_counter = 0
        best_hard_ever = abs(best.fitness.values[0])

        for gen in range(cfg.generations):
            gen_t0 = _time.monotonic()

            # ── Selection ─────────────────────────────────────────
            offspring = tools.selNSGA2(self.population, len(self.population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # ── Crossover ─────────────────────────────────────────
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < cfg.crossover_prob:
                    child1, child2 = crossover_course_group_aware(
                        offspring[i], offspring[i + 1], cx_prob=1.0
                    )
                    offspring[i][:] = child1
                    offspring[i + 1][:] = child2
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # ── Mutation ──────────────────────────────────────────
            for ind in offspring:
                if random.random() < cfg.mutation_prob:
                    mutated = mutate_individual(ind, context=self.ctx, mut_prob=1.0)
                    ind[:] = mutated[0]
                    del ind.fitness.values

            # ── Evaluate invalid individuals ──────────────────────
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate_population(invalids)

            # ── (μ + λ) replacement ───────────────────────────────
            self.population = tools.selNSGA2(self.population + offspring, cfg.pop_size)

            # ── CP-SAT polish (periodic) ──────────────────────────
            cp_applied = False
            if (gen + 1) % cfg.cp_interval == 0:
                fix_only = True
                if cfg.cp_full_interval > 0 and (gen + 1) % cfg.cp_full_interval == 0:
                    fix_only = False
                self._cp_polish_elite(
                    n_elite=cfg.cp_elite_count,
                    fix_only=fix_only,
                )
                cp_applied = True

            # ── Track metrics ─────────────────────────────────────
            best = tools.selBest(self.population, 1)[0]
            current_hard = abs(best.fitness.values[0])
            current_soft = abs(best.fitness.values[1])
            gen_time = _time.monotonic() - gen_t0

            # Stagnation tracking
            if current_hard < best_hard_ever:
                best_hard_ever = current_hard
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Feasibility rate
            feasible = sum(1 for ind in self.population if ind.fitness.values[0] == 0)
            feas_rate = feasible / len(self.population) * 100

            result.convergence_history.append(
                {
                    "gen": gen,
                    "hard": current_hard,
                    "soft": current_soft,
                    "feasible_rate": feas_rate,
                    "cp_applied": cp_applied,
                    "time": gen_time,
                }
            )

            # Log every generation
            cp_marker = " [CP]" if cp_applied else ""
            console.print(
                f"  Gen {gen:3d}: "
                f"hard={current_hard:6.0f}  soft={current_soft:6.0f}  "
                f"feasible={feas_rate:5.1f}%  "
                f"stag={stagnation_counter:2d}  "
                f"t={gen_time:.1f}s{cp_marker}"
            )

            # Early termination: all feasible and stable
            if current_hard == 0 and stagnation_counter > 20:
                console.print(
                    f"\n[green]Converged at generation {gen} "
                    f"(feasible + 20 gens stability)[/green]"
                )
                break

            # Hypermutation on stagnation
            if stagnation_counter > 0 and stagnation_counter % 15 == 0:
                self._hypermutation(rate=0.3)

        # ── Phase 3: Final CP-SAT optimization ────────────────────
        console.print("\n[bold cyan]Final CP-SAT optimization pass...[/bold cyan]")
        best = tools.selBest(self.population, 1)[0]
        final_result = self.cp_optimizer.optimize_individual(
            list(best), self.ctx, self.partition, fix_only=False
        )

        if final_result.hard_after <= abs(best.fitness.values[0]):
            best[:] = final_result.genes
            # Re-evaluate
            fit = evaluate(
                list(best),
                courses=self.ctx.courses,
                instructors=self.ctx.instructors,
                groups=self.ctx.groups,
                rooms=self.ctx.rooms,
            )
            best.fitness.values = fit

        # ── Build result ──────────────────────────────────────────
        final_best = tools.selBest(self.population, 1)[0]
        result.best_individual = list(final_best)
        result.best_hard = abs(final_best.fitness.values[0])
        result.best_soft = abs(final_best.fitness.values[1])
        result.generations_run = len(result.convergence_history)
        result.total_time = _time.monotonic() - t0

        # Cluster info
        for cid, cc in self.partition.cluster_contexts.items():
            result.cluster_info[cid] = {
                "programmes": sorted(cc.cluster.programmes),
                "groups": len(cc.cluster.group_ids),
                "courses": len(cc.sub_ctx.courses),
                "instructors": len(cc.sub_ctx.instructors),
            }

        # Final summary
        self._print_summary(result)

        return result

    # ── Internal Methods ──────────────────────────────────────────────────

    def _setup_toolbox(self) -> None:
        """Configure DEAP toolbox with operators."""
        self.toolbox.register(
            "population",
            generate_course_group_aware_population,
            context=self.ctx,
        )
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register(
            "evaluate",
            evaluate,
            courses=self.ctx.courses,
            instructors=self.ctx.instructors,
            groups=self.ctx.groups,
            rooms=self.ctx.rooms,
        )

    def _evaluate_population(self, individuals: list[Any]) -> None:
        """Evaluate fitness for a list of individuals."""
        for ind in individuals:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)

    def _cp_polish_elite(
        self,
        n_elite: int,
        fix_only: bool = True,
    ) -> None:
        """Apply CP-SAT optimization to the top-N elite individuals.

        If CP-SAT produces a better individual, it replaces the original
        in the population.
        """
        elites = tools.selBest(self.population, n_elite)

        for ind in elites:
            before_hard = abs(ind.fitness.values[0])

            cp_result = self.cp_optimizer.optimize_individual(
                list(ind), self.ctx, self.partition, fix_only=fix_only
            )

            # Accept if hard violations improved (or equal hard + better soft)
            if cp_result.hard_after < before_hard:
                ind[:] = cp_result.genes
                del ind.fitness.values
            elif cp_result.hard_after == before_hard:
                # Check soft improvement
                new_fit = evaluate(
                    cp_result.genes,
                    courses=self.ctx.courses,
                    instructors=self.ctx.instructors,
                    groups=self.ctx.groups,
                    rooms=self.ctx.rooms,
                )
                if new_fit[1] < abs(ind.fitness.values[1]):
                    ind[:] = cp_result.genes
                    del ind.fitness.values

        # Re-evaluate any modified individuals
        self._evaluate_population(elites)

    def _hypermutation(self, rate: float = 0.3) -> None:
        """Replace worst individuals with hypermutated versions on stagnation."""
        n_replace = max(1, int(rate * len(self.population)))
        worst = tools.selWorst(self.population, n_replace)

        console.print(
            f"  [yellow]Stagnation: hypermutating {n_replace} individuals[/yellow]"
        )

        for ind in worst:
            # Apply extra mutation passes
            for _ in range(3):
                mutated = mutate_individual(ind, context=self.ctx, mut_prob=1.0)
                ind[:] = mutated[0]
            del ind.fitness.values

        self._evaluate_population(worst)

    def _print_summary(self, result: DecomposedResult) -> None:
        """Print a final summary table."""
        console.print("\n" + "=" * 60)
        console.print("[bold green]DECOMPOSED GA + CP-SAT — FINAL RESULTS[/bold green]")
        console.print("=" * 60)

        # Constraint breakdown
        best_tt = Timetable.from_individual(result.best_individual, self.ctx)
        _, _, hard_bd, soft_bd = self.evaluator.evaluate_all(best_tt)

        table = Table(title="Constraint Breakdown")
        table.add_column("Constraint", style="bold")
        table.add_column("Type")
        table.add_column("Penalty", justify="right")

        for name, val in sorted(hard_bd.items()):
            style = "red" if val > 0 else "green"
            table.add_row(name, "HARD", f"[{style}]{val:.0f}[/{style}]")
        for name, val in sorted(soft_bd.items()):
            style = "yellow" if val > 0 else "green"
            table.add_row(name, "soft", f"[{style}]{val:.1f}[/{style}]")

        console.print(table)

        console.print(f"\n  [bold]Hard violations:[/bold] {result.best_hard:.0f}")
        console.print(f"  [bold]Soft penalty:[/bold]    {result.best_soft:.1f}")
        console.print(f"  [bold]Generations:[/bold]     {result.generations_run}")
        console.print(f"  [bold]Total time:[/bold]      {result.total_time:.1f}s")

        # Per-cluster summary
        table2 = Table(title="Per-Cluster Summary")
        table2.add_column("Cluster")
        table2.add_column("Programmes")
        table2.add_column("Groups", justify="right")
        table2.add_column("Courses", justify="right")

        for cid, info in sorted(result.cluster_info.items()):
            table2.add_row(
                cid,
                ", ".join(info["programmes"]),
                str(info["groups"]),
                str(info["courses"]),
            )
        console.print(table2)
        console.print()
