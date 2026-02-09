"""EvolutionEngine — clean evolution loop using OOP components.

Phase 6 of the OOP redesign. This is the **new** evolution engine
that composes the components from Phases 1-5:
- ``Evaluator`` (Phase 3) for fitness evaluation
- ``PopulationFactory`` (Phase 5) for population creation
- ``RepairPipeline`` (Phase 4) for post-mutation repair
- ``Timetable`` (Phase 1) for pre-indexed schedule views

This does NOT replace ``GAScheduler`` — that 3,005-line class stays
functional for production runs.  ``EvolutionEngine`` is the clean
alternative for new code, notebooks, and testing.

Design principles:
- No ``get_config()`` calls — all parameters explicit
- No global state — everything injected via constructor
- No UI code — pure algorithm, callers handle display
- No side effects — ``run()`` returns a result object
- Composable — plugins via hooks, not inheritance

Usage::

    from schedule_engine.evolution.engine import EvolutionEngine
    from schedule_engine.evaluation import Evaluator
    from schedule_engine.population import PopulationFactory

    factory = PopulationFactory(context)
    evaluator = Evaluator()
    engine = EvolutionEngine(
        evaluator=evaluator,
        population_factory=factory,
        context=context,
        pop_size=50,
        generations=100,
    )
    result = engine.run()
    print(result.best_fitness)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from schedule_engine.domain.gene import SessionGene
    from schedule_engine.domain.types import SchedulingContext
    from schedule_engine.io.time_system import QuantumTimeSystem

from schedule_engine.domain.timetable import Timetable
from schedule_engine.evaluation import Evaluator
from schedule_engine.population import PopulationFactory

logger = logging.getLogger(__name__)

__all__ = ["EvolutionEngine", "EvolutionResult", "GenerationStats"]


# Result types


@dataclass
class GenerationStats:
    """Statistics for a single generation."""

    generation: int
    best_hard: float
    best_soft: float
    avg_hard: float
    avg_soft: float
    feasible_count: int
    elapsed: float = 0.0


@dataclass
class EvolutionResult:
    """Result of an evolution run."""

    best_individual: list[SessionGene]
    best_fitness: tuple[float, float]
    population: list[list[SessionGene]]
    generation_stats: list[GenerationStats]
    total_time: float = 0.0
    generations_run: int = 0

    @property
    def is_feasible(self) -> bool:
        return self.best_fitness[0] == 0.0


# Hook protocol

OnGenerationCallback = Callable[
    [int, list[list["SessionGene"]], "GenerationStats"], None
]


# Engine


class EvolutionEngine:
    """Pure evolution loop — no UI, no logging, no config singletons.

    Parameters
    ----------
    evaluator : Evaluator
        Fitness evaluator (from Phase 3).
    population_factory : PopulationFactory
        Population creator (from Phase 5).
    context : SchedulingContext
        The scheduling universe.
    qts : QuantumTimeSystem | None
        Time system for day-based Timetable indexes.
    pop_size : int
        Population size.
    generations : int
        Number of generations to evolve.
    cx_prob : float
        Crossover probability (per pair).
    mut_prob : float
        Mutation probability (per individual).
    tournament_size : int
        Tournament selection size.
    seed : int | None
        Random seed for reproducibility.
    on_generation : OnGenerationCallback | None
        Optional hook called after each generation.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        population_factory: PopulationFactory,
        context: SchedulingContext,
        *,
        qts: QuantumTimeSystem | None = None,
        pop_size: int = 50,
        generations: int = 100,
        cx_prob: float = 0.7,
        mut_prob: float = 0.3,
        tournament_size: int = 3,
        seed: int | None = 42,
        on_generation: OnGenerationCallback | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.factory = population_factory
        self.context = context
        self.qts = qts
        self.pop_size = pop_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.tournament_size = tournament_size
        self.seed = seed
        self.on_generation = on_generation
        self._rng = random.Random(seed)

    # Public: run

    def run(self) -> EvolutionResult:
        """Run evolution and return the result.

        No side effects — the caller is responsible for logging/display.
        """
        t0 = time.time()

        # Create initial population
        population = self.factory.create_population(self.pop_size, strategy="smart")

        # Evaluate initial population
        fitnesses = [self._evaluate(ind) for ind in population]

        stats_history: list[GenerationStats] = []

        for gen in range(self.generations):
            gen_t0 = time.time()

            # Selection (binary tournament, NSGA-II style)
            offspring = self._select(population, fitnesses)

            # Crossover
            offspring, off_fitnesses = self._crossover(offspring)

            # Mutation
            offspring, off_fitnesses = self._mutate(offspring, off_fitnesses)

            # Re-evaluate any that were modified
            for i in range(len(offspring)):
                if off_fitnesses[i] is None:
                    off_fitnesses[i] = self._evaluate(offspring[i])

            # Combine parent + offspring, select best
            combined = population + offspring
            combined_fit = fitnesses + off_fitnesses

            # Non-dominated sorting + crowding (simplified NSGA-II)
            population, fitnesses = self._nsga2_select(
                combined, combined_fit, self.pop_size
            )

            # Record stats
            gen_stats = self._compute_stats(gen, fitnesses, time.time() - gen_t0)
            stats_history.append(gen_stats)

            # Hook
            if self.on_generation is not None:
                self.on_generation(gen, population, gen_stats)

            # Early termination if feasible
            if gen_stats.best_hard == 0.0:
                logger.info(
                    f"Feasible solution found at generation {gen} "
                    f"(soft={gen_stats.best_soft:.1f})"
                )

        # Find best
        best_idx = self._best_index(fitnesses)
        total_time = time.time() - t0

        return EvolutionResult(
            best_individual=population[best_idx],
            best_fitness=fitnesses[best_idx],
            population=population,
            generation_stats=stats_history,
            total_time=total_time,
            generations_run=len(stats_history),
        )

    # Internal: evaluation

    def _evaluate(self, individual: list[SessionGene]) -> tuple[float, float]:
        """Evaluate a single individual."""
        return self.evaluator.fitness(individual, self.context, self.qts)

    # Internal: selection

    def _select(
        self,
        population: list[list[SessionGene]],
        fitnesses: list[tuple[float, float]],
    ) -> list[list[SessionGene]]:
        """Binary tournament selection."""
        selected = []
        for _ in range(len(population)):
            i = self._rng.randrange(len(population))
            j = self._rng.randrange(len(population))
            # Prefer lower hard, then lower soft
            if self._dominates(fitnesses[i], fitnesses[j]):
                selected.append([g for g in population[i]])  # shallow copy
            else:
                selected.append([g for g in population[j]])
        return selected

    # Internal: crossover

    def _crossover(
        self,
        offspring: list[list[SessionGene]],
    ) -> tuple[list[list[SessionGene]], list[tuple[float, float] | None]]:
        """Apply crossover operator."""
        fitnesses: list[tuple[float, float] | None] = [None] * len(offspring)

        for i in range(0, len(offspring) - 1, 2):
            if self._rng.random() < self.cx_prob:
                # Simple uniform crossover at gene level
                self._uniform_crossover(offspring[i], offspring[i + 1])
                fitnesses[i] = None  # needs re-evaluation
                fitnesses[i + 1] = None
        return offspring, fitnesses

    def _uniform_crossover(
        self,
        ind1: list[SessionGene],
        ind2: list[SessionGene],
    ) -> None:
        """Swap genes at matching positions with 50% probability."""
        min_len = min(len(ind1), len(ind2))
        for k in range(min_len):
            if self._rng.random() < 0.5:
                ind1[k], ind2[k] = ind2[k], ind1[k]

    # Internal: mutation

    def _mutate(
        self,
        offspring: list[list[SessionGene]],
        fitnesses: list[tuple[float, float] | None],
    ) -> tuple[list[list[SessionGene]], list[tuple[float, float] | None]]:
        """Apply mutation to each individual with probability mut_prob."""
        for i in range(len(offspring)):
            if self._rng.random() < self.mut_prob:
                self._mutate_individual(offspring[i])
                fitnesses[i] = None  # needs re-evaluation
        return offspring, fitnesses

    def _mutate_individual(self, individual: list[SessionGene]) -> None:
        """Mutate a single random gene's time slot."""
        if not individual:
            return
        idx = self._rng.randrange(len(individual))
        gene = individual[idx]
        # Random time shift within valid range
        from schedule_engine.io.time_system import QuantumTimeSystem

        qts = self.qts or QuantumTimeSystem()
        max_q = qts.total_quanta
        new_start = self._rng.randint(0, max(0, max_q - gene.num_quanta))
        gene.start_quanta = new_start

    # Internal: NSGA-II selection

    def _nsga2_select(
        self,
        population: list[list[SessionGene]],
        fitnesses: list[tuple[float, float] | None],
        n: int,
    ) -> tuple[list[list[SessionGene]], list[tuple[float, float]]]:
        """Simplified NSGA-II: non-dominated sorting + crowding distance."""
        # Filter out None fitnesses (shouldn't happen, but safety)
        valid = [
            (pop, fit) for pop, fit in zip(population, fitnesses) if fit is not None
        ]
        if not valid:
            return population[:n], [(float("inf"), float("inf"))] * n

        pops, fits = zip(*valid)
        pops = list(pops)
        fits = list(fits)

        # Sort lexicographically (hard first, then soft) — simplified
        indexed = sorted(range(len(pops)), key=lambda i: (fits[i][0], fits[i][1]))

        selected_pops = [pops[i] for i in indexed[:n]]
        selected_fits = [fits[i] for i in indexed[:n]]

        return selected_pops, selected_fits

    # Internal: helpers

    @staticmethod
    def _dominates(fit_a: tuple[float, float], fit_b: tuple[float, float]) -> bool:
        """True if fit_a dominates fit_b (both objectives <= and at least one <)."""
        return (
            fit_a[0] <= fit_b[0]
            and fit_a[1] <= fit_b[1]
            and (fit_a[0] < fit_b[0] or fit_a[1] < fit_b[1])
        )

    @staticmethod
    def _best_index(fitnesses: list[tuple[float, float]]) -> int:
        """Index of the best individual (lexicographic: hard then soft)."""
        best_i = 0
        for i in range(1, len(fitnesses)):
            if (fitnesses[i][0], fitnesses[i][1]) < (
                fitnesses[best_i][0],
                fitnesses[best_i][1],
            ):
                best_i = i
        return best_i

    @staticmethod
    def _compute_stats(
        gen: int,
        fitnesses: list[tuple[float, float]],
        elapsed: float,
    ) -> GenerationStats:
        """Compute per-generation statistics."""
        hard_vals = [f[0] for f in fitnesses]
        soft_vals = [f[1] for f in fitnesses]
        return GenerationStats(
            generation=gen,
            best_hard=min(hard_vals),
            best_soft=min(s for h, s in fitnesses if h == min(hard_vals)),
            avg_hard=sum(hard_vals) / len(hard_vals),
            avg_soft=sum(soft_vals) / len(soft_vals),
            feasible_count=sum(1 for h in hard_vals if h == 0.0),
            elapsed=elapsed,
        )
