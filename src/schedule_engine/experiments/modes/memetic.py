"""
MemeticExperiment - NSGA-II + Local Search (Mode B).

Applies local search (repair operators) to improve individuals after genetic operators.

Usage:
    from schedule_engine.experiments import MemeticExperiment

    exp = MemeticExperiment(
        seed=42,
        pop_size=50,
        ngen=200,
        local_search_prob=0.5,
        repair_policy="round_robin",
    )
    exp.run()
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any

from schedule_engine.experiments.base import BaseExperiment
from schedule_engine.experiments.output.repair_exporter import RepairExporter
from schedule_engine.ga.run_helpers import EvolutionStats


class MemeticExperiment(BaseExperiment):
    """
    Memetic NSGA-II experiment (NSGA-II + Local Search).

    Enhances genetic algorithm with local search:
    - After crossover/mutation, selected offspring undergo repair
    - RepairEngine applies targeted heuristics to fix violations
    - Repairs are applied probabilistically to avoid overhead

    Additional Parameters:
    ---------------------
    local_search_prob : float
        Probability of applying local search to an offspring (0.0 to 1.0)
    local_search_iterations : int
        Maximum repair steps per individual
    repair_policy : str
        Heuristic selection policy: "round_robin", "ucb", "random"
    repair_budget_ms : float
        Time budget for repairs per generation (milliseconds)
    repair_max_candidates : int
        Maximum candidate moves to evaluate per step
    repair_epsilon : float
        Exploration rate for adaptive policies
    """

    def __init__(
        self,
        *,
        # Memetic-specific parameters
        local_search_prob: float = 0.5,
        local_search_iterations: int = 15,
        repair_policy: str = "round_robin",
        repair_budget_ms: float = 120.0,
        repair_max_candidates: int = 30,
        repair_epsilon: float = 0.1,
        # Pass through to base
        **kwargs: Any,
    ) -> None:
        """Initialize memetic experiment."""
        super().__init__(**kwargs)

        # Memetic parameters
        self.local_search_prob = local_search_prob
        self.local_search_iterations = local_search_iterations
        self.repair_policy = repair_policy
        self.repair_budget_ms = repair_budget_ms
        self.repair_max_candidates = repair_max_candidates
        self.repair_epsilon = repair_epsilon

        # Track repair stats
        self._repair_history: list[dict[str, float | int]] = []
        self._total_repairs: int = 0

    def _get_experiment_name(self) -> str:
        """Return experiment name."""
        return "ga_02_memetic"

    def _get_extra_config(self) -> dict[str, Any]:
        """Return memetic-specific configuration."""
        return {
            "local_search_prob": self.local_search_prob,
            "local_search_iterations": self.local_search_iterations,
            "repair_policy": self.repair_policy,
            "repair_budget_ms": self.repair_budget_ms,
            "repair_max_candidates": self.repair_max_candidates,
            "repair_epsilon": self.repair_epsilon,
        }

    def _get_extra_results(self) -> dict[str, Any]:
        """Return memetic-specific results."""
        return {
            "total_repairs": self._total_repairs,
        }

    def _run_evolution(self) -> tuple[list[Any], EvolutionStats]:
        """
        Run memetic NSGA-II evolution.

        Standard NSGA-II with local search applied to selected offspring.
        """
        from schedule_engine.ga.repair.engine import RepairEngine

        start_time = time.time()

        # Create repair engine
        repair_engine = RepairEngine(
            context=self.data.context,
            evaluator=self.evaluate,
            policy=self.repair_policy,
            max_steps=self.local_search_iterations,
            max_candidates=self.repair_max_candidates,
            budget_ms=self.repair_budget_ms,
            epsilon=self.repair_epsilon,
            rng=random.Random(self.seed),
            logger=self.logger,
            log_steps=True,
            log_candidates=True,
        )

        # Create initial population
        pop = self.create_initial_population()

        stats = EvolutionStats()
        self._repair_history = []
        self._total_repairs = 0

        for gen in range(self.ngen):
            gen_start = time.time()

            # Selection
            offspring = [
                copy.deepcopy(ind) for ind in self.toolbox.select(pop, len(pop))
            ]

            # Crossover
            self.apply_crossover(offspring)

            # Mutation
            self.apply_mutation(offspring)

            # Local Search (memetic component)
            repair_start = time.time()
            repair_indices = [
                idx
                for idx in range(len(offspring))
                if random.random() < self.local_search_prob
            ]
            per_individual_budget = (
                self.repair_budget_ms / max(1, len(repair_indices))
                if self.repair_budget_ms > 0
                else 0
            )

            gen_repairs = 0
            gen_delta_hard = 0.0
            gen_delta_soft = 0.0

            for idx in repair_indices:
                ind = offspring[idx]
                repair_stats = repair_engine.repair_individual(
                    ind, budget_ms=per_individual_budget
                )
                gen_repairs += repair_stats.applied_steps
                gen_delta_hard += repair_stats.total_delta_hard
                gen_delta_soft += repair_stats.total_delta_soft
                if repair_stats.applied_steps > 0:
                    del ind.fitness.values

            self._total_repairs += gen_repairs
            repair_time_ms = (time.time() - repair_start) * 1000

            self._repair_history.append(
                {
                    "generation": gen,
                    "repairs_applied": gen_repairs,
                    "delta_hard": gen_delta_hard,
                    "delta_soft": gen_delta_soft,
                    "repair_time_ms": repair_time_ms,
                }
            )

            # Evaluate
            self.evaluate_offspring(offspring)

            # Survivor selection
            pop = self.toolbox.select(pop + offspring, self.pop_size)

            # Record stats
            self.record_generation_stats(pop, stats, gen, gen_start)

            # Log progress with repair info
            if gen % self.log_interval == 0 or gen == self.ngen - 1:
                self.log_generation_progress(pop, gen)
                self.logger.debug(
                    f"Gen {gen}: repairs={gen_repairs}, "
                    f"delta_hard={gen_delta_hard:.2f}, "
                    f"delta_soft={gen_delta_soft:.2f}"
                )

        stats.elapsed_time = time.time() - start_time

        return pop, stats

    def _create_exporter(self) -> RepairExporter:
        """Create repair-aware exporter."""
        return RepairExporter(
            output_dir=self.output_dir,
            data=self.data,
            logger=self.logger,
        )

    def _export_results(self) -> None:
        """Export results with repair history."""
        exporter = self._create_exporter()
        exporter.export_all(
            final_pop=self._final_pop,
            stats=self._stats,
            best_individual=self._best_individual,
            metadata=self._build_metadata(),
            repair_history=self._repair_history,
        )
