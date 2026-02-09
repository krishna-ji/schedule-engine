"""
BaselineExperiment - Pure NSGA-II (Mode A).

No enhancements, no repair heuristics, no RL guidance.
Foundation for comparing all other modes.

Usage:
    from schedule_engine.experiments import BaselineExperiment

    exp = BaselineExperiment(seed=42, pop_size=50, ngen=100)
    exp.run()
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any

from schedule_engine.experiments.base import BaseExperiment
from schedule_engine.ga.run_helpers import EvolutionStats


class BaselineExperiment(BaseExperiment):
    """
    Pure NSGA-II baseline experiment.

    Standard genetic algorithm with:
    - Tournament selection via NSGA-II
    - Course-aware crossover
    - Smart mutation
    - No local search or repair

    This is the simplest mode and serves as the baseline for comparison.
    """

    def _get_experiment_name(self) -> str:
        """Return experiment name."""
        return "mode_a_baseline"

    def _run_evolution(self) -> tuple[list[Any], EvolutionStats]:
        """
        Run pure NSGA-II evolution.

        Standard evolution loop without any enhancements.
        """
        start_time = time.time()

        # Create initial population
        pop = self.create_initial_population()

        stats = EvolutionStats()

        for gen in range(self.ngen):
            gen_start = time.time()

            # Selection: copy selected individuals
            offspring = [
                copy.deepcopy(ind) for ind in self.toolbox.select(pop, len(pop))
            ]

            # Crossover
            self.apply_crossover(offspring)

            # Mutation
            self.apply_mutation(offspring)

            # Evaluate
            self.evaluate_offspring(offspring)

            # Survivor selection (NSGA-II environmental selection)
            pop = self.toolbox.select(pop + offspring, self.pop_size)

            # Record stats
            self.record_generation_stats(pop, stats, gen, gen_start)

            # Log progress
            self.log_generation_progress(pop, gen)

        stats.elapsed_time = time.time() - start_time

        return pop, stats
