"""
RLGuidedExperiment - NSGA-II + RL-Guided Heuristic Selection (Mode E).

Uses Q-learning to learn optimal heuristic selection.

Usage:
    from schedule_engine.experiments import RLGuidedExperiment

    exp = RLGuidedExperiment(
        seed=42,
        pop_size=50,
        ngen=100,
        learning_rate=0.2,
        epsilon_start=1.0,
        epsilon_end=0.1,
    )
    exp.run()
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any

import numpy as np

from schedule_engine.experiments.base import BaseExperiment
from schedule_engine.experiments.output.rl_exporter import RLExporter
from schedule_engine.ga.run_helpers import EvolutionStats


class RLGuidedExperiment(BaseExperiment):
    """
    RL-Guided heuristic selection experiment.

    Uses Q-learning to learn which heuristics are effective:
    - State: constraint violation pattern (discretized)
    - Actions: available repair heuristics
    - Reward: fitness improvement after heuristic application
    - Policy: epsilon-greedy with decay

    Additional Parameters:
    ---------------------
    repair_prob : float
        Probability of applying repair to an offspring
    learning_rate : float
        Q-learning alpha (how fast to update Q-values)
    epsilon_start : float
        Initial exploration rate (1.0 = fully random)
    epsilon_end : float
        Final exploration rate (0.0 = fully greedy)
    epsilon_decay : float
        Multiplicative decay per generation
    """

    def __init__(
        self,
        *,
        # RL-specific parameters
        repair_prob: float = 0.3,
        learning_rate: float = 0.2,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        **kwargs: Any,
    ) -> None:
        """Initialize RL-guided experiment."""
        super().__init__(**kwargs)

        # RL parameters
        self.repair_prob = repair_prob
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Track RL state
        self._epsilon_history: list[float] = []
        self._q_table_history: list[dict[str, float]] = []
        self._rewards_history: list[float] = []
        self._total_repairs: int = 0

    def _get_experiment_name(self) -> str:
        """Return experiment name."""
        return "mode_e_rl_guided"

    def _get_extra_config(self) -> dict[str, Any]:
        """Return RL-specific configuration."""
        return {
            "repair_prob": self.repair_prob,
            "learning_rate": self.learning_rate,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }

    def _get_extra_results(self) -> dict[str, Any]:
        """Return RL-specific results."""
        return {
            "total_repairs": self._total_repairs,
            "final_epsilon": (
                self._epsilon_history[-1] if self._epsilon_history else None
            ),
            "final_q_table": self._q_table_history[-1] if self._q_table_history else {},
        }

    def _run_evolution(self) -> tuple[list[Any], EvolutionStats]:
        """
        Run RL-guided NSGA-II evolution.

        Uses SimpleRLSelector for heuristic selection with Q-learning.
        """
        from schedule_engine.heuristics.strategies import SimpleRLSelector

        start_time = time.time()

        # Create RL selector
        selector = SimpleRLSelector(
            learning_rate=self.learning_rate,
            epsilon=self.epsilon_start,
            epsilon_decay=self.epsilon_decay,
            min_epsilon=self.epsilon_end,
        )

        # Create initial population
        pop = self.create_initial_population()

        stats = EvolutionStats()
        self._epsilon_history = []
        self._q_table_history = []
        self._rewards_history = []
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

            # RL-Guided Repair
            gen_rewards: list[float] = []
            for ind in offspring:
                if random.random() < self.repair_prob:
                    genes = list(ind)
                    _, fixes, reward = selector.apply(genes, self.data, self.evaluate)
                    self._total_repairs += fixes
                    ind[:] = genes
                    del ind.fitness.values
                    gen_rewards.append(reward)

            # Record rewards
            if gen_rewards:
                self._rewards_history.append(float(np.mean(gen_rewards)))
            else:
                self._rewards_history.append(0.0)

            # Decay epsilon
            selector.decay_epsilon()

            # Evaluate
            self.evaluate_offspring(offspring)

            # Survivor selection
            pop = self.toolbox.select(pop + offspring, self.pop_size)

            # Record stats
            self.record_generation_stats(pop, stats, gen, gen_start)

            # Track RL state
            self._epsilon_history.append(selector.epsilon)

            # Compute average Q-values across states
            if selector.q_table:
                action_qs = {action: 0.0 for action in selector.actions}
                for state_qs in selector.q_table.values():
                    for action, q_value in state_qs.items():
                        action_qs[action] += q_value
                action_qs = {
                    action: q_value / len(selector.q_table)
                    for action, q_value in action_qs.items()
                }
            else:
                action_qs = {action: 0.0 for action in selector.actions}
            self._q_table_history.append(action_qs)

            # Log progress with RL info
            if gen % self.log_interval == 0 or gen == self.ngen - 1:
                self.log_generation_progress(pop, gen)
                q_str = ", ".join(f"{k}:{v:.2f}" for k, v in action_qs.items())
                self.logger.debug(
                    f"Gen {gen}: epsilon={selector.epsilon:.3f}, Q=[{q_str}]"
                )

        stats.elapsed_time = time.time() - start_time

        # Log final state
        self.logger.info(f"Final Q-table: {selector.q_table}")
        self.logger.info(f"Final epsilon: {selector.epsilon:.4f}")

        return pop, stats

    def _create_exporter(self) -> RLExporter:
        """Create RL-aware exporter."""
        return RLExporter(
            output_dir=self.output_dir,
            data=self.data,
            logger=self.logger,
        )

    def _export_results(self) -> None:
        """Export results with RL history."""
        exporter = self._create_exporter()
        exporter.export_all(
            final_pop=self._final_pop,
            stats=self._stats,
            best_individual=self._best_individual,
            metadata=self._build_metadata(),
            q_table_history=self._q_table_history,
            epsilon_history=self._epsilon_history,
            rewards_history=self._rewards_history,
        )
