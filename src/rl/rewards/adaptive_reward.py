"""
Adaptive preference reward calculator.

ENHANCEMENT #1: Dynamically adjusts objective weights based on
search progress and constraint satisfaction state.

Prioritizes hard constraints early, then shifts to soft constraints
once feasibility is achieved.
"""

import numpy as np

from src.core.types import Individual
from src.rl.rewards.base_reward import BaseRewardCalculator


class AdaptiveReward(BaseRewardCalculator):
    """
    Adaptive reward with dynamic preference weighting.

    Reward = w_hard(t) * Δhard + w_soft(t) * Δsoft

    Where weights adapt based on:
    - Feasibility status (infeasible → feasible transition)
    - Search progress (generation number, stagnation)
    - Population quality (convergence, diversity)

    Strategies:
    1. Feasibility-adaptive: w_hard = 100 if infeasible, 10 if feasible
    2. Progress-adaptive: Gradually shift from hard to soft focus
    3. Stagnation-adaptive: Boost diversity when stuck
    """

    def __init__(self, config: dict = None):
        """
        Initialize adaptive reward calculator.

        Args:
            config: Configuration with:
                - strategy: 'feasibility' | 'progress' | 'stagnation'
                - initial_hard_weight: Starting weight for hard violations
                - initial_soft_weight: Starting weight for soft violations
                - adaptation_rate: How quickly to adapt weights
        """
        super().__init__(config)

        self.strategy = self.config.get("strategy", "feasibility")
        self.initial_hard_weight = self.config.get("initial_hard_weight", 100.0)
        self.initial_soft_weight = self.config.get("initial_soft_weight", 1.0)
        self.adaptation_rate = self.config.get("adaptation_rate", 0.1)

        # Current weights (will adapt over time)
        self.current_hard_weight = self.initial_hard_weight
        self.current_soft_weight = self.initial_soft_weight

        # Track search state
        self.is_feasible = False
        self.generation_count = 0
        self.stagnation_count = 0

    def calculate(
        self,
        prev_population: list[Individual],
        current_population: list[Individual],
        action_cost: float = 0.0,
    ) -> float:
        """
        Calculate adaptive reward with dynamic weights.

        Args:
            prev_population: Population before action
            current_population: Population after action
            action_cost: Cost of the action taken

        Returns:
            Weighted reward with adapted preferences
        """
        # Update search state
        self._update_state(current_population)

        # Adapt weights based on strategy
        self._adapt_weights()

        # Get fitness improvements
        prev_best = self.get_best_fitness(prev_population)
        curr_best = self.get_best_fitness(current_population)

        hard_improvement = prev_best[0] - curr_best[0]
        soft_improvement = prev_best[1] - curr_best[1]

        # Calculate reward with adapted weights
        reward = (
            self.current_hard_weight * hard_improvement
            + self.current_soft_weight * soft_improvement
        )

        # Normalize to [-1, 1]
        reward = np.tanh(reward / 100.0)

        return float(reward)

    def _update_state(self, population: list[Individual]) -> None:
        """
        Update internal state based on current population.

        Args:
            population: Current GA population
        """
        self.generation_count += 1

        # Check feasibility
        best = self.get_best_fitness(population)
        prev_feasible = self.is_feasible
        self.is_feasible = best[0] == 0  # No hard violations

        # Track stagnation (simplified)
        if not hasattr(self, "_prev_best"):
            self._prev_best = best
            self.stagnation_count = 0
        elif best == self._prev_best:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        self._prev_best = best

        # Detect feasibility transition
        if self.is_feasible and not prev_feasible:
            print(f"Feasibility achieved at generation {self.generation_count}!")

    def _adapt_weights(self) -> None:
        """Adapt objective weights based on current strategy."""

        if self.strategy == "feasibility":
            # Strategy 1: Focus on hard constraints until feasible
            if self.is_feasible:
                # Feasible: reduce hard weight, increase soft weight
                target_hard = 10.0
                target_soft = 10.0
            else:
                # Infeasible: maximize hard weight
                target_hard = 100.0
                target_soft = 1.0

            # Smooth transition
            self.current_hard_weight += self.adaptation_rate * (
                target_hard - self.current_hard_weight
            )
            self.current_soft_weight += self.adaptation_rate * (
                target_soft - self.current_soft_weight
            )

        elif self.strategy == "progress":
            # Strategy 2: Gradually shift from hard to soft focus
            progress = min(self.generation_count / 1000.0, 1.0)  # 0 → 1

            # Linear interpolation
            self.current_hard_weight = (
                self.initial_hard_weight * (1 - progress) + 10.0 * progress
            )
            self.current_soft_weight = (
                self.initial_soft_weight * (1 - progress) + 10.0 * progress
            )

        elif self.strategy == "stagnation":
            # Strategy 3: Boost exploration when stagnated
            if self.stagnation_count > 10:
                # Stagnated: reduce weights to encourage exploration
                self.current_hard_weight = max(self.current_hard_weight * 0.9, 10.0)
                self.current_soft_weight = max(self.current_soft_weight * 0.9, 1.0)
            else:
                # Not stagnated: restore original weights
                self.current_hard_weight = min(
                    self.current_hard_weight * 1.1, self.initial_hard_weight
                )
                self.current_soft_weight = min(
                    self.current_soft_weight * 1.1, self.initial_soft_weight
                )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def reset(self) -> None:
        """Reset adaptive state (call at episode start)."""
        self.current_hard_weight = self.initial_hard_weight
        self.current_soft_weight = self.initial_soft_weight
        self.is_feasible = False
        self.generation_count = 0
        self.stagnation_count = 0
        if hasattr(self, "_prev_best"):
            delattr(self, "_prev_best")

    def get_current_weights(self) -> tuple[float, float]:
        """
        Get current objective weights.

        Returns:
            (hard_weight, soft_weight)
        """
        return (self.current_hard_weight, self.current_soft_weight)
