"""
Scalar reward calculator (weighted sum of objectives).

Traditional approach: reward = w1 * hard_improvement + w2 * soft_improvement
"""

from typing import List
from src.core.types import Individual
from src.rl.rewards.base_reward import BaseRewardCalculator


class ScalarReward(BaseRewardCalculator):
    """
    Scalar reward using weighted sum of hard/soft violation improvements.

    Reward = w_hard * Δhard + w_soft * Δsoft - action_cost

    Where:
    - Δhard = prev_hard - curr_hard (positive if improved)
    - Δsoft = prev_soft - curr_soft (positive if improved)
    - action_cost = computational cost penalty (optional)
    """

    def __init__(self, config: dict = None):
        """
        Initialize scalar reward calculator.

        Args:
            config: Configuration with:
                - hard_weight: Weight for hard violations (default: 100.0)
                - soft_weight: Weight for soft violations (default: 1.0)
                - cost_weight: Weight for action cost (default: 0.01)
                - normalize: Whether to normalize rewards (default: True)
        """
        super().__init__(config)
        self.hard_weight = self.config.get("hard_weight", 100.0)
        self.soft_weight = self.config.get("soft_weight", 1.0)
        self.cost_weight = self.config.get("cost_weight", 0.01)
        self.normalize = self.config.get("normalize", True)

    def calculate(
        self,
        prev_population: List[Individual],
        current_population: List[Individual],
        action_cost: float = 0.0,
    ) -> float:
        """
        Calculate scalar reward from fitness improvement.

        Args:
            prev_population: Population before action
            current_population: Population after action
            action_cost: Cost of the action taken

        Returns:
            Scalar reward (positive if improved, negative if worsened)
        """
        # Get best fitness from each population
        prev_best = self.get_best_fitness(prev_population)
        curr_best = self.get_best_fitness(current_population)

        # Calculate improvements (positive = better)
        hard_improvement = prev_best[0] - curr_best[0]
        soft_improvement = prev_best[1] - curr_best[1]

        # Weighted sum
        reward = (
            self.hard_weight * hard_improvement
            + self.soft_weight * soft_improvement
            - self.cost_weight * action_cost
        )

        # Optional normalization to [-1, 1] range
        if self.normalize:
            # Clip to reasonable range to prevent explosions
            reward = max(min(reward / 100.0, 1.0), -1.0)

        return reward

    def calculate_with_avg(
        self,
        prev_population: List[Individual],
        current_population: List[Individual],
        action_cost: float = 0.0,
    ) -> float:
        """
        Calculate scalar reward using average fitness (instead of best).

        More stable for small populations, less sensitive to outliers.

        Args:
            prev_population: Population before action
            current_population: Population after action
            action_cost: Cost of the action taken

        Returns:
            Scalar reward based on average fitness improvement
        """
        # Get average fitness from each population
        prev_avg = self.get_avg_fitness(prev_population)
        curr_avg = self.get_avg_fitness(current_population)

        # Calculate improvements
        hard_improvement = prev_avg[0] - curr_avg[0]
        soft_improvement = prev_avg[1] - curr_avg[1]

        # Weighted sum
        reward = (
            self.hard_weight * hard_improvement
            + self.soft_weight * soft_improvement
            - self.cost_weight * action_cost
        )

        # Optional normalization
        if self.normalize:
            reward = max(min(reward / 100.0, 1.0), -1.0)

        return reward
