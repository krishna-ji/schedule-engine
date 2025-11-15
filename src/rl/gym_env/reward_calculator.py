"""
Reward calculator for RL environment.

Computes rewards based on:
1. Fitness improvement (primary signal)
2. Diversity bonus (encourage exploration)
3. Time penalty (discourage slow convergence)
"""

from typing import Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

from src.core.types import Individual


@dataclass
class RewardComponents:
    """Breakdown of reward components."""

    fitness_reward: float
    diversity_bonus: float
    time_penalty: float
    total_reward: float


class RewardCalculator:
    """
    Calculate rewards for RL agent actions.

    Reward Formula:
    reward = w1 * fitness_improvement + w2 * diversity_bonus - w3 * time_penalty

    Where:
    - fitness_improvement: Decrease in violations (negative fitness change)
    - diversity_bonus: Reward for maintaining population diversity
    - time_penalty: Small penalty per generation to encourage fast convergence
    """

    def __init__(
        self,
        fitness_weight: float = 1.0,
        diversity_weight: float = 0.1,
        time_weight: float = 0.01,
        normalize: bool = True,
    ):
        """
        Initialize reward calculator.

        Args:
            fitness_weight: Weight for fitness improvement reward
            diversity_weight: Weight for diversity bonus
            time_weight: Weight for time penalty
            normalize: Whether to normalize rewards to [-1, 1]
        """
        self.fitness_weight = fitness_weight
        self.diversity_weight = diversity_weight
        self.time_weight = time_weight
        self.normalize = normalize

        # Track previous state for delta calculation
        self.prev_best_fitness: float | None = None
        self.prev_avg_fitness: float | None = None
        self.prev_diversity: float | None = None

    def calculate_reward(
        self,
        prev_individual: Individual,
        new_individual: Individual,
        population_diversity: float,
        generation: int,
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate reward for a single action.

        Args:
            prev_individual: Individual before action
            new_individual: Individual after action
            population_diversity: Current population diversity metric
            generation: Current generation number

        Returns:
            (total_reward, reward_components)
        """
        # 1. Fitness improvement reward
        fitness_reward = self._calculate_fitness_reward(prev_individual, new_individual)

        # 2. Diversity bonus
        diversity_bonus = self._calculate_diversity_bonus(population_diversity)

        # 3. Time penalty
        time_penalty = self._calculate_time_penalty(generation)

        # Weighted sum
        total_reward = (
            self.fitness_weight * fitness_reward
            + self.diversity_weight * diversity_bonus
            - self.time_weight * time_penalty
        )

        if self.normalize:
            total_reward = np.clip(total_reward, -1.0, 1.0)

        components = RewardComponents(
            fitness_reward=fitness_reward,
            diversity_bonus=diversity_bonus,
            time_penalty=time_penalty,
            total_reward=total_reward,
        )

        return total_reward, components

    def _calculate_fitness_reward(
        self, prev_individual: Individual, new_individual: Individual
    ) -> float:
        """
        Calculate fitness improvement reward.

        Positive reward for fitness decrease (fewer violations).
        Reward = -(new_fitness - prev_fitness)
        """
        prev_fitness = self._get_combined_fitness(prev_individual)
        new_fitness = self._get_combined_fitness(new_individual)

        # Improvement = decrease in fitness (violations)
        improvement = prev_fitness - new_fitness

        # Normalize by previous fitness to make scale-invariant
        if prev_fitness != 0:
            normalized_improvement = improvement / abs(prev_fitness)
        else:
            normalized_improvement = improvement

        return normalized_improvement

    def _calculate_diversity_bonus(self, population_diversity: float) -> float:
        """
        Calculate diversity bonus.

        Reward for maintaining diversity (exploration).
        """
        if self.prev_diversity is None:
            self.prev_diversity = population_diversity
            return 0.0

        # Bonus for increasing diversity
        diversity_delta = population_diversity - self.prev_diversity
        self.prev_diversity = population_diversity

        # Small bonus for positive diversity change
        return diversity_delta * 0.1

    def _calculate_time_penalty(self, generation: int) -> float:
        """
        Calculate time penalty.

        Small penalty to encourage fast convergence.
        """
        return 0.001 * generation  # Small linear penalty

    def _get_combined_fitness(self, individual: Individual) -> float:
        """
        Get combined fitness value.

        Combines hard and soft violations: hard * 100 + soft
        """
        if not hasattr(individual, "fitness") or not individual.fitness.valid:
            return float("inf")

        hard, soft = individual.fitness.values
        return abs(hard) * 100 + abs(soft)

    def calculate_episode_reward(
        self,
        initial_best_fitness: float,
        final_best_fitness: float,
        generations_used: int,
        max_generations: int,
    ) -> float:
        """
        Calculate total episode reward.

        Used for episode-level evaluation.

        Args:
            initial_best_fitness: Best fitness at episode start
            final_best_fitness: Best fitness at episode end
            generations_used: Number of generations used
            max_generations: Maximum allowed generations

        Returns:
            Episode reward
        """
        # Fitness improvement
        improvement = initial_best_fitness - final_best_fitness
        improvement_ratio = improvement / (abs(initial_best_fitness) + 1e-6)

        # Efficiency bonus (finish faster)
        efficiency = 1.0 - (generations_used / max_generations)

        # Combined episode reward
        episode_reward = improvement_ratio + 0.1 * efficiency

        return episode_reward

    def reset(self) -> None:
        """Reset calculator state (call at episode start)."""
        self.prev_best_fitness = None
        self.prev_avg_fitness = None
        self.prev_diversity = None

    def get_config(self) -> Dict[str, Any]:
        """Get reward calculator configuration."""
        return {
            "fitness_weight": self.fitness_weight,
            "diversity_weight": self.diversity_weight,
            "time_weight": self.time_weight,
            "normalize": self.normalize,
        }


def create_reward_calculator(
    fitness_weight: float = 1.0,
    diversity_weight: float = 0.1,
    time_weight: float = 0.01,
) -> RewardCalculator:
    """
    Factory function to create reward calculator.

    Args:
        fitness_weight: Weight for fitness improvement
        diversity_weight: Weight for diversity bonus
        time_weight: Weight for time penalty

    Returns:
        Configured RewardCalculator instance
    """
    return RewardCalculator(
        fitness_weight=fitness_weight,
        diversity_weight=diversity_weight,
        time_weight=time_weight,
        normalize=True,
    )
