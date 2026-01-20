"""
Reward calculator for RL environment.

Computes rewards based on:
1. Fitness improvement (primary signal)
2. Diversity bonus (encourage exploration)
3. Time penalty (discourage slow convergence)

ENHANCEMENT #1: Multi-objective reward shaping using hypervolume indicator.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.domain.types import Individual


@dataclass
class RewardComponents:
    """Breakdown of reward components."""

    fitness_reward: float
    diversity_bonus: float
    time_penalty: float
    hypervolume_reward: float  # ENHANCEMENT #1
    total_reward: float


class RewardCalculator:
    """
    Calculate rewards for RL agent actions.

    Reward Formula (scalar mode):
    reward = w1 * fitness_improvement + w2 * diversity_bonus - w3 * time_penalty

    Reward Formula (hypervolume mode - ENHANCEMENT #1):
    reward = w1 * hypervolume_delta + w2 * diversity_bonus - w3 * time_penalty

    Where:
    - fitness_improvement: Decrease in violations (negative fitness change)
    - hypervolume_delta: Change in hypervolume indicator (multi-objective)
    - diversity_bonus: Reward for maintaining population diversity
    - time_penalty: Small penalty per generation to encourage fast convergence
    """

    def __init__(
        self,
        fitness_weight: float = 1.0,
        diversity_weight: float = 0.1,
        time_weight: float = 0.01,
        normalize: bool = True,
        use_hypervolume: bool = False,
        reference_point: NDArray[np.float64] | None = None,
        hypervolume_scale: float = 1000.0,
    ):
        """
        Initialize reward calculator.

        Args:
            fitness_weight: Weight for fitness improvement reward
            diversity_weight: Weight for diversity bonus
            time_weight: Weight for time penalty
            normalize: Whether to normalize rewards to [-1, 1]
            use_hypervolume: Use hypervolume-based reward (ENHANCEMENT #1)
            reference_point: Reference point for hypervolume (e.g., [1000, 10000])
            hypervolume_scale: Scale factor for hypervolume normalization
        """
        self.fitness_weight = fitness_weight  # type: ignore[attr-defined]
        self.diversity_weight = diversity_weight
        self.time_weight = time_weight
        self.normalize = normalize

        # ENHANCEMENT #1: Hypervolume-based rewards
        self.use_hypervolume = use_hypervolume
        self.hypervolume_scale = hypervolume_scale

        if use_hypervolume:
            from src.rl.gym_env.hypervolume import HypervolumeCalculator

            if reference_point is None:
                # Default reference point for schedule optimization
                reference_point = np.array([1000.0, 10000.0])

            self.reference_point = np.asarray(reference_point, dtype=np.float64)
            self.hv_calculator: HypervolumeCalculator | None = HypervolumeCalculator(
                reference_point=self.reference_point, minimize=True
            )
        else:
            self.hv_calculator = None

        # Track previous state for delta calculation
        self.prev_best_fitness: float | None = None
        self.prev_avg_fitness: float | None = None
        self.prev_diversity: float | None = None
        self.prev_pareto_front: NDArray[np.float64] | None = None

    def calculate_reward(
        self,
        prev_individual: Individual,
        new_individual: Individual,
        population_diversity: float,
        generation: int,
        population: list[Individual] | None = None,
    ) -> tuple[float, RewardComponents]:
        """
        Calculate reward for a single action.

        Args:
            prev_individual: Individual before action
            new_individual: Individual after action
            population_diversity: Current population diversity metric
            generation: Current generation number
            population: Full population (needed for hypervolume calculation)

        Returns:
            (total_reward, reward_components)
        """
        # 1. Fitness improvement reward (scalar or hypervolume)
        if self.use_hypervolume and population is not None:
            fitness_reward = self._calculate_hypervolume_reward(population)
            hypervolume_reward = fitness_reward
        else:
            fitness_reward = self._calculate_fitness_reward(
                prev_individual, new_individual
            )
            hypervolume_reward = 0.0

        # 2. Diversity bonus
        diversity_bonus = self._calculate_diversity_bonus(population_diversity)

        # 3. Time penalty
        time_penalty = self._calculate_time_penalty(generation)

        # Weighted sum
        total_reward = (
            self.fitness_weight * fitness_reward  # type: ignore[attr-defined]
            + self.diversity_weight * diversity_bonus
            - self.time_weight * time_penalty
        )

        if self.normalize:
            total_reward = np.clip(total_reward, -1.0, 1.0)

        components = RewardComponents(
            fitness_reward=fitness_reward if not self.use_hypervolume else 0.0,
            diversity_bonus=diversity_bonus,
            time_penalty=time_penalty,
            hypervolume_reward=hypervolume_reward,
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

    def _calculate_hypervolume_reward(self, population: list[Individual]) -> float:
        """
        Calculate hypervolume-based reward (ENHANCEMENT #1).

        Reward = improvement in hypervolume indicator.
        Positive reward: Pareto front improved
        Negative reward: Pareto front degraded

        Args:
            population: Current GA population

        Returns:
            Hypervolume-based reward
        """
        if self.hv_calculator is None:
            return 0.0

        # Extract Pareto front (fitness values for all individuals)
        pareto_front = np.array(
            [
                ind.fitness.values  # type: ignore[attr-defined]
                for ind in population
                if hasattr(ind, "fitness") and ind.fitness.valid  # type: ignore[attr-defined]
            ]
        )

        if len(pareto_front) == 0:
            return 0.0

        # Calculate hypervolume
        current_hv = self.hv_calculator.compute(pareto_front)

        # Calculate reward as delta from previous state
        if self.prev_pareto_front is None:
            self.prev_pareto_front = pareto_front
            prev_hv = current_hv
        else:
            prev_hv = self.hv_calculator.compute(self.prev_pareto_front)
            self.prev_pareto_front = pareto_front

        delta_hv = current_hv - prev_hv

        # Normalize using tanh to [-1, 1]
        normalized_reward = np.tanh(delta_hv / self.hypervolume_scale)

        return float(normalized_reward)

    def _get_combined_fitness(self, individual: Individual) -> float:
        """
        Get combined fitness value.

        Combines hard and soft violations: hard * 100 + soft
        """
        if not hasattr(individual, "fitness") or not individual.fitness.valid:  # type: ignore[attr-defined]
            return float("inf")

        hard, soft = individual.fitness.values  # type: ignore[attr-defined]
        return float(abs(hard) * 100 + abs(soft))

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
        self.prev_pareto_front = None  # ENHANCEMENT #1

    def get_config(self) -> dict[str, Any]:
        """Get reward calculator configuration."""
        config = {
            "fitness_weight": self.fitness_weight,  # type: ignore[attr-defined]
            "diversity_weight": self.diversity_weight,
            "time_weight": self.time_weight,
            "normalize": self.normalize,
        }

        # ENHANCEMENT #1: Add hypervolume config
        if self.use_hypervolume:
            config.update(
                {
                    "use_hypervolume": True,
                    "reference_point": self.reference_point.tolist(),
                    "hypervolume_scale": self.hypervolume_scale,
                }
            )

        return config


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
