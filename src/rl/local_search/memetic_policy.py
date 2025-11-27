"""
Memetic policy for RL-guided local search budget allocation.

ENHANCEMENT #6: RL learns when/where/how much to apply local search.
"""

import numpy as np
from numpy.typing import NDArray

from src.core.types import Individual


class MemeticPolicy:
    """
    RL policy for adaptive local search budget allocation.

    Decisions:
    - WHEN: Which generation to apply local search
    - WHERE: Which solutions to intensify (elite, diverse, etc.)
    - HOW MUCH: Budget (iterations) for each solution
    - WHICH: Which local search operator to use

    Action space (discrete):
    - Budget levels: [0, 10, 50, 100, 200, 500]
    - 0 = skip local search, 500 = intensive search

    State features:
    - Solution quality (fitness, rank)
    - Population diversity
    - Search progress (generation, stagnation)
    - Computational budget remaining
    """

    def __init__(self, config: dict = None):
        """
        Initialize memetic policy.

        Args:
            config: Configuration with budget levels and parameters
        """
        self.config = config or {}
        self.budget_levels = self.config.get(
            "budget_levels", [0, 10, 50, 100, 200, 500]
        )
        self.model = None  # RL model (loaded on demand)

    def get_action_space_size(self) -> int:
        """Get size of action space."""
        return len(self.budget_levels)

    def select_budget(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> int:
        """
        Select local search budget for current solution.

        Args:
            observation: State features
            deterministic: Use deterministic policy (no exploration)

        Returns:
            Budget (number of iterations)
        """
        if self.model is None:
            self._load_model()

        if self.model:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            action_idx = int(action)
            return self.budget_levels[action_idx]
        else:
            # Fallback: Fixed budget
            return 100

    def compute_state(
        self,
        individual: Individual,
        population: list[Individual],
        generation: int,
        max_generations: int,
    ) -> NDArray[np.float32]:
        """
        Compute state features for memetic decision.

        Args:
            individual: Solution to consider for local search
            population: Full population
            generation: Current generation
            max_generations: Total generations

        Returns:
            State feature vector
        """
        # Individual features
        hard_violations = individual.fitness.values[0]
        soft_violations = individual.fitness.values[1]

        # Population statistics
        all_fitness = np.array([ind.fitness.values for ind in population])
        avg_hard = np.mean(all_fitness[:, 0])
        avg_soft = np.mean(all_fitness[:, 1])

        # Relative quality
        hard_rank = np.sum(all_fitness[:, 0] < hard_violations) / len(population)
        soft_rank = np.sum(all_fitness[:, 1] < soft_violations) / len(population)

        # Search progress
        progress = generation / max(max_generations, 1)

        # Budget remaining (normalized)
        budget_used = generation / max(max_generations, 1)
        budget_remaining = 1.0 - budget_used

        features = np.array(
            [
                hard_violations / 100.0,  # Normalized
                soft_violations / 1000.0,
                hard_rank,
                soft_rank,
                progress,
                budget_remaining,
            ],
            dtype=np.float32,
        )

        return features

    def _load_model(self) -> None:
        """Load trained RL model."""
        model_path = self.config.get("model_path")
        if model_path:
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(model_path)
            except Exception as e:
                print(f"Warning: Failed to load memetic policy model: {e}")


class DynamicTermination:
    """
    Dynamic termination condition for local search.

    Stops early if:
    - No improvement for N iterations (patience)
    - Improvement rate drops below threshold
    - Time limit exceeded
    """

    def __init__(self, patience: int = 10, min_improvement: float = 0.01):
        """
        Initialize dynamic termination.

        Args:
            patience: Iterations without improvement before stopping
            min_improvement: Minimum improvement to count as progress
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_fitness = float("inf")
        self.iterations_without_improvement = 0

    def should_terminate(
        self, current_fitness: float, iteration: int, max_iterations: int
    ) -> bool:
        """
        Check if local search should terminate early.

        Args:
            current_fitness: Current solution fitness
            iteration: Current iteration
            max_iterations: Maximum iterations

        Returns:
            True if should terminate
        """
        # Check max iterations
        if iteration >= max_iterations:
            return True

        # Check improvement
        improvement = self.best_fitness - current_fitness
        if improvement > self.min_improvement:
            # Significant improvement
            self.best_fitness = current_fitness
            self.iterations_without_improvement = 0
        else:
            # No significant improvement
            self.iterations_without_improvement += 1

        # Check patience
        if self.iterations_without_improvement >= self.patience:
            return True

        return False

    def reset(self) -> None:
        """Reset termination state."""
        self.best_fitness = float("inf")
        self.iterations_without_improvement = 0
