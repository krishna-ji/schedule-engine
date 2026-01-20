"""
Base reward calculator interface.

All reward calculators implement this interface for consistent usage.
"""

from abc import ABC, abstractmethod

from src.domain.types import Individual


class BaseRewardCalculator(ABC):
    """
    Abstract base class for reward calculation strategies.

    Reward calculators convert fitness improvements and state changes
    into scalar reward signals for RL training.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize reward calculator.

        Args:
            config: Configuration dictionary with reward parameters
        """
        self.config = config or {}

    @abstractmethod
    def calculate(
        self,
        prev_population: list[Individual],
        current_population: list[Individual],
        action_cost: float = 0.0,
    ) -> float:
        """
        Calculate reward for a transition.

        Args:
            prev_population: Population before action
            current_population: Population after action
            action_cost: Cost of the action taken (optional)

        Returns:
            Scalar reward value (positive = good, negative = bad)
        """
        pass

    def get_fitness_values(
        self, population: list[Individual]
    ) -> list[tuple[float, float]]:
        """
        Extract fitness values from population.

        Args:
            population: GA population

        Returns:
            List of (hard_violations, soft_violations) tuples
        """
        return [ind.fitness.values for ind in population]  # type: ignore[attr-defined]

    def get_best_fitness(self, population: list[Individual]) -> tuple[float, float]:
        """
        Get best fitness in population (lowest violations).

        Args:
            population: GA population

        Returns:
            (hard_violations, soft_violations) of best individual
        """
        if not population:
            return (float("inf"), float("inf"))

        fitness_values = self.get_fitness_values(population)
        # Sort by hard violations first, soft violations second
        best = min(fitness_values, key=lambda x: (x[0], x[1]))
        return best

    def get_avg_fitness(self, population: list[Individual]) -> tuple[float, float]:
        """
        Get average fitness in population.

        Args:
            population: GA population

        Returns:
            (avg_hard_violations, avg_soft_violations)
        """
        if not population:
            return (float("inf"), float("inf"))

        fitness_values = self.get_fitness_values(population)
        avg_hard = sum(f[0] for f in fitness_values) / len(fitness_values)
        avg_soft = sum(f[1] for f in fitness_values) / len(fitness_values)
        return (avg_hard, avg_soft)
