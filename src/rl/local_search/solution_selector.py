"""
Solution selector for local search targeting.

ENHANCEMENT #6: Selects which solutions to apply local search to.
"""

from typing import List
import numpy as np

from src.core.types import Individual


class SolutionSelector:
    """
    Selects solutions for local search intensification.

    Strategies:
    - UCB (Upper Confidence Bound): Balance exploration/exploitation
    - Elite: Always select best solutions
    - Diverse: Select diverse solutions
    - Stochastic: Random selection with fitness-based probability
    """

    def __init__(self, strategy: str = "ucb", num_solutions: int = 5):
        """
        Initialize solution selector.

        Args:
            strategy: 'ucb', 'elite', 'diverse', or 'stochastic'
            num_solutions: Number of solutions to select
        """
        self.strategy = strategy
        self.num_solutions = num_solutions

        # UCB statistics
        self.solution_attempts = {}
        self.solution_improvements = {}

    def select(self, population: List[Individual]) -> List[Individual]:
        """
        Select solutions for local search.

        Args:
            population: GA population

        Returns:
            List of selected individuals
        """
        if len(population) <= self.num_solutions:
            return population

        if self.strategy == "elite":
            return self._select_elite(population)
        elif self.strategy == "diverse":
            return self._select_diverse(population)
        elif self.strategy == "ucb":
            return self._select_ucb(population)
        elif self.strategy == "stochastic":
            return self._select_stochastic(population)
        else:
            return population[: self.num_solutions]

    def _select_elite(self, population: List[Individual]) -> List[Individual]:
        """Select best N solutions."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values)
        return sorted_pop[: self.num_solutions]

    def _select_diverse(self, population: List[Individual]) -> List[Individual]:
        """Select diverse solutions using greedy diversification."""
        selected = []
        candidates = list(population)

        # Start with best solution
        best_idx = np.argmin([ind.fitness.values for ind in candidates])
        selected.append(candidates.pop(best_idx))

        # Greedily add most diverse solutions
        while len(selected) < self.num_solutions and candidates:
            max_diversity = -1
            max_idx = 0

            for i, candidate in enumerate(candidates):
                # Calculate minimum distance to selected solutions
                min_dist = min(
                    self._fitness_distance(candidate, sel) for sel in selected
                )
                if min_dist > max_diversity:
                    max_diversity = min_dist
                    max_idx = i

            selected.append(candidates.pop(max_idx))

        return selected

    def _select_ucb(
        self, population: List[Individual], exploration_param: float = 1.0
    ) -> List[Individual]:
        """
        Select using Upper Confidence Bound.

        UCB(i) = avg_improvement(i) + c * sqrt(ln(N) / attempts(i))
        """
        total_attempts = sum(self.solution_attempts.values())

        ucb_scores = []
        for ind in population:
            ind_id = id(ind)
            attempts = self.solution_attempts.get(ind_id, 0)
            improvements = self.solution_improvements.get(ind_id, 0.0)

            if attempts == 0:
                # Unvisited: infinite UCB
                ucb = float("inf")
            else:
                avg_improvement = improvements / attempts
                exploration = exploration_param * np.sqrt(
                    np.log(total_attempts + 1) / attempts
                )
                ucb = avg_improvement + exploration

            ucb_scores.append(ucb)

        # Select top N by UCB
        top_indices = np.argsort(ucb_scores)[-self.num_solutions :]
        return [population[i] for i in top_indices]

    def _select_stochastic(self, population: List[Individual]) -> List[Individual]:
        """Select with probability proportional to fitness quality."""
        # Calculate selection probabilities (better fitness = higher probability)
        fitness_values = np.array([ind.fitness.values for ind in population])
        combined_fitness = fitness_values[:, 0] * 100 + fitness_values[:, 1]

        # Invert (lower fitness = better)
        max_fitness = np.max(combined_fitness)
        probabilities = max_fitness - combined_fitness + 1  # Add 1 to avoid zero
        probabilities = probabilities / np.sum(probabilities)

        # Sample without replacement
        indices = np.random.choice(
            len(population),
            size=min(self.num_solutions, len(population)),
            replace=False,
            p=probabilities,
        )

        return [population[i] for i in indices]

    def update_statistics(self, individual: Individual, improvement: float) -> None:
        """
        Update UCB statistics after local search.

        Args:
            individual: Solution that was intensified
            improvement: Fitness improvement achieved
        """
        ind_id = id(individual)
        self.solution_attempts[ind_id] = self.solution_attempts.get(ind_id, 0) + 1
        self.solution_improvements[ind_id] = (
            self.solution_improvements.get(ind_id, 0.0) + improvement
        )

    @staticmethod
    def _fitness_distance(ind1: Individual, ind2: Individual) -> float:
        """Calculate Euclidean distance in fitness space."""
        f1 = np.array(ind1.fitness.values)
        f2 = np.array(ind2.fitness.values)
        return float(np.linalg.norm(f1 - f2))
