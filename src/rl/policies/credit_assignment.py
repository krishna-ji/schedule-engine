"""
Credit assignment system for tracking operator success.

ENHANCEMENT #3: Tracks which probability settings lead to good offspring.
"""

from typing import Dict, List, Tuple
from collections import deque, defaultdict
import numpy as np


class CreditAssignmentTracker:
    """
    Tracks credit assignment for probability settings.

    Records which (cxpb, mutpb) combinations produce good offspring,
    enabling the RL agent to learn effective probability policies.

    Metrics tracked:
    - Offspring fitness improvement
    - Feasibility transitions (infeasible → feasible)
    - Diversity contribution
    - Success rate over time
    """

    def __init__(self, history_size: int = 100):
        """
        Initialize credit assignment tracker.

        Args:
            history_size: Number of recent assignments to track
        """
        self.history_size = history_size

        # Track recent assignments: (cxpb, mutpb, fitness_improvement, was_successful)
        self.history: deque = deque(maxlen=history_size)

        # Success statistics by probability bucket
        self.success_counts: Dict[Tuple[float, float], int] = defaultdict(int)
        self.attempt_counts: Dict[Tuple[float, float], int] = defaultdict(int)

        # Running statistics
        self.total_attempts = 0
        self.total_successes = 0
        self.total_fitness_improvement = 0.0

    def record_outcome(
        self,
        cxpb: float,
        mutpb: float,
        parent_fitness: Tuple[float, float],
        offspring_fitness: Tuple[float, float],
        was_feasibility_transition: bool = False,
    ) -> None:
        """
        Record outcome of applying operator with given probabilities.

        Args:
            cxpb: Crossover probability used
            mutpb: Mutation probability used
            parent_fitness: (hard, soft) violations of parent
            offspring_fitness: (hard, soft) violations of offspring
            was_feasibility_transition: Whether offspring achieved feasibility
        """
        # Calculate fitness improvement (positive = better)
        hard_improvement = parent_fitness[0] - offspring_fitness[0]
        soft_improvement = parent_fitness[1] - offspring_fitness[1]

        # Combined improvement (weighted)
        total_improvement = 100 * hard_improvement + soft_improvement

        # Success criteria:
        # 1. Any improvement in fitness
        # 2. Feasibility transition
        is_successful = total_improvement > 0 or was_feasibility_transition

        # Record in history
        self.history.append((cxpb, mutpb, total_improvement, is_successful))

        # Update statistics (bucket probabilities to nearest 0.1)
        prob_bucket = (round(cxpb, 1), round(mutpb, 1))
        self.attempt_counts[prob_bucket] += 1
        if is_successful:
            self.success_counts[prob_bucket] += 1

        # Update running totals
        self.total_attempts += 1
        if is_successful:
            self.total_successes += 1
        self.total_fitness_improvement += total_improvement

    def get_success_rate(
        self, cxpb: float, mutpb: float, window_size: int = None
    ) -> float:
        """
        Get success rate for given probability setting.

        Args:
            cxpb: Crossover probability
            mutpb: Mutation probability
            window_size: Number of recent attempts to consider (None = all)

        Returns:
            Success rate in [0, 1], or 0.5 if no data
        """
        prob_bucket = (round(cxpb, 1), round(mutpb, 1))

        if window_size is None:
            # Use all-time statistics
            attempts = self.attempt_counts[prob_bucket]
            successes = self.success_counts[prob_bucket]
        else:
            # Use recent history
            relevant_history = [
                (c, m, imp, success)
                for c, m, imp, success in self.history
                if abs(c - cxpb) < 0.15 and abs(m - mutpb) < 0.15
            ][-window_size:]

            attempts = len(relevant_history)
            successes = sum(1 for _, _, _, success in relevant_history if success)

        if attempts == 0:
            return 0.5  # Prior (assume 50% success for unknown settings)

        return successes / attempts

    def get_average_improvement(
        self, cxpb: float, mutpb: float, window_size: int = 50
    ) -> float:
        """
        Get average fitness improvement for given probability setting.

        Args:
            cxpb: Crossover probability
            mutpb: Mutation probability
            window_size: Number of recent attempts to consider

        Returns:
            Average fitness improvement (positive = better)
        """
        # Filter recent history
        relevant_history = [
            improvement
            for c, m, improvement, _ in self.history
            if abs(c - cxpb) < 0.15 and abs(m - mutpb) < 0.15
        ][-window_size:]

        if not relevant_history:
            return 0.0

        return float(np.mean(relevant_history))

    def get_best_probabilities(
        self, metric: str = "success_rate", min_attempts: int = 10
    ) -> Tuple[float, float]:
        """
        Get best probability setting based on historical performance.

        Args:
            metric: 'success_rate' or 'average_improvement'
            min_attempts: Minimum attempts required to consider a setting

        Returns:
            (cxpb, mutpb) with best performance
        """
        valid_buckets = [
            bucket
            for bucket, attempts in self.attempt_counts.items()
            if attempts >= min_attempts
        ]

        if not valid_buckets:
            return (0.75, 0.25)  # Default

        if metric == "success_rate":
            scores = {
                bucket: self.success_counts[bucket] / self.attempt_counts[bucket]
                for bucket in valid_buckets
            }
        else:
            # Average improvement
            scores = {}
            for bucket in valid_buckets:
                cxpb, mutpb = bucket
                scores[bucket] = self.get_average_improvement(cxpb, mutpb)

        best_bucket = max(scores.items(), key=lambda x: x[1])[0]
        return best_bucket

    def get_statistics(self) -> Dict[str, float]:
        """
        Get overall statistics.

        Returns:
            Dictionary with success rate, avg improvement, etc.
        """
        return {
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "overall_success_rate": (
                self.total_successes / self.total_attempts
                if self.total_attempts > 0
                else 0.0
            ),
            "average_improvement": (
                self.total_fitness_improvement / self.total_attempts
                if self.total_attempts > 0
                else 0.0
            ),
            "num_unique_settings": len(self.attempt_counts),
        }

    def get_probability_heatmap(self) -> np.ndarray:
        """
        Get heatmap of success rates across probability space.

        Returns:
            2D array of shape (10, 10) with success rates
            Rows = cxpb (0.0-1.0), Cols = mutpb (0.0-1.0)
        """
        heatmap = np.full((10, 10), 0.5)  # Initialize with prior

        for (cxpb, mutpb), attempts in self.attempt_counts.items():
            if attempts >= 5:  # Minimum confidence
                cx_idx = min(int(cxpb * 10), 9)
                mut_idx = min(int(mutpb * 10), 9)
                success_rate = self.success_counts[(cxpb, mutpb)] / attempts
                heatmap[cx_idx, mut_idx] = success_rate

        return heatmap

    def reset(self) -> None:
        """Reset all statistics."""
        self.history.clear()
        self.success_counts.clear()
        self.attempt_counts.clear()
        self.total_attempts = 0
        self.total_successes = 0
        self.total_fitness_improvement = 0.0
