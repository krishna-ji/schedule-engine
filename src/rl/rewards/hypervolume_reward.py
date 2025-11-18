"""
Hypervolume indicator reward calculator.

ENHANCEMENT #1: Pareto-aware reward using hypervolume improvement.

Hypervolume measures the volume of objective space dominated by the Pareto front.
Increasing hypervolume = improving the Pareto front = good reward.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray

from src.core.types import Individual
from src.rl.rewards.base_reward import BaseRewardCalculator

try:
    from pymoo.indicators.hv import HV

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not installed. Hypervolume reward unavailable.")
    print("Install with: pip install pymoo")


class HypervolumeReward(BaseRewardCalculator):
    """
    Hypervolume indicator reward for multi-objective optimization.

    Reward = HV(current_front) - HV(previous_front)

    Where HV is the hypervolume indicator calculated relative to a reference point.

    Benefits:
    - Pareto-aware (rewards any Pareto improvement)
    - Encourages diversity (spread along front)
    - Theoretically sound (monotonic with Pareto dominance)

    Limitations:
    - Computationally expensive for large populations (O(n log n))
    - Requires reference point selection
    """

    def __init__(self, config: dict = None):
        """
        Initialize hypervolume reward calculator.

        Args:
            config: Configuration with:
                - reference_point: Reference point for HV calculation
                                  (default: [100, 1000] for hard/soft)
                - normalize: Whether to normalize HV to [0, 1]
                - min_improvement: Minimum HV change to count as reward
        """
        super().__init__(config)

        if not PYMOO_AVAILABLE:
            raise ImportError(
                "pymoo required for hypervolume reward. "
                "Install with: pip install pymoo"
            )

        # Reference point (worst acceptable fitness)
        self.reference_point = np.array(
            self.config.get("reference_point", [100.0, 1000.0])
        )
        self.normalize = self.config.get("normalize", True)
        self.min_improvement = self.config.get("min_improvement", 0.01)

        # Initialize HV calculator
        self.hv_calculator = HV(ref_point=self.reference_point)

    def calculate(
        self,
        prev_population: List[Individual],
        current_population: List[Individual],
        action_cost: float = 0.0,
    ) -> float:
        """
        Calculate reward based on hypervolume improvement.

        Args:
            prev_population: Population before action
            current_population: Population after action
            action_cost: Cost of the action taken (ignored for HV)

        Returns:
            Hypervolume improvement (positive if Pareto front improved)
        """
        # Extract Pareto fronts
        prev_front = self._extract_pareto_front(prev_population)
        curr_front = self._extract_pareto_front(current_population)

        # Calculate hypervolumes
        prev_hv = self._calculate_hypervolume(prev_front)
        curr_hv = self._calculate_hypervolume(curr_front)

        # Reward = improvement in hypervolume
        hv_improvement = curr_hv - prev_hv

        # Apply minimum improvement threshold
        if abs(hv_improvement) < self.min_improvement:
            return 0.0

        # Optional normalization
        if self.normalize:
            # Normalize by maximum possible hypervolume
            max_hv = np.prod(self.reference_point)
            hv_improvement = hv_improvement / (max_hv + 1e-6)

        return float(hv_improvement)

    def _extract_pareto_front(
        self, population: List[Individual]
    ) -> NDArray[np.float64]:
        """
        Extract non-dominated solutions (Pareto front) from population.

        Args:
            population: GA population

        Returns:
            Array of shape (n, 2) with fitness values of Pareto front
        """
        if not population:
            return np.array([]).reshape(0, 2)

        # Get all fitness values
        fitness_array = np.array(self.get_fitness_values(population))

        # Find non-dominated solutions
        is_pareto = self._is_pareto_efficient(fitness_array)
        pareto_front = fitness_array[is_pareto]

        return pareto_front

    def _is_pareto_efficient(self, costs: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        Find Pareto-efficient points (minimization problem).

        Args:
            costs: Array of shape (n, 2) with (hard, soft) violations

        Returns:
            Boolean array indicating which points are Pareto-efficient
        """
        n = len(costs)
        if n == 0:
            return np.array([], dtype=bool)

        is_efficient = np.ones(n, dtype=bool)

        for i in range(n):
            if is_efficient[i]:
                # Point i is efficient if no other point dominates it
                # Point j dominates i if j <= i in all objectives AND j < i in at least one
                dominated = np.all(costs <= costs[i], axis=1) & np.any(
                    costs < costs[i], axis=1
                )
                is_efficient[dominated] = False
                is_efficient[i] = True  # Restore i after checking dominance

        return is_efficient

    def _calculate_hypervolume(self, front: NDArray[np.float64]) -> float:
        """
        Calculate hypervolume of Pareto front.

        Args:
            front: Array of shape (n, 2) with fitness values

        Returns:
            Hypervolume value (0.0 if empty front)
        """
        if len(front) == 0:
            return 0.0

        # Check if all points are dominated by reference point
        if np.any(front >= self.reference_point):
            # Some points are worse than reference - clip to reference
            front = np.minimum(front, self.reference_point)

        try:
            hv = self.hv_calculator.do(front)
            return float(hv)
        except Exception as e:
            print(f"Warning: HV calculation failed: {e}")
            return 0.0

    def get_pareto_front_size(self, population: List[Individual]) -> int:
        """
        Get number of solutions in Pareto front.

        Useful metric for monitoring diversity.

        Args:
            population: GA population

        Returns:
            Number of non-dominated solutions
        """
        front = self._extract_pareto_front(population)
        return len(front)
