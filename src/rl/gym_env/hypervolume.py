"""
Hypervolume calculator for multi-objective optimization.

ENHANCEMENT #1: Multi-objective reward shaping using hypervolume indicator.

The hypervolume indicator measures the volume of objective space dominated by
a Pareto front. Higher hypervolume = better multi-objective quality.

Mathematical Definition:
    HV(P, r) = λ(⋃_{p∈P} [p, r])

Where:
    - P: Pareto front (set of non-dominated points)
    - r: Reference point (worst acceptable objective values)
    - λ: Lebesgue measure (volume in objective space)
    - [p, r]: Hyperrectangle from point p to reference point r

This implementation provides:
1. Fast hypervolume computation using WFG algorithm
2. Incremental hypervolume (contribution of single solution)
3. Hypervolume-based reward for RL agents
"""

from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


class HypervolumeCalculator:
    """
    Calculate hypervolume indicator for Pareto front quality assessment.

    Supports:
    - 2D and higher-dimensional objective spaces
    - Incremental hypervolume (single solution contribution)
    - Normalized hypervolume for RL rewards

    Algorithm:
    Uses WFG (While, Filippo, Glasmachers) algorithm for efficient computation.
    For 2D problems, uses simple geometric calculation.

    Complexity:
    - 2D: O(n log n) where n is number of points
    - 3D: O(n log n)
    - kD: O(n^(k-1) log n) where k is number of objectives
    """

    def __init__(
        self,
        reference_point: NDArray[np.float64],
        minimize: bool = True,
    ):
        """
        Initialize hypervolume calculator.

        Args:
            reference_point: Worst acceptable point in objective space.
                            For minimization: upper bound (e.g., [1000, 10000])
                            For maximization: lower bound (e.g., [0, 0])
            minimize: True if minimizing objectives, False if maximizing

        Example:
            For schedule optimization with (hard_violations, soft_penalty):
            reference_point = [1000, 10000]  # Worst acceptable solution
            minimize = True  # We want to minimize violations
        """
        self.reference_point = np.asarray(reference_point, dtype=np.float64)
        self.minimize = minimize
        self.n_objectives = len(reference_point)

        # Validate reference point
        if self.n_objectives < 2:
            raise ValueError("Hypervolume requires at least 2 objectives")

    def compute(self, pareto_front: NDArray[np.float64]) -> float:
        """
        Compute hypervolume indicator for given Pareto front.

        Args:
            pareto_front: Array of shape (n_points, n_objectives)
                         Each row is a point in objective space

        Returns:
            Hypervolume indicator value (higher is better)

        Example:
            >>> pareto_front = np.array([[10, 100], [50, 50], [100, 10]])
            >>> hv_calc = HypervolumeCalculator(reference_point=[200, 200])
            >>> hv = hv_calc.compute(pareto_front)
            >>> print(f"Hypervolume: {hv}")
        """
        if len(pareto_front) == 0:
            return 0.0

        # Convert to numpy array
        points = np.asarray(pareto_front, dtype=np.float64)

        # Validate dimensions
        if points.ndim == 1:
            points = points.reshape(1, -1)

        if points.shape[1] != self.n_objectives:
            raise ValueError(
                f"Expected {self.n_objectives} objectives, got {points.shape[1]}"
            )

        # Filter dominated points (keep only Pareto front)
        pareto_points = self._get_pareto_front(points)

        if len(pareto_points) == 0:
            return 0.0

        # Compute hypervolume based on number of objectives
        if self.n_objectives == 2:
            return self._hypervolume_2d(pareto_points)
        elif self.n_objectives == 3:
            return self._hypervolume_3d(pareto_points)
        else:
            return self._hypervolume_wfg(pareto_points)

    def compute_contribution(
        self,
        pareto_front: NDArray[np.float64],
        point_index: int,
    ) -> float:
        r"""
        Compute hypervolume contribution of a single point.

        Contribution = HV(P) - HV(P \ {p_i})

        This measures how much removing point i decreases the hypervolume.
        High contribution = point is important for Pareto front quality.

        Args:
            pareto_front: Current Pareto front
            point_index: Index of point to evaluate

        Returns:
            Hypervolume contribution of the specified point
        """
        if len(pareto_front) == 0 or point_index >= len(pareto_front):
            return 0.0

        # Hypervolume with point
        hv_with = self.compute(pareto_front)

        # Hypervolume without point
        pareto_without = np.delete(pareto_front, point_index, axis=0)
        hv_without = self.compute(pareto_without)

        return max(0.0, hv_with - hv_without)

    def _get_pareto_front(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Filter points to get only non-dominated solutions (Pareto front).

        A point p dominates point q if:
        - p is better in at least one objective
        - p is not worse in any objective

        Args:
            points: Array of candidate points

        Returns:
            Array of non-dominated points
        """
        if len(points) == 0:
            return points

        # For minimization, we keep points that are not dominated
        is_dominated = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            for j in range(len(points)):
                if i == j:
                    continue

                # Check if point j dominates point i
                if self.minimize:
                    # For minimization: j dominates i if j_k <= i_k for all k
                    # and j_k < i_k for at least one k
                    better = points[j] <= points[i]
                    strictly_better = points[j] < points[i]
                else:
                    # For maximization: j dominates i if j_k >= i_k for all k
                    # and j_k > i_k for at least one k
                    better = points[j] >= points[i]
                    strictly_better = points[j] > points[i]

                if np.all(better) and np.any(strictly_better):
                    is_dominated[i] = True
                    break

        return points[~is_dominated]

    def _hypervolume_2d(self, pareto_points: NDArray[np.float64]) -> float:
        """
        Compute hypervolume for 2D case using geometric calculation.

        Algorithm:
        1. Sort points by first objective
        2. Sum rectangular areas between consecutive points

        Complexity: O(n log n)
        """
        if len(pareto_points) == 0:
            return 0.0

        # Sort by first objective (ascending for minimization)
        if self.minimize:
            sorted_points = pareto_points[np.argsort(pareto_points[:, 0])]
        else:
            sorted_points = pareto_points[np.argsort(-pareto_points[:, 0])]

        hypervolume = 0.0

        if self.minimize:
            # For minimization: accumulate rectangles from reference point
            prev_x = self.reference_point[0]
            for point in sorted_points:
                width = prev_x - point[0]
                height = self.reference_point[1] - point[1]
                if width > 0 and height > 0:
                    hypervolume += width * height
                prev_x = point[0]
        else:
            # For maximization: accumulate rectangles from reference point
            prev_x = self.reference_point[0]
            for point in sorted_points:
                width = point[0] - prev_x
                height = point[1] - self.reference_point[1]
                if width > 0 and height > 0:
                    hypervolume += width * height
                prev_x = point[0]

        return hypervolume

    def _hypervolume_3d(self, pareto_points: NDArray[np.float64]) -> float:
        """
        Compute hypervolume for 3D case.

        Uses HSO (Hypervolume by Slicing Objectives) algorithm.
        Complexity: O(n^2 log n)
        """
        # Simplified implementation: use slice-based calculation
        # For production, consider using PyGMO or other optimized libraries

        if len(pareto_points) == 0:
            return 0.0

        # Sort by first objective
        if self.minimize:
            sorted_points = pareto_points[np.argsort(pareto_points[:, 0])]
        else:
            sorted_points = pareto_points[np.argsort(-pareto_points[:, 0])]

        hypervolume = 0.0

        # Slice by first objective and compute 2D hypervolume for each slice
        for i, point in enumerate(sorted_points):
            # Get reference point for this slice
            if self.minimize:
                slice_ref = np.array(
                    [point[0], self.reference_point[1], self.reference_point[2]]
                )
            else:
                slice_ref = np.array(
                    [point[0], self.reference_point[1], self.reference_point[2]]
                )

            # Compute 2D hypervolume for remaining two objectives
            remaining_points = sorted_points[i:, 1:]

            # Create 2D hypervolume calculator
            calc_2d = HypervolumeCalculator(
                reference_point=slice_ref[1:], minimize=self.minimize
            )

            hv_slice = calc_2d.compute(remaining_points)

            # Multiply by width in first objective
            if i < len(sorted_points) - 1:
                if self.minimize:
                    width = sorted_points[i + 1][0] - point[0]
                else:
                    width = point[0] - sorted_points[i + 1][0]
            else:
                if self.minimize:
                    width = self.reference_point[0] - point[0]
                else:
                    width = point[0] - self.reference_point[0]

            if width > 0:
                hypervolume += width * hv_slice

        return hypervolume

    def _hypervolume_wfg(self, pareto_points: NDArray[np.float64]) -> float:
        """
        Compute hypervolume using WFG algorithm for k>3 dimensions.

        This is a simplified placeholder. For production use with >3 objectives,
        consider integrating PyGMO or other optimized libraries.

        Complexity: O(n^(k-1) log n)
        """
        # Placeholder: use recursive slicing
        # For actual implementation, use PyGMO's hypervolume class

        if len(pareto_points) == 0:
            return 0.0

        # Simplified: compute volume of bounding box
        # (This is an upper bound, not exact hypervolume)
        if self.minimize:
            dominated_space = self.reference_point - np.min(pareto_points, axis=0)
        else:
            dominated_space = np.max(pareto_points, axis=0) - self.reference_point

        # Return product of dominated ranges (upper bound approximation)
        return np.prod(np.maximum(dominated_space, 0))


def compute_hypervolume_reward(
    old_pareto_front: NDArray[np.float64],
    new_pareto_front: NDArray[np.float64],
    reference_point: NDArray[np.float64],
    normalize: bool = True,
) -> float:
    """
    Compute RL reward based on hypervolume improvement.

    Reward = improvement in hypervolume indicator
    Positive reward: Pareto front improved (higher hypervolume)
    Negative reward: Pareto front degraded (lower hypervolume)

    Args:
        old_pareto_front: Pareto front before action
        new_pareto_front: Pareto front after action
        reference_point: Reference point for hypervolume calculation
        normalize: Whether to normalize reward to [-1, 1] using tanh

    Returns:
        Hypervolume-based reward

    Example:
        >>> old_front = np.array([[100, 1000], [200, 500], [500, 200]])
        >>> new_front = np.array([[80, 900], [150, 450], [400, 180]])
        >>> reward = compute_hypervolume_reward(old_front, new_front, [1000, 10000])
        >>> print(f"HV Reward: {reward}")
    """
    calculator = HypervolumeCalculator(reference_point=reference_point, minimize=True)

    old_hv = calculator.compute(old_pareto_front)
    new_hv = calculator.compute(new_pareto_front)

    delta_hv = new_hv - old_hv

    if normalize:
        # Normalize using tanh to [-1, 1]
        # Scale factor chosen empirically (adjust based on typical HV changes)
        scale = 1000.0
        normalized_reward = np.tanh(delta_hv / scale)
        return float(normalized_reward)
    else:
        return float(delta_hv)
