"""
MAP-Elites algorithm for quality-diversity optimization.

ENHANCEMENT #5: Maintain archive of diverse high-quality solutions.
"""

import numpy as np
from numpy.typing import NDArray

from src.core.types import Individual
from src.ga.archive.behavioral_descriptors import BehavioralDescriptors


class MAPElites:
    """
    MAP-Elites: Illumination algorithm for quality-diversity.

    Maintains a feature map where each cell contains the best solution
    for that behavioral region. Produces diverse portfolios of high-quality
    solutions covering different trade-offs.

    Feature map dimensions (example):
    - Axis 1: Temporal distribution (5 bins: concentrated → spread)
    - Axis 2: Compactness (5 bins: compact → sparse)
    - Result: 5×5 = 25 cells, each containing best solution for that region
    """

    def __init__(
        self,
        feature_dimensions: list[tuple[int, int]] = None,
        feature_bins: int = 5,
    ):
        """
        Initialize MAP-Elites archive.

        Args:
            feature_dimensions: List of (start_idx, end_idx) for each axis
                               Default: [(0, 7), (15, 17)] = temporal + compactness
            feature_bins: Number of bins per dimension
        """
        self.feature_dimensions = feature_dimensions or [(0, 7), (15, 17)]
        self.feature_bins = feature_bins

        # Feature map: dict mapping (bin_x, bin_y, ...) → (individual, fitness)
        self.feature_map: dict[
            tuple[int, ...], tuple[Individual, tuple[float, float]]
        ] = {}

        # Behavioral descriptor extractor
        self.descriptor_extractor = BehavioralDescriptors()

    def get_feature_indices(self, descriptor: NDArray[np.float64]) -> tuple[int, ...]:
        """
        Map behavioral descriptor to feature map cell indices.

        Args:
            descriptor: 17D behavioral descriptor

        Returns:
            Tuple of bin indices (one per dimension)
        """
        indices = []

        for dim_start, dim_end in self.feature_dimensions:
            # Extract features for this dimension
            dim_features = descriptor[dim_start:dim_end]

            # Aggregate to single value (e.g., mean or max)
            dim_value = float(np.mean(dim_features))

            # Discretize to bin [0, feature_bins-1]
            bin_idx = int(
                np.clip(dim_value * self.feature_bins, 0, self.feature_bins - 1)
            )
            indices.append(bin_idx)

        return tuple(indices)

    def add_or_replace(
        self, individual: Individual, descriptor: NDArray[np.float64] = None
    ) -> bool:
        """
        Add individual to feature map (or replace if better).

        Args:
            individual: Individual to add
            descriptor: Pre-computed behavioral descriptor

        Returns:
            True if added/replaced an existing solution
        """
        if descriptor is None:
            descriptor = self.descriptor_extractor.extract(individual)

        # Get feature map cell
        cell_indices = self.get_feature_indices(descriptor)
        fitness = individual.fitness.values

        # Check if cell is empty or this individual is better
        if cell_indices not in self.feature_map:
            # Empty cell: add
            self.feature_map[cell_indices] = (individual, fitness)
            return True
        else:
            # Cell occupied: replace if better
            _, existing_fitness = self.feature_map[cell_indices]

            # Compare fitness (lexicographic: hard first, then soft)
            is_better = fitness[0] < existing_fitness[0] or (
                fitness[0] == existing_fitness[0] and fitness[1] < existing_fitness[1]
            )

            if is_better:
                self.feature_map[cell_indices] = (individual, fitness)
                return True

        return False

    def get_all_elites(self) -> list[Individual]:
        """
        Get all elite solutions from feature map.

        Returns:
            List of individuals (one per occupied cell)
        """
        return [ind for ind, _ in self.feature_map.values()]

    def get_random_elites(
        self, n: int, prefer_feasible: bool = True
    ) -> list[Individual]:
        """
        Sample n random elites for injection into population.

        Args:
            n: Number of elites to sample
            prefer_feasible: Prioritize feasible solutions

        Returns:
            List of sampled individuals
        """
        if len(self.feature_map) == 0:
            return []

        if prefer_feasible:
            # Filter feasible solutions
            feasible_cells = [
                cell_idx
                for cell_idx, (_, fit) in self.feature_map.items()
                if fit[0] == 0
            ]
            if len(feasible_cells) >= n:
                # Sample from feasible
                sampled_cells = np.random.choice(feasible_cells, size=n, replace=False)
                return [self.feature_map[cell][0] for cell in sampled_cells]

        # Sample from all elites
        n_sample = min(n, len(self.feature_map))
        all_cells = list(self.feature_map.keys())
        sampled_cells = np.random.choice(len(all_cells), size=n_sample, replace=False)
        return [self.feature_map[all_cells[i]][0] for i in sampled_cells]

    def get_coverage(self) -> float:
        """
        Get feature map coverage (% of cells occupied).

        Returns:
            Coverage percentage in [0, 1]
        """
        total_cells = self.feature_bins ** len(self.feature_dimensions)
        occupied_cells = len(self.feature_map)
        return occupied_cells / total_cells

    def get_statistics(self) -> dict[str, float]:
        """Get archive statistics."""
        if len(self.feature_map) == 0:
            return {
                "coverage": 0.0,
                "num_elites": 0,
                "feasible_elites": 0,
                "best_fitness": (float("inf"), float("inf")),
            }

        all_fitness = [fit for _, fit in self.feature_map.values()]
        best_fitness = min(all_fitness, key=lambda f: (f[0], f[1]))
        feasible_count = sum(1 for fit in all_fitness if fit[0] == 0)

        return {
            "coverage": self.get_coverage(),
            "num_elites": len(self.feature_map),
            "feasible_elites": feasible_count,
            "best_fitness": best_fitness,
        }

    def reset(self) -> None:
        """Clear feature map."""
        self.feature_map.clear()
