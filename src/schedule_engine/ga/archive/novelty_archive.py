"""
Novelty archive for maintaining behavioral diversity.

ENHANCEMENT #5: Novelty search implementation.
"""

from collections import deque

import numpy as np
from numpy.typing import NDArray

from schedule_engine.domain.types import Individual
from schedule_engine.ga.archive.behavioral_descriptors import BehavioralDescriptors


class NoveltyArchive:
    """
    Maintains archive of behaviorally novel solutions.

    Novelty metric: Average distance to k-nearest neighbors in behavior space.
    Solutions with high novelty (far from existing solutions) are added to archive.

    Use cases:
    - Escape local optima by exploring behavioral space
    - Maintain diverse solution portfolio
    - Inject novel solutions into NSGA-II population
    """

    def __init__(
        self,
        max_size: int = 100,
        k_nearest: int = 15,
        novelty_threshold: float = 0.1,
    ):
        """
        Initialize novelty archive.

        Args:
            max_size: Maximum archive size
            k_nearest: Number of nearest neighbors for novelty calculation
            novelty_threshold: Minimum novelty to add to archive
        """
        self.max_size = max_size
        self.k_nearest = k_nearest
        self.novelty_threshold = novelty_threshold

        # Archive: list of (individual, behavioral_descriptor, fitness)
        self.archive: deque = deque(maxlen=max_size)

        # Behavioral descriptor extractor
        self.descriptor_extractor = BehavioralDescriptors()

    def compute_novelty(
        self, individual: Individual, descriptor: NDArray[np.float64] | None = None
    ) -> float:
        """
        Compute novelty of an individual.

        Novelty = average distance to k-nearest neighbors in behavior space.

        Args:
            individual: Individual to evaluate
            descriptor: Pre-computed behavioral descriptor (optional)

        Returns:
            Novelty score (higher = more novel)
        """
        if descriptor is None:
            descriptor = self.descriptor_extractor.extract(individual)

        if len(self.archive) == 0:
            return float("inf")  # First individual is maximally novel

        # Calculate distances to all archive members
        distances = []
        for _, archived_desc, _ in self.archive:
            dist = self.descriptor_extractor.distance(descriptor, archived_desc)
            distances.append(dist)

        # Average distance to k-nearest neighbors
        k = min(self.k_nearest, len(distances))
        k_nearest_distances = sorted(distances)[:k]

        return float(np.mean(k_nearest_distances))

    def add_if_novel(
        self, individual: Individual, descriptor: NDArray[np.float64] | None = None
    ) -> bool:
        """
        Add individual to archive if sufficiently novel.

        Args:
            individual: Individual to potentially add
            descriptor: Pre-computed behavioral descriptor

        Returns:
            True if added to archive
        """
        if descriptor is None:
            descriptor = self.descriptor_extractor.extract(individual)

        novelty = self.compute_novelty(individual, descriptor)

        if novelty >= self.novelty_threshold:
            # Novel enough to add
            fitness = individual.fitness.values  # type: ignore[attr-defined]
            self.archive.append((individual, descriptor, fitness))
            return True

        return False

    def get_novel_individuals(
        self, n: int = 10, prefer_feasible: bool = True
    ) -> list[Individual]:
        """
        Get n most novel individuals from archive for injection.

        Args:
            n: Number of individuals to retrieve
            prefer_feasible: Prioritize feasible solutions

        Returns:
            List of novel individuals
        """
        if len(self.archive) == 0:
            return []

        if prefer_feasible:
            # Filter feasible solutions (hard violations == 0)
            feasible = [
                (ind, desc, fit) for ind, desc, fit in self.archive if fit[0] == 0
            ]
            if len(feasible) >= n:
                # Randomly sample from feasible
                indices = np.random.choice(len(feasible), size=n, replace=False)
                return [feasible[i][0] for i in indices]

        # Sample from entire archive
        n_sample = min(n, len(self.archive))
        indices = np.random.choice(len(self.archive), size=n_sample, replace=False)
        return [self.archive[i][0] for i in indices]

    def get_statistics(self) -> dict:
        """Get archive statistics."""
        if len(self.archive) == 0:
            return {
                "size": 0,
                "avg_novelty": 0.0,
                "diversity": 0.0,
            }

        # Calculate pairwise diversity in archive
        descriptors = [desc for _, desc, _ in self.archive]
        if len(descriptors) > 1:
            distances = []
            for i in range(len(descriptors)):
                for j in range(i + 1, len(descriptors)):
                    dist = self.descriptor_extractor.distance(
                        descriptors[i], descriptors[j]
                    )
                    distances.append(dist)
            diversity = float(np.mean(distances))
        else:
            diversity = 0.0

        return {
            "size": len(self.archive),
            "diversity": diversity,
            "feasible_count": sum(1 for _, _, fit in self.archive if fit[0] == 0),
        }

    def reset(self) -> None:
        """Clear archive."""
        self.archive.clear()
