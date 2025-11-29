"""
Behavioral archive for quality-diversity optimization.

ENHANCEMENT #6: Maintain archive of elite solutions for novelty search.

The behavioral archive stores a diverse set of high-quality solutions
characterized by their behavioral features (phenotype), not just fitness.

Archive Management Strategies:
1. **Novelty-based**: Add solutions with high novelty scores
2. **Quality-based**: Add solutions with good fitness
3. **Hybrid**: Balance novelty and quality (Pareto front in novelty-quality space)

Mathematical Formulation:
    Archive A = {(x₁, φ(x₁), f(x₁)), ..., (xₙ, φ(xₙ), f(xₙ))}

Where:
    - xᵢ: Solution (schedule individual)
    - φ(xᵢ): Behavioral features (phenotype)
    - f(xᵢ): Fitness value
    - n ≤ max_size: Archive capacity

Replacement Strategy:
When archive is full and new solution S arrives:
1. Compute novelty(S, A) and fitness(S)
2. If S dominates any archive member in (novelty, fitness) space: add S
3. If adding S, remove solution with worst (novelty, fitness) trade-off

References:
- Lehman & Stanley (2011): Novelty Search
- Mouret & Clune (2015): MAP-Elites (Illumination)
- Pugh et al. (2016): Quality Diversity
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.types import Individual
from src.diversity.novelty_metric import compute_novelty


@dataclass
class ArchiveEntry:
    """Single entry in behavioral archive."""

    individual: Individual
    behavioral_features: NDArray[np.float64]
    fitness: tuple[float, float]  # (hard_violations, soft_violations)
    novelty: float
    generation_added: int

    @property
    def combined_fitness(self) -> float:
        """Get combined fitness score (hard * 100 + soft)."""
        return abs(self.fitness[0]) * 100 + abs(self.fitness[1])


class BehavioralArchive:
    """
    Archive for storing behaviorally diverse elite solutions.

    Supports:
    - Novelty-based selection
    - Quality-based selection
    - Hybrid (novelty + quality) selection
    - Bounded archive size with replacement strategies

    Usage:
        >>> archive = BehavioralArchive(max_size=100, novelty_weight=0.7)
        >>> archive.add(individual, features, fitness, generation)
        >>> novelty = archive.compute_novelty(features)
        >>> diverse_solutions = archive.get_diverse_subset(k=10)
    """

    def __init__(
        self,
        max_size: int = 100,
        novelty_weight: float = 0.5,
        quality_threshold: float | None = None,
        k_nearest: int = 15,
        metric: str = "euclidean",
    ):
        """
        Initialize behavioral archive.

        Args:
            max_size: Maximum number of solutions to store
            novelty_weight: Weight for novelty vs quality (α ∈ [0, 1])
                           0 = pure quality, 1 = pure novelty, 0.5 = balanced
            quality_threshold: Minimum fitness quality to enter archive
                              (None = no threshold)
            k_nearest: Number of nearest neighbors for novelty calculation
            metric: Distance metric for behavioral space
        """
        self.max_size = max_size
        self.novelty_weight = novelty_weight
        self.quality_weight = 1.0 - novelty_weight
        self.quality_threshold = quality_threshold
        self.k_nearest = k_nearest
        self.metric = metric

        # Archive storage
        self.entries: list[ArchiveEntry] = []

        # Statistics
        self.total_additions = 0
        self.total_rejections = 0
        self.total_replacements = 0

    def add(
        self,
        individual: Individual,
        behavioral_features: NDArray[np.float64],
        fitness: tuple[float, float],
        generation: int,
    ) -> bool:
        """
        Add solution to archive if sufficiently novel or high quality.

        Addition criteria:
        1. If archive not full: add if above quality threshold
        2. If archive full: add if dominates worst entry in (novelty, quality)

        Args:
            individual: Schedule individual
            behavioral_features: Behavioral feature vector
            fitness: (hard_violations, soft_violations)
            generation: Current GA generation

        Returns:
            True if solution was added, False if rejected
        """
        # Compute novelty
        novelty = self.compute_novelty(behavioral_features)

        # Check quality threshold
        combined_fitness = abs(fitness[0]) * 100 + abs(fitness[1])
        if (
            self.quality_threshold is not None
            and combined_fitness > self.quality_threshold
        ):
            self.total_rejections += 1
            return False

        # Create entry
        entry = ArchiveEntry(
            individual=individual,
            behavioral_features=behavioral_features,
            fitness=fitness,
            novelty=novelty,
            generation_added=generation,
        )

        # If archive not full, add directly
        if len(self.entries) < self.max_size:
            self.entries.append(entry)
            self.total_additions += 1
            return True

        # Archive full: check if new entry dominates any existing entry
        should_add, replace_idx = self._should_replace(entry)

        if should_add:
            if replace_idx is not None:
                self.entries[replace_idx] = entry
                self.total_replacements += 1
            else:
                # Should not happen, but handle gracefully
                self.entries.append(entry)
                self.total_additions += 1
            return True
        else:
            self.total_rejections += 1
            return False

    def compute_novelty(self, behavioral_features: NDArray[np.float64]) -> float:
        """
        Compute novelty score for given behavioral features.

        Novelty = average distance to k-nearest neighbors in archive.

        Args:
            behavioral_features: Behavioral feature vector

        Returns:
            Novelty score (higher = more novel)
        """
        if len(self.entries) == 0:
            return float("inf")  # First solution is infinitely novel

        archive_features = [entry.behavioral_features for entry in self.entries]
        novelty = compute_novelty(
            behavioral_features, archive_features, k=self.k_nearest, metric=self.metric
        )

        return novelty

    def _should_replace(self, new_entry: ArchiveEntry) -> tuple[bool, int | None]:
        """
        Determine if new entry should replace an existing entry.

        Replacement criteria:
        - Compute novelty-quality score for new entry and all archive entries
        - If new entry has higher score than worst entry: replace worst

        Score = α * novelty + (1 - α) * (1 / fitness)

        Args:
            new_entry: Candidate entry

        Returns:
            (should_replace, index_to_replace)
        """
        # Compute novelty-quality scores for all entries
        scores = []
        for entry in self.entries:
            # Recompute novelty for existing entries (may change as archive grows)
            # For efficiency, could cache and update incrementally
            novelty_normalized = entry.novelty / (
                entry.novelty + 1.0
            )  # Normalize to [0, 1]
            quality_normalized = 1.0 / (
                entry.combined_fitness + 1.0
            )  # Higher is better

            score = (
                self.novelty_weight * novelty_normalized
                + self.quality_weight * quality_normalized
            )
            scores.append(score)

        # Score for new entry
        new_novelty_normalized = new_entry.novelty / (new_entry.novelty + 1.0)
        new_quality_normalized = 1.0 / (new_entry.combined_fitness + 1.0)
        new_score = (
            self.novelty_weight * new_novelty_normalized
            + self.quality_weight * new_quality_normalized
        )

        # Find worst existing entry
        worst_idx = int(np.argmin(scores))
        worst_score = scores[worst_idx]

        # Replace if new entry is better than worst
        if new_score > worst_score:
            return True, worst_idx
        else:
            return False, None

    def get_diverse_subset(self, k: int = 10) -> list[Individual]:
        """
        Get k most diverse solutions from archive.

        Uses greedy farthest-point sampling to maximize diversity.

        Args:
            k: Number of solutions to return

        Returns:
            List of k diverse individuals
        """
        if len(self.entries) == 0:
            return []

        k = min(k, len(self.entries))

        # Start with highest novelty solution
        selected_indices: list[int] = [
            int(np.argmax([e.novelty for e in self.entries]))
        ]

        # Greedily add solutions farthest from selected set
        while len(selected_indices) < k:
            max_min_distance: float = -1.0
            best_idx: int = -1

            for i in range(len(self.entries)):
                if i in selected_indices:
                    continue

                # Compute minimum distance to selected set
                min_distance = float("inf")
                for j in selected_indices:
                    dist = np.linalg.norm(
                        self.entries[i].behavioral_features
                        - self.entries[j].behavioral_features
                    )
                    min_distance = float(min(min_distance, dist))  # type: ignore[arg-type]

                # Track solution with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = int(i)

            selected_indices.append(best_idx)

        return [self.entries[i].individual for i in selected_indices]

    def get_best_quality(self, k: int = 10) -> list[Individual]:
        """
        Get k best quality solutions from archive.

        Args:
            k: Number of solutions to return

        Returns:
            List of k best individuals by fitness
        """
        if len(self.entries) == 0:
            return []

        k = min(k, len(self.entries))

        # Sort by fitness (ascending, lower is better)
        sorted_entries = sorted(self.entries, key=lambda e: e.combined_fitness)

        return [entry.individual for entry in sorted_entries[:k]]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get archive statistics.

        Returns:
            Dictionary with archive metrics
        """
        if len(self.entries) == 0:
            return {
                "size": 0,
                "mean_novelty": 0.0,
                "mean_fitness": 0.0,
                "total_additions": self.total_additions,
                "total_rejections": self.total_rejections,
                "total_replacements": self.total_replacements,
            }

        novelties = [e.novelty for e in self.entries]
        fitnesses = [e.combined_fitness for e in self.entries]

        return {
            "size": len(self.entries),
            "mean_novelty": float(np.mean(novelties)),
            "std_novelty": float(np.std(novelties)),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "best_fitness": float(np.min(fitnesses)),
            "worst_fitness": float(np.max(fitnesses)),
            "total_additions": self.total_additions,
            "total_rejections": self.total_rejections,
            "total_replacements": self.total_replacements,
            "acceptance_rate": (
                self.total_additions / (self.total_additions + self.total_rejections)
                if (self.total_additions + self.total_rejections) > 0
                else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear archive and reset statistics."""
        self.entries.clear()
        self.total_additions = 0
        self.total_rejections = 0
        self.total_replacements = 0

    def __len__(self) -> int:
        """Get number of entries in archive."""
        return len(self.entries)

    def __iter__(self):
        """Iterate over archive entries."""
        return iter(self.entries)

    def __getitem__(self, index: int) -> ArchiveEntry:
        """Get entry by index."""
        return self.entries[index]
