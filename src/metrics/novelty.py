"""
Novelty metrics for behavioral diversity measurement.

ENHANCEMENT #6: Novelty search using behavioral characterization.

Novelty search rewards solutions that are different from previously discovered
solutions, measured in behavior space (not fitness space).

Mathematical Definition:
    novelty(x) = (1/k) Σ_{i=1}^k dist(x, μ_i)

Where:
    - x: Current solution's behavior vector
    - μ_i: i-th nearest neighbor in behavior space
    - k: Number of nearest neighbors (typically 10-15)
    - dist: Distance metric (Euclidean, Manhattan, Cosine)

This encourages exploration of diverse behavioral strategies, leading to
quality-diversity optimization beyond pure fitness maximization.

References:
- Lehman & Stanley (2011): "Abandoning Objectives: Evolution through the
  Search for Novelty Alone"
- Mouret & Clune (2015): "Illuminating the Space of Possible Solutions"
"""

import numpy as np
from numpy.typing import NDArray


def compute_novelty(
    features: NDArray[np.float64],
    archive_features: list[NDArray[np.float64]],
    k: int = 15,
    metric: str = "euclidean",
) -> float:
    """
    Compute novelty score for a solution.

    Novelty is the average distance to k-nearest neighbors in behavior space.
    Higher novelty = more different from archive = more valuable for diversity.

    Args:
        features: Behavioral feature vector of current solution
        archive_features: List of feature vectors from archive
        k: Number of nearest neighbors for novelty calculation
        metric: Distance metric ("euclidean", "manhattan", "cosine")

    Returns:
        Novelty score (higher = more novel)

    Example:
        >>> current_features = extract_behavioral_features(individual, context)
        >>> archive_features = [extract_behavioral_features(a, context) for a in archive]
        >>> novelty = compute_novelty(current_features, archive_features, k=15)
        >>> print(f"Novelty score: {novelty}")
    """
    if len(archive_features) == 0:
        # No archive yet: infinite novelty (first solution is always novel)
        return float("inf")

    # Compute distances to all archive members
    distances = []
    for archive_feature in archive_features:
        dist = _compute_distance(features, archive_feature, metric)
        distances.append(dist)

    # Get k nearest neighbors
    k_actual = min(k, len(distances))
    k_nearest = sorted(distances)[:k_actual]

    # Novelty is average distance to k-nearest neighbors
    novelty_score = np.mean(k_nearest)

    return float(novelty_score)


def k_nearest_neighbors(
    features: NDArray[np.float64],
    archive_features: list[NDArray[np.float64]],
    k: int = 15,
    metric: str = "euclidean",
) -> tuple[list[int], list[float]]:
    """
    Find k-nearest neighbors in behavior space.

    Args:
        features: Behavioral feature vector of query solution
        archive_features: List of feature vectors from archive
        k: Number of nearest neighbors to return
        metric: Distance metric

    Returns:
        (indices, distances) of k-nearest neighbors

    Example:
        >>> indices, distances = k_nearest_neighbors(features, archive_features, k=5)
        >>> print(f"5 nearest neighbors: {indices}")
        >>> print(f"Distances: {distances}")
    """
    if len(archive_features) == 0:
        return [], []

    # Compute all distances
    distances = [
        _compute_distance(features, archive_feature, metric)
        for archive_feature in archive_features
    ]

    # Get k nearest
    k_actual = min(k, len(distances))
    sorted_indices = np.argsort(distances)[:k_actual]
    sorted_distances = [distances[i] for i in sorted_indices]

    return sorted_indices.tolist(), sorted_distances


def compute_sparseness(
    features: NDArray[np.float64],
    population_features: list[NDArray[np.float64]],
    k: int = 15,
    metric: str = "euclidean",
) -> float:
    """
    Compute sparseness metric for a solution within current population.

    Sparseness measures how isolated a solution is in behavior space.
    Similar to novelty but computed within current population, not archive.

    Args:
        features: Behavioral feature vector of current solution
        population_features: Feature vectors of current population
        k: Number of nearest neighbors
        metric: Distance metric

    Returns:
        Sparseness score (higher = more sparse/isolated)
    """
    return compute_novelty(features, population_features, k, metric)


def compute_local_competition(
    features: NDArray[np.float64],
    population_features: list[NDArray[np.float64]],
    population_fitness: list[float],
    k: int = 15,
    metric: str = "euclidean",
) -> float:
    """
    Compute local competition score.

    Measures how competitive a solution is compared to behaviorally similar
    solutions (k-nearest neighbors). Higher score = dominates local niche.

    Args:
        features: Behavioral feature vector of current solution
        population_features: Feature vectors of population
        population_fitness: Fitness values of population (lower is better)
        k: Number of nearest neighbors for local competition
        metric: Distance metric

    Returns:
        Local competition score (higher = better than local neighbors)
    """
    if len(population_features) == 0:
        return 0.0

    # Find k-nearest neighbors
    indices, distances = k_nearest_neighbors(features, population_features, k, metric)

    if len(indices) == 0:
        return 0.0

    # Get fitness of neighbors
    neighbor_fitness = [population_fitness[i] for i in indices]

    # Compute own fitness (assume last element in population_fitness)
    own_fitness = population_fitness[-1]

    # Competition score: how much better than neighbors (for minimization)
    # Positive score = better than neighbors
    better_count = sum(1 for nf in neighbor_fitness if own_fitness < nf)
    competition_score = better_count / len(neighbor_fitness)

    return float(competition_score)


def _compute_distance(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
    metric: str = "euclidean",
) -> float:
    """
    Compute distance between two feature vectors.

    Args:
        features1: First feature vector
        features2: Second feature vector
        metric: Distance metric

    Returns:
        Distance value
    """
    if metric == "euclidean":
        return float(np.linalg.norm(features1 - features2))
    elif metric == "manhattan":
        return float(np.sum(np.abs(features1 - features2)))
    elif metric == "cosine":
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        return float(1.0 - (dot_product / (norm1 * norm2)))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_coverage(
    archive_features: list[NDArray[np.float64]],
    n_bins: int = 10,
) -> float:
    """
    Compute behavioral space coverage.

    Measures how well the archive covers the behavioral space by dividing
    space into bins and counting occupied bins.

    Args:
        archive_features: List of feature vectors from archive
        n_bins: Number of bins per dimension for discretization

    Returns:
        Coverage ratio in [0, 1] (1 = perfect coverage)
    """
    if len(archive_features) == 0:
        return 0.0

    # Convert to numpy array
    features_array = np.array(archive_features)
    n_features = features_array.shape[1]

    # Compute min/max for each dimension
    min_vals = features_array.min(axis=0)
    max_vals = features_array.max(axis=0)

    # Discretize features into bins
    occupied_bins = set()

    for features in archive_features:
        # Compute bin index for each dimension
        bin_indices = []
        for i in range(n_features):
            if max_vals[i] == min_vals[i]:
                bin_idx = 0
            else:
                normalized = (features[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
                bin_idx = min(int(normalized * n_bins), n_bins - 1)
            bin_indices.append(bin_idx)

        # Add to occupied bins
        occupied_bins.add(tuple(bin_indices))

    # Compute coverage
    total_bins = n_bins**n_features
    coverage = len(occupied_bins) / total_bins

    return float(coverage)


def compute_diversity_metrics(
    archive_features: list[NDArray[np.float64]],
) -> dict:
    """
    Compute comprehensive diversity metrics for archive.

    Returns:
        Dictionary with diversity metrics:
        - mean_pairwise_distance: Average distance between all pairs
        - std_pairwise_distance: Std of pairwise distances
        - coverage: Behavioral space coverage
        - archive_size: Number of solutions in archive
    """
    if len(archive_features) == 0:
        return {
            "mean_pairwise_distance": 0.0,
            "std_pairwise_distance": 0.0,
            "coverage": 0.0,
            "archive_size": 0,
        }

    # Compute all pairwise distances
    distances = []
    for i in range(len(archive_features)):
        for j in range(i + 1, len(archive_features)):
            dist = _compute_distance(archive_features[i], archive_features[j])
            distances.append(dist)

    mean_dist = np.mean(distances) if distances else 0.0
    std_dist = np.std(distances) if distances else 0.0

    # Compute coverage
    coverage = compute_coverage(archive_features, n_bins=10)

    return {
        "mean_pairwise_distance": float(mean_dist),
        "std_pairwise_distance": float(std_dist),
        "coverage": coverage,
        "archive_size": len(archive_features),
    }
