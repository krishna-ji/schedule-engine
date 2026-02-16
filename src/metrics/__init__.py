"""Metrics package: Optimization metrics and analysis.

Provides metrics for measuring algorithm performance:
- Hypervolume calculation
- Pareto front analysis
- Population diversity
- Convergence tracking
- Behavioral novelty

Usage:
    from src.metrics import calculate_hypervolume, average_pairwise_diversity
    from src.metrics import compute_novelty, BehavioralArchive
"""

from __future__ import annotations

from src.metrics.behavioral_archive import BehavioralArchive
from src.metrics.behavioral_features import extract_behavioral_features
from src.metrics.convergence import calculate_convergence_rate, detect_stagnation
from src.metrics.diversity import average_pairwise_diversity, individual_distance
from src.metrics.hypervolume import calculate_hypervolume
from src.metrics.novelty import compute_novelty, k_nearest_neighbors
from src.metrics.pareto_metrics import (
    calculate_generational_distance,
    calculate_inverted_generational_distance,
    calculate_spacing,
)

__all__ = [
    # Behavioral diversity
    "BehavioralArchive",
    "average_pairwise_diversity",
    "calculate_convergence_rate",
    "calculate_generational_distance",
    # Core metrics
    "calculate_hypervolume",
    "calculate_inverted_generational_distance",
    "calculate_spacing",
    "compute_novelty",
    "detect_stagnation",
    "extract_behavioral_features",
    "individual_distance",
    "k_nearest_neighbors",
]
