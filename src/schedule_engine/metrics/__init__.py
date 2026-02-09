"""Metrics package: Optimization metrics and analysis.

Provides metrics for measuring algorithm performance:
- Hypervolume calculation
- Pareto front analysis
- Population diversity
- Convergence tracking

Usage:
    from schedule_engine.metrics import calculate_hypervolume, average_pairwise_diversity
"""

from __future__ import annotations

from schedule_engine.metrics.convergence import (
    calculate_convergence_rate,
    detect_stagnation,
)
from schedule_engine.metrics.diversity import (
    average_pairwise_diversity,
    individual_distance,
)
from schedule_engine.metrics.hypervolume import calculate_hypervolume
from schedule_engine.metrics.novelty import compute_novelty, k_nearest_neighbors
from schedule_engine.metrics.pareto_metrics import (
    calculate_generational_distance,
    calculate_inverted_generational_distance,
    calculate_spacing,
)

__all__ = [
    # Core metrics
    "calculate_hypervolume",
    "calculate_generational_distance",
    "calculate_inverted_generational_distance",
    "calculate_spacing",
    "average_pairwise_diversity",
    "individual_distance",
    "calculate_convergence_rate",
    "detect_stagnation",
    "compute_novelty",
    "k_nearest_neighbors",
]
