"""GA Metrics package: Performance metrics for genetic algorithm optimization.

This package provides metrics for measuring GA performance:
- Hypervolume calculation (Pareto front quality)
- Spacing and diversity metrics
- Convergence tracking
- Violation heatmaps

Usage:
    from src.ga.metrics import (
        calculate_hypervolume,
        average_pairwise_diversity,
        calculate_spacing,
        ViolationHeatmap,
    )
"""

from __future__ import annotations

from src.ga.metrics.convergence import (
    calculate_convergence_rate,
    detect_stagnation,
)
from src.ga.metrics.diversity import (
    average_pairwise_diversity,
    individual_distance,
)
from src.ga.metrics.hypervolume import calculate_hypervolume
from src.ga.metrics.pareto_metrics import (
    calculate_generational_distance,
    calculate_inverted_generational_distance,
    calculate_spacing,
)
from src.ga.metrics.violation_heatmap import ViolationHeatmap
from src.ga.metrics.violation_recorder import record_violations_to_heatmap

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
    # Violation tracking
    "ViolationHeatmap",
    "record_violations_to_heatmap",
]
