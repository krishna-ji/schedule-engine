"""
Exhaustive Repair Heuristic

Performs steepest-descent style exhaustive neighbor evaluation for each
gene in an individual. Extremely expensive but finds high-quality local
optima when other repair passes stall.
"""

from src.domain.gene import SessionGene
from src.domain.types import SchedulingContext


def exhaustive_repair(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_neighborhood_size: int = 25,
    timeout_seconds: float = 10.0,
) -> int:
    """Apply exhaustive local improvements to a single individual."""
    from src.ga.operators.intensive_local_search import apply_exhaustive_search

    _, metrics = apply_exhaustive_search(
        population=[individual],
        context=context,
        population_coverage=1.0,
        max_neighborhood_size=max_neighborhood_size,
        timeout_seconds=int(timeout_seconds),
    )

    return int(metrics.get("total_improvements", 0))
