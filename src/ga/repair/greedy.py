"""
Greedy Repair Heuristic

First-improving repair operator integrated as a heuristic. Provides a
fast hill-climbing repair that quickly addresses obvious constraint
violations without the overhead of exhaustive search.
"""

from src.domain.gene import SessionGene
from src.domain.types import SchedulingContext


def greedy_repair(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 5,
) -> int:
    """Apply greedy repair (hill climbing) to fix violations."""
    from src.ga.operators.intensive_local_search import apply_greedy_search

    _, metrics = apply_greedy_search(
        population=[individual],
        context=context,
        population_coverage=1.0,
        timeout_seconds=10,
        max_iterations=max_iterations,
    )

    return int(metrics.get("total_improvements", 0))
