"""
Selective Repair Heuristic

Selective repair operator integrated as a heuristic.
This operator targets only genes known to have violations for efficiency.
"""

from src.core.types import SchedulingContext

# Import the original selective repair logic
from src.ga.operators.repair_selective import repair_individual_selective
from src.ga.sessiongene import SessionGene
from src.heuristics.registry import repair_heuristic


@repair_heuristic(
    name="selective_repair",
    description="Selective repair targeting only violated genes for efficiency",
    priority=3,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def selective_repair(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 2,
) -> int:
    """
    Apply selective repair to fix constraint violations.

    Args:
        individual: Individual to repair
        context: Scheduling context
        max_iterations: Maximum repair iterations

    Returns:
        Number of violations fixed
    """
    # Use the existing selective repair function
    stats = repair_individual_selective(
        individual=individual, context=context, max_iterations=max_iterations
    )

    # Handle both dict and None return types
    if stats is None:
        return 0
    if isinstance(stats, dict):
        return int(stats.get("total_fixes", 0))
    # If it returns an integer directly
    if isinstance(stats, int):
        return stats
    return 0
