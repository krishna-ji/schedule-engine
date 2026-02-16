"""
Selective Repair Heuristic

Selective repair operator integrated as a heuristic.
Targets only genes with detected violations for 3-4x speedup.

Performance:
- Scans ~5-15% of genes instead of 100%
- Uses violation detection (fast/full/hybrid strategies)
- Falls back to full scan if detection fails

Constraint Coverage: Same as base repairs (6 of 8 hard, 1 of 4 soft)

Application: Default repair mode in Modes B-E (recommended for speed)
"""

from src.domain.gene import SessionGene
from src.domain.types import SchedulingContext

# Import the original selective repair logic
from src.ga.repair.selective import repair_individual_selective


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
