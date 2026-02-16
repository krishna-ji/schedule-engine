"""
IGLS Repair Heuristic

Iterative Greedy Local Search repair operator integrated as a heuristic.
Fixes 6 of 8 hard constraint violations using unified repair system.

Constraint Coverage:
   HC1, HC2, HC3, HC4, HC5, HC8 (via 7 base repair operators)
   HC6 (room always available), HC7 (structural integrity)

Application: Stagnation-triggered repair in Modes C-E (adaptive strategy)
"""

from src.domain.gene import SessionGene
from src.domain.types import SchedulingContext

# Import the original IGLS repair logic
from src.ga.repair.basic import repair_individual_unified


def igls_repair(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 2,
    selective: bool = True,
) -> int:
    """
    Apply IGLS repair to fix constraint violations.

    Args:
        individual: Individual to repair
        context: Scheduling context
        max_iterations: Maximum repair iterations
        selective: Use selective repair (faster)

    Returns:
        Number of violations fixed
    """
    # Use the existing unified repair function
    stats = repair_individual_unified(
        individual=individual,
        context=context,
        max_iterations=max_iterations,
        selective=selective,
    )

    return int(stats.get("total_fixes", 0))
