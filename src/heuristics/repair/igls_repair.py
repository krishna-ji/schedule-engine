"""
IGLS Repair Heuristic

Iterative Greedy Local Search repair operator integrated as a heuristic.
This operator fixes hard constraint violations using stagnation-triggered repair.
"""

from typing import List, Optional
import copy

from src.ga.sessiongene import SessionGene
from src.core.types import SchedulingContext
from src.heuristics.registry import HeuristicCategory

# Import the original IGLS repair logic
from src.ga.operators.repair import repair_individual_unified


from src.heuristics.registry import repair_heuristic


@repair_heuristic(
    name="igls_repair",
    description="Iterative Greedy Local Search repair for constraint violations",
    priority=1,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def igls_repair(
    individual: List[SessionGene],
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
        selective=selective
    )
    
    return stats.get('total_fixes', 0)