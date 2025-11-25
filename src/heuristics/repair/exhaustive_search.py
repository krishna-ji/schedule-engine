"""
Exhaustive Search Heuristic

Intensive local search that evaluates ALL neighbors for each gene.
This is the most thorough (and slowest) repair operator.

Previously: Hardcoded generation trigger (gens 3, 25)
Now: Proper heuristic in registry, controlled by priority/enabled flag
"""

from typing import List

from src.ga.sessiongene import SessionGene
from src.core.types import SchedulingContext
from src.heuristics.registry import repair_heuristic


@repair_heuristic(
    name="exhaustive_search",
    description="Intensive exhaustive local search on top individuals (steepest descent)",
    priority=10,  # High priority - apply after lighter repairs
    enabled_by_default=True,
    requires_population=True,  # Needs full population to select top N%
    modifies_individual=True,
)
def exhaustive_search(
    individual: List[SessionGene],
    population: List[List[SessionGene]],
    context: SchedulingContext,
    population_coverage: float = 0.15,  # Apply to top 15% only
    max_neighborhood_size: int = 100,
    timeout_seconds: int = 180,
) -> int:
    """
    Apply exhaustive local search to individual.

    This operator evaluates ALL possible neighbors for each gene and selects
    the best improvement (steepest descent).

    Args:
        individual: Individual to optimize
        population: Full population (to check if this individual is in top N%)
        context: Scheduling context
        population_coverage: Only apply if individual is in top N% (default 15%)
        max_neighborhood_size: Max neighbors per gene
        timeout_seconds: Abort if operation takes too long

    Returns:
        Number of genes improved
    """
    from src.ga.operators.intensive_local_search import (
        exhaustive_search_individual,
    )

    # Check if this individual is worth intensive optimization
    # (only apply to top individuals to save time)
    from deap import tools
    
    num_elite = max(1, int(len(population) * population_coverage))
    elite = tools.selBest(population, num_elite)
    
    # Skip if individual is not in elite set
    if individual not in elite:
        return 0
    
    # Apply exhaustive search
    improved_individual, metrics = exhaustive_search_individual(
        individual=individual,
        context=context,
        max_neighborhood_size=max_neighborhood_size,
        timeout_seconds=timeout_seconds,
    )
    
    # Update individual in-place
    if improved_individual is not individual:
        individual[:] = improved_individual
    
    # Invalidate fitness
    if hasattr(individual, 'fitness'):
        del individual.fitness.values
    
    return metrics.get('genes_improved', 0)
