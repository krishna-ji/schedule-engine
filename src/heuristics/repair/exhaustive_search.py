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
    description="Intensive exhaustive local search on individual (steepest descent)",
    priority=10,  # High priority - apply after lighter repairs
    enabled_by_default=False,  # Expensive - disable by default, enable in specific modes
    requires_population=False,
    modifies_individual=True,
)
def exhaustive_search(
    individual: List[SessionGene],
    context: SchedulingContext,
    max_neighborhood_size: int = 50,  # Reduced for individual-level application
    timeout_seconds: float = 5.0,
) -> int:
    """
    Apply exhaustive local search to individual.

    This operator evaluates ALL possible neighbors for each gene and selects
    the best improvement (steepest descent). Applied gene-by-gene.

    Args:
        individual: Individual to optimize
        context: Scheduling context
        max_neighborhood_size: Max neighbors per gene (default 50 for speed)
        timeout_seconds: Abort if operation takes too long

    Returns:
        Number of genes improved
    """
    from src.ga.operators.local_search import optimize_gene_exhaustive
    import time

    start_time = time.time()
    genes_improved = 0
    total_improvement = 0

    # Optimize each gene exhaustively
    for gene_idx, gene in enumerate(individual):
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            break

        # Apply exhaustive search on this gene
        improved_gene, improvement = optimize_gene_exhaustive(
            gene=gene,
            individual=individual,
            gene_index=gene_idx,
            context=context,
            max_neighborhood_size=max_neighborhood_size,
        )

        # Update if improved
        if improvement > 0:
            individual[gene_idx] = improved_gene
            genes_improved += 1
            total_improvement += improvement

    # Invalidate fitness if any improvements
    if genes_improved > 0 and hasattr(individual, 'fitness'):
        del individual.fitness.values

    return genes_improved
