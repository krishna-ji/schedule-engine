"""
Repair Heuristics - Constraint Violation Fixing Operators

Provides repair operators that fix constraint violations to restore feasibility.
These are academically correct "repair heuristics" for constraint satisfaction.

Repair heuristics are useful for:
1. Restoring feasibility after crossover/mutation
2. Memetic algorithms (intensive local improvement)
3. Handling constraint violations during search

Strategies:
1. IGLS Repair: Guided local search for violation fixing
2. Greedy Repair: First-improving repair moves
3. Exhaustive Repair: Steepest descent on all genes
4. LNS Repair: Destroy-and-repair large neighborhoods
5. Memetic Repair: Intensive repair on elite individuals

Architecture:
- Decorator-based registration with @repair_heuristic
- Wraps existing repair operators from src/ga/operators/
- Returns number of violations fixed
- Can be expensive (controlled by max_iterations)

Usage:
    from src.heuristics.repair import igls_repair

    # Apply IGLS repair
    fixes = igls_repair(individual, context, max_iterations=2)
    print(f"Fixed {fixes} violations")
"""

from typing import List
import time

from src.ga.sessiongene import SessionGene
from src.core.types import SchedulingContext
from src.heuristics.registry import repair_heuristic


# ================
# IGLS REPAIR (Guided local search for violation fixing)
# ================


@repair_heuristic(
    name="igls_repair",
    description="Iterative Guided Local Search for constraint violation fixing",
    priority=1,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def igls_repair(
    individual: List[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 2,
) -> int:
    """
    Apply IGLS repair to fix constraint violations.

    IGLS (Iterative Guided Local Search) uses intelligent neighborhood
    selection to systematically fix violations.

    Args:
        individual: List of SessionGene to repair
        context: Scheduling context
        max_iterations: Maximum repair iterations

    Returns:
        Number of violations fixed
    """
    from src.ga.operators.repair import repair_individual_unified

    stats = repair_individual_unified(
        individual,
        context,
        max_iterations=max_iterations,
        selective=True,
    )

    return stats.get("total_fixes", 0)


# ================
# GREEDY REPAIR (First-improving repair moves)
# ================


@repair_heuristic(
    name="greedy_repair",
    description="Fast greedy repair with first-improving moves",
    priority=2,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def greedy_repair(
    individual: List[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 5,
) -> int:
    """
    Apply greedy repair (hill climbing) to fix violations.

    Greedy repair makes first-improving moves without exhaustive search.
    Faster than IGLS but may find local optima.

    Args:
        individual: List of SessionGene to repair
        context: Scheduling context
        max_iterations: Maximum repair iterations

    Returns:
        Number of violations fixed
    """
    from src.ga.operators.intensive_local_search import apply_greedy_search

    _, metrics = apply_greedy_search(
        population=[individual],
        context=context,
        population_coverage=1.0,  # Apply to all
        timeout_seconds=10,
    )

    return metrics.get("total_improvements", 0)


# ================
# EXHAUSTIVE REPAIR (Steepest descent on all genes)
# ================


@repair_heuristic(
    name="exhaustive_repair",
    description="Exhaustive steepest descent repair (very intensive)",
    priority=3,
    enabled_by_default=False,  # Expensive - disabled by default
    requires_population=False,
    modifies_individual=True,
)
def exhaustive_repair(
    individual: List[SessionGene],
    context: SchedulingContext,
    max_neighborhood_size: int = 10,
) -> int:
    """
    Apply exhaustive repair (steepest descent).

    Exhaustive repair tries ALL possible moves and picks the best.
    Very expensive but finds optimal local improvements.

    WARNING: Can take 10+ seconds per individual!

    Args:
        individual: List of SessionGene to repair
        context: Scheduling context
        max_neighborhood_size: Maximum moves to try per gene

    Returns:
        Number of violations fixed
    """
    from src.ga.operators.intensive_local_search import apply_exhaustive_search

    _, metrics = apply_exhaustive_search(
        population=[individual],
        context=context,
        population_coverage=1.0,
        max_neighborhood_size=max_neighborhood_size,
        timeout_seconds=30,
    )

    return metrics.get("total_improvements", 0)


# ================
# LNS REPAIR (Large Neighborhood Search - destroy and rebuild)
# ================


@repair_heuristic(
    name="lns_repair",
    description="Large Neighborhood Search (destroy-rebuild strategy)",
    priority=4,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def lns_repair(
    individual: List[SessionGene],
    context: SchedulingContext,
    subproblem_size: int = 10,
) -> int:
    """
    Apply LNS repair (destroy and rebuild).

    LNS (Large Neighborhood Search) destroys part of the schedule
    and rebuilds it with IGLS, allowing larger structural changes.

    Args:
        individual: List of SessionGene to repair
        context: Scheduling context
        subproblem_size: Number of sessions to destroy/rebuild

    Returns:
        Number of violations fixed
    """
    from src.lns.lns_operator import apply_lns_repair

    stats = apply_lns_repair(
        individual=individual,
        context=context,
        subproblem_size=subproblem_size,
        igls_max_iterations=100,
        igls_time_limit=5.0,
    )

    conflicts_repaired = stats.get("total_conflicts_repaired", 0)
    return conflicts_repaired


# ================
# MEMETIC REPAIR (Intensive repair on best individuals)
# ================


@repair_heuristic(
    name="memetic_repair",
    description="Memetic repair (intensive IGLS on elite individuals)",
    priority=5,
    enabled_by_default=False,  # Controlled by memetic_mode config
    requires_population=True,  # Needs population to select elite
    modifies_individual=True,
)
def memetic_repair(
    individual: List[SessionGene],
    population: List[List[SessionGene]],
    context: SchedulingContext,
    elite_percentage: float = 0.05,
    memetic_iterations: int = 5,
) -> int:
    """
    Apply memetic repair (intensive IGLS on elite).

    Memetic repair applies intensive local search to the best
    individuals in the population. More thorough than basic IGLS.

    This is typically applied EVERY generation in memetic algorithms.

    Args:
        individual: List of SessionGene to repair (ignored - uses population)
        population: Full population for elite selection
        context: Scheduling context
        elite_percentage: Percentage of population to repair
        memetic_iterations: IGLS iterations per individual

    Returns:
        Number of violations fixed across all elite
    """
    from src.ga.operators.repair import repair_individual_unified
    from deap import tools

    # Convert population to DEAP individuals for selection
    # (This is a simplified version - real implementation uses proper DEAP types)
    elite_count = max(1, int(elite_percentage * len(population)))

    # Repair elite individuals
    total_fixes = 0
    for ind in population[:elite_count]:  # Assume sorted by fitness
        stats = repair_individual_unified(
            ind,
            context,
            max_iterations=memetic_iterations,
            selective=True,
        )
        total_fixes += stats.get("total_fixes", 0)

    return total_fixes
