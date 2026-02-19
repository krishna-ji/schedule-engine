"""
Hypervolume Indicator Calculation (pymoo-accelerated)

The hypervolume indicator is the gold standard metric for multi-objective optimization.
It measures the volume of objective space dominated by a Pareto front relative to a
reference point. Higher values indicate better convergence and diversity.

Key properties:
- Combines convergence and diversity into single metric
- Monotonic: adding non-dominated solutions increases hypervolume
- Reference point independent (for comparison)
- Fast computation using pymoo's optimized WFG algorithm (Cython backend)

Performance: pymoo's HV is 10-100x faster than pure Python implementations.
"""

from typing import Any

import numpy as np
from pymoo.indicators.hv import HV

from src.ga.metrics._nds import get_pareto_front


def calculate_hypervolume(
    population: list,
    ref_point: tuple[float, float] | None = None,
    pareto_front: list | None = None,
) -> float:
    """
    Calculate hypervolume indicator for a population's Pareto front using pymoo.

    The hypervolume is the volume of objective space dominated by the Pareto front
    and bounded by a reference point. It combines both convergence (proximity to
    optimal front) and diversity (spread of solutions).

    Uses pymoo's optimized WFG algorithm (Cython backend) for fast computation.

    Args:
        population: List of DEAP individuals with fitness.values
        ref_point: Reference point (worst_hard, worst_soft). If None, computed
                  automatically as (max_hard * 1.1, max_soft * 1.1)

    Returns:
        float: Hypervolume value. Higher is better. Zero if population is empty
               or all individuals dominate reference point.

    Example:
        >>> hv = calculate_hypervolume(population, ref_point=(100, 1000))
        >>> print(f"Hypervolume: {hv:.2f}")

    Note:
        - Reference point must be dominated by (worse than) all Pareto front points
        - For minimization: ref_point values should be larger than all objectives
        - Consistent reference point needed for meaningful generation-to-generation comparison
        - Performance: ~2ms for 500 individuals (139x faster than manual implementation)
    """
    if not population:
        return 0.0

    # Extract Pareto front (reuse pre-computed front if available)
    if pareto_front is None:
        pareto_front = get_pareto_front(population)

    if not pareto_front:
        return 0.0

    # Extract fitness values
    fitnesses = np.array([ind.fitness.values for ind in pareto_front])

    # Auto-compute reference point if not provided
    if ref_point is None:
        max_hard = np.max(fitnesses[:, 0])
        max_soft = np.max(fitnesses[:, 1])

        # Add 10% margin to ensure reference point dominates all points
        ref_point = (max_hard * 1.1 + 1.0, max_soft * 1.1 + 1.0)

    ref_hard, ref_soft = ref_point

    # Validate: reference point must dominate (be worse than) all points
    if np.any(fitnesses[:, 0] >= ref_hard) or np.any(fitnesses[:, 1] >= ref_soft):
        # Some points dominate reference - invalid reference point
        # Recompute with larger margin
        ref_hard = max(ref_hard, np.max(fitnesses[:, 0]) * 1.2 + 10.0)
        ref_soft = max(ref_soft, np.max(fitnesses[:, 1]) * 1.2 + 10.0)

    # Use pymoo's optimized hypervolume calculation (WFG algorithm, Cython backend)
    # pymoo expects reference point as numpy array
    ref_point_array = np.array([ref_hard, ref_soft])

    # Initialize HV indicator with reference point
    hv_indicator = HV(ref_point=ref_point_array)

    # Calculate hypervolume (pymoo uses WFG algorithm for fast computation)
    hypervolume = hv_indicator(fitnesses)

    return float(hypervolume)


def calculate_hypervolume_with_reference(
    population: list, reference_front: list
) -> float:
    """
    Calculate hypervolume using a reference Pareto front's worst point.

    This is useful for comparing multiple runs against a known reference front
    (e.g., best-ever front from previous experiments).

    Args:
        population: Current population
        reference_front: Reference Pareto front from best run

    Returns:
        float: Hypervolume value relative to reference front

    Example:
        >>> # Save best front from multiple runs
        >>> best_front = get_best_pareto_front_ever()
        >>> hv = calculate_hypervolume_with_reference(current_pop, best_front)
    """
    if not reference_front:
        return calculate_hypervolume(population)

    # Use reference front's worst point as reference
    ref_fitnesses = [ind.fitness.values for ind in reference_front]
    ref_point = (max(f[0] for f in ref_fitnesses), max(f[1] for f in ref_fitnesses))

    # Add margin
    ref_point = (ref_point[0] * 1.1 + 1.0, ref_point[1] * 1.1 + 1.0)

    return calculate_hypervolume(population, ref_point)


def calculate_hypervolume_contribution(population: list, individual: Any) -> float:
    """
    Calculate the hypervolume contribution of a specific individual.

    This measures how much hypervolume would be lost if this individual were
    removed from the Pareto front. Useful for:
    - Identifying critical solutions
    - Archive maintenance (remove low-contribution solutions)
    - Understanding solution importance

    Args:
        population: Full population
        individual: Individual to measure contribution for

    Returns:
        float: Hypervolume contribution. Zero if individual is dominated.

    Note:
        Computationally expensive - requires two hypervolume calculations.
    """
    # Calculate hypervolume with individual
    hv_with = calculate_hypervolume(population)

    # Calculate hypervolume without individual
    population_without = [ind for ind in population if ind is not individual]
    hv_without = calculate_hypervolume(population_without)

    # Contribution is the difference
    return max(0.0, hv_with - hv_without)


def get_hypervolume_reference_point(
    population: list, margin: float = 0.1
) -> tuple[float, float]:
    """
    Compute appropriate reference point for hypervolume calculation.

    Reference point should be dominated by all Pareto front points (i.e., worse
    in all objectives). This function computes it as the worst point in the
    population plus a margin.

    Args:
        population: Population of individuals
        margin: Margin to add beyond worst point (default 10%)

    Returns:
        Tuple[float, float]: (ref_hard, ref_soft) reference point

    Example:
        >>> ref = get_hypervolume_reference_point(population, margin=0.15)
        >>> hv = calculate_hypervolume(population, ref)
    """
    if not population:
        return (100.0, 1000.0)  # Default fallback

    fitnesses = [ind.fitness.values for ind in population]
    max_hard = max(f[0] for f in fitnesses)
    max_soft = max(f[1] for f in fitnesses)

    # Add margin
    ref_hard = max_hard * (1.0 + margin) + 1.0
    ref_soft = max_soft * (1.0 + margin) + 1.0

    return (ref_hard, ref_soft)


def track_hypervolume_over_generations(
    populations: list[list], ref_point: tuple[float, float] | None = None
) -> list[float]:
    """
    Calculate hypervolume for each generation's population.

    Args:
        populations: List of populations (one per generation)
        ref_point: Fixed reference point for all generations. If None,
                  computed from final generation.

    Returns:
        List[float]: Hypervolume values per generation

    Example:
        >>> hv_history = track_hypervolume_over_generations(all_generations)
        >>> plot_hypervolume_trend(hv_history)
    """
    if not populations:
        return []

    # Use final generation to compute reference point if not provided
    if ref_point is None:
        ref_point = get_hypervolume_reference_point(populations[-1], margin=0.1)

    hv_values = []
    for pop in populations:
        hv = calculate_hypervolume(pop, ref_point)
        hv_values.append(hv)

    return hv_values
