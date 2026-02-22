"""
Intensive Global Local Search (IGLS) System

Orchestrates intensive local search operations on populations for deep optimization.
Works at population level, applying local search to multiple individuals with
performance controls (timeouts, population sampling, parallelization).

PARALLELIZATION: Gene-level parallel optimization using ProcessPoolExecutor.
Expected speedup: 4-8x on multi-core systems.

Strategies:
- Exhaustive: Steepest descent on all genes (gen 3, 25)
- Greedy: Hill climbing on all genes (stagnation trigger)

Performance Controls:
- Population sampling (optimize top N% only)
- Timeout protection (abort if too slow)
- Parallelization support (gene-level or individual-level)
- Early stopping (if no improvement)

Usage:
    from src.ga.operators.intensive_local_search import (
        apply_exhaustive_search,
        apply_greedy_search
    )

    # Exhaustive at gen 3
    improved_pop, metrics = apply_exhaustive_search(
        population=population,
        context=context,
        config=config.exhaustive_search
    )

    # Greedy on stagnation
    improved_pop, metrics = apply_greedy_search(
        population=population,
        context=context,
        config=config.stagnation_repair
    )
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.domain.gene import SessionGene
from src.domain.types import Individual, SchedulingContext
from src.ga.operators.local_search import optimize_gene_exhaustive, optimize_gene_greedy
from src.utils.system_info import get_cpu_count

type ExhaustiveArgs = tuple[
    SessionGene,
    list[SessionGene],
    int,
    SchedulingContext,
    int,
]
type GreedyArgs = tuple[
    SessionGene,
    list[SessionGene],
    int,
    SchedulingContext,
    int,
]
type GeneOptimizationResult = tuple[int, SessionGene, int]
type MetricsDict = dict[str, float | int | bool]


def _optimize_gene_wrapper_exhaustive(args: ExhaustiveArgs) -> GeneOptimizationResult:
    """
    Wrapper for parallel gene optimization (exhaustive).

    Args:
        args: Tuple of (gene, individual, gene_index, context, max_neighborhood_size)

    Returns:
        Tuple of (gene_index, improved_gene, improvement)
    """
    gene, individual, gene_index, context, max_neighborhood_size = args
    improved_gene, improvement = optimize_gene_exhaustive(
        gene=gene,
        individual=individual,
        gene_index=gene_index,
        context=context,
        max_neighborhood_size=max_neighborhood_size,
    )
    return (gene_index, improved_gene, improvement)


def _optimize_gene_wrapper_greedy(args: GreedyArgs) -> GeneOptimizationResult:
    """
    Wrapper for parallel gene optimization (greedy).

    Args:
        args: Tuple of (gene, individual, gene_index, context, max_iterations)

    Returns:
        Tuple of (gene_index, improved_gene, improvement)
    """
    gene, individual, gene_index, context, max_iterations = args
    improved_gene, improvement = optimize_gene_greedy(
        gene=gene,
        individual=individual,
        gene_index=gene_index,
        context=context,
        max_iterations=max_iterations,
    )
    return (gene_index, improved_gene, improvement)


def apply_exhaustive_search(
    population: list[Individual],
    context: SchedulingContext,
    population_coverage: float = 0.3,
    max_neighborhood_size: int = 100,
    timeout_seconds: int = 180,
    parallel: bool = True,  # NEW: Enable/disable parallelization
) -> tuple[list[Individual], MetricsDict]:
    """
    Apply exhaustive local search to population.

    PARALLELIZED: Optimizes all genes of an individual in parallel using ProcessPoolExecutor.
    Expected speedup: 4-8x on multi-core systems.

    Used for fixed-generation intensive optimization (e.g., gen 3, 25).
    Performs steepest descent on ALL genes in selected individuals.

    Args:
        population: List of individuals (schedules)
        context: Scheduling context
        population_coverage: Fraction of population to optimize (0.3 = top 30%)
        max_neighborhood_size: Max neighbors per gene
        timeout_seconds: Abort if exceeds this time
        parallel: Use parallel gene optimization (default True)

    Returns:
        Tuple of (improved_population, metrics_dict)

    Metrics:
        - individuals_processed: Number of individuals optimized
        - genes_improved: Total genes improved across population
        - total_improvement: Sum of violation reductions
        - execution_time: Time taken in seconds
        - timed_out: Whether operation was aborted
    """
    start_time = time.time()

    metrics: MetricsDict = {
        "individuals_processed": 0,
        "genes_improved": 0,
        "total_improvement": 0,
        "genes_evaluated": 0,
        "execution_time": 0.0,
        "timed_out": False,
    }

    # Sort population by fitness (best first)
    sorted_pop = sorted(
        population,
        key=lambda ind: (
            ind.fitness.values
            if hasattr(ind, "fitness") and ind.fitness.valid
            else (float("inf"), float("inf"))
        ),
    )

    # Select top N% for optimization
    num_to_optimize = max(1, int(len(sorted_pop) * population_coverage))
    individuals_to_optimize = sorted_pop[:num_to_optimize]

    improved_population = population.copy()

    # Determine number of workers - USE ALL AVAILABLE CORES
    num_workers = get_cpu_count() if parallel else 1

    for _pop_idx, original_ind in enumerate(individuals_to_optimize):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            metrics["timed_out"] = True
            break

        # Fast shallow copy + list copy (10-50x faster than deepcopy)
        # Same optimization used in RL environment (proven effective)
        improved_ind = type(original_ind)(original_ind[:])  # Copy genes list
        # Copy fitness if it exists
        if hasattr(original_ind, "fitness") and hasattr(original_ind.fitness, "values"):
            improved_ind.fitness.values = original_ind.fitness.values  # type: ignore[attr-defined]

        if parallel and num_workers > 1:
            # PARALLEL: Optimize all genes concurrently
            gene_tasks = [
                (
                    improved_ind[gene_idx],
                    improved_ind,
                    gene_idx,
                    context,
                    max_neighborhood_size,
                )
                for gene_idx in range(len(improved_ind))
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all gene optimization tasks
                futures = {
                    executor.submit(_optimize_gene_wrapper_exhaustive, task): idx
                    for idx, task in enumerate(gene_tasks)
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        metrics["timed_out"] = True
                        # Cancel remaining tasks
                        for f in futures:
                            f.cancel()
                        break

                    try:
                        gene_idx, improved_gene, improvement = future.result(timeout=5)

                        # Update gene if improved
                        if improvement > 0:
                            improved_ind[gene_idx] = improved_gene
                            metrics["genes_improved"] += 1
                            metrics["total_improvement"] += improvement

                        metrics["genes_evaluated"] += 1

                    except Exception as e:
                        # Log error but continue
                        logging.getLogger(__name__).warning(
                            "Gene optimization failed for gene %s: %s",
                            futures[future],
                            e,
                        )

        else:
            # SEQUENTIAL: Fallback for small populations or debugging
            for gene_idx in range(len(improved_ind)):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    metrics["timed_out"] = True
                    break

                improved_gene, improvement = optimize_gene_exhaustive(
                    gene=improved_ind[gene_idx],
                    individual=improved_ind,
                    gene_index=gene_idx,
                    context=context,
                    max_neighborhood_size=max_neighborhood_size,
                )

                # Update gene if improved
                if improvement > 0:
                    improved_ind[gene_idx] = improved_gene
                    metrics["genes_improved"] += 1
                    metrics["total_improvement"] += improvement

                metrics["genes_evaluated"] += 1

        # Replace in population
        original_index = population.index(original_ind)
        improved_population[original_index] = improved_ind
        metrics["individuals_processed"] += 1

        if metrics["timed_out"]:
            break

    metrics["execution_time"] = time.time() - start_time

    return improved_population, metrics


def apply_greedy_search(
    population: list[Individual],
    context: SchedulingContext,
    population_coverage: float = 0.5,
    max_iterations: int = 10,
    timeout_seconds: int = 60,
    parallel: bool = True,  # NEW: Enable/disable parallelization
) -> tuple[list[Individual], MetricsDict]:
    """
    Apply greedy local search to population.

    PARALLELIZED: Optimizes all genes of an individual in parallel using ProcessPoolExecutor.
    Expected speedup: 4-8x on multi-core systems.

    Used for stagnation-triggered adaptive optimization.
    Performs hill climbing on ALL genes in selected individuals.
    Faster than exhaustive but still thorough.

    Args:
        population: List of individuals (schedules)
        context: Scheduling context
        population_coverage: Fraction of population to optimize (0.5 = top 50%)
        max_iterations: Max iterations per gene for greedy search
        timeout_seconds: Abort if exceeds this time
        parallel: Use parallel gene optimization (default True)

    Returns:
        Tuple of (improved_population, metrics_dict)

    Metrics:
        - individuals_processed: Number of individuals optimized
        - genes_improved: Total genes improved across population
        - total_improvement: Sum of violation reductions
        - execution_time: Time taken in seconds
        - timed_out: Whether operation was aborted
    """
    start_time = time.time()

    metrics: MetricsDict = {
        "individuals_processed": 0,
        "genes_improved": 0,
        "total_improvement": 0,
        "genes_evaluated": 0,
        "execution_time": 0.0,
        "timed_out": False,
    }

    # Sort population by fitness (best first)
    sorted_pop = sorted(
        population,
        key=lambda ind: (
            ind.fitness.values
            if hasattr(ind, "fitness") and ind.fitness.valid
            else (float("inf"), float("inf"))
        ),
    )

    # Select top N% for optimization
    num_to_optimize = max(1, int(len(sorted_pop) * population_coverage))
    individuals_to_optimize = sorted_pop[:num_to_optimize]

    improved_population = population.copy()

    # Determine number of workers - USE ALL AVAILABLE CORES
    num_workers = get_cpu_count() if parallel else 1

    for _pop_idx, original_ind in enumerate(individuals_to_optimize):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            metrics["timed_out"] = True
            break

        # Fast shallow copy + list copy (10-50x faster than deepcopy)
        improved_ind = type(original_ind)(original_ind[:])  # Copy genes list
        # Copy fitness if it exists
        if hasattr(original_ind, "fitness") and hasattr(original_ind.fitness, "values"):
            improved_ind.fitness.values = original_ind.fitness.values  # type: ignore[attr-defined]

        if parallel and num_workers > 1:
            # PARALLEL: Optimize all genes concurrently
            gene_tasks = [
                (
                    improved_ind[gene_idx],
                    improved_ind,
                    gene_idx,
                    context,
                    max_iterations,
                )
                for gene_idx in range(len(improved_ind))
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all gene optimization tasks
                futures = {
                    executor.submit(_optimize_gene_wrapper_greedy, task): idx
                    for idx, task in enumerate(gene_tasks)
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        metrics["timed_out"] = True
                        # Cancel remaining tasks
                        for f in futures:
                            f.cancel()
                        break

                    try:
                        gene_idx, improved_gene, improvement = future.result(timeout=5)

                        # Update gene if improved
                        if improvement > 0:
                            improved_ind[gene_idx] = improved_gene
                            metrics["genes_improved"] += 1
                            metrics["total_improvement"] += improvement

                        metrics["genes_evaluated"] += 1

                    except Exception as e:
                        # Log error but continue
                        logging.getLogger(__name__).warning(
                            "Gene optimization failed for gene %s: %s",
                            futures[future],
                            e,
                        )

        else:
            # SEQUENTIAL: Fallback for small populations or debugging
            for gene_idx in range(len(improved_ind)):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    metrics["timed_out"] = True
                    break

                improved_gene, improvement = optimize_gene_greedy(
                    gene=improved_ind[gene_idx],
                    individual=improved_ind,
                    gene_index=gene_idx,
                    context=context,
                    max_iterations=max_iterations,
                )

                # Update gene if improved
                if improvement > 0:
                    improved_ind[gene_idx] = improved_gene
                    metrics["genes_improved"] += 1
                    metrics["total_improvement"] += improvement

                metrics["genes_evaluated"] += 1

        # Replace in population
        original_index = population.index(original_ind)
        improved_population[original_index] = improved_ind
        metrics["individuals_processed"] += 1

        if metrics["timed_out"]:
            break

    metrics["execution_time"] = time.time() - start_time

    return improved_population, metrics


def apply_selective_probabilistic(
    individual: list[SessionGene],
    context: SchedulingContext,
    apply_probability: float = 0.3,
) -> tuple[list[SessionGene], bool]:
    """
    Apply selective repair probabilistically.

    Used for post-mutation cleanup. Only repairs violated genes,
    and only applies to a fraction of individuals (not all).

    Args:
        individual: Single individual to potentially repair
        context: Scheduling context
        apply_probability: Probability of applying repair (0.3 = 30%)

    Returns:
        Tuple of (individual, was_repaired)
        - individual: Potentially repaired individual
        - was_repaired: Whether repair was applied
    """
    import random

    # Probabilistic gate
    if random.random() > apply_probability:
        return individual, False

    # Apply selective repair (violations only)
    from src.ga.repair.basic import repair_individual_unified
    from src.ga.repair.detector import detect_violated_genes

    violations = detect_violated_genes(individual, context, strategy="hybrid")

    if not violations:
        return individual, False  # No violations, nothing to repair

    # Apply repair (pass max_iterations and selective as separate args)
    stats = repair_individual_unified(
        individual, context, max_iterations=3, selective=True
    )

    return individual, stats.get("total_fixes", 0) > 0


if __name__ == "__main__":
    """Quick test of IGLS module."""
    from rich.console import Console

    console = Console()
    console.print("\n[bold cyan]Intensive Global Local Search (IGLS)[/bold cyan]")
    console.print("[dim]Population-level intensive optimization[/dim]")
    console.print("\nAvailable functions:")
    console.print("  • apply_exhaustive_search() - Steepest descent (gen 3, 25)")
    console.print("  • apply_greedy_search() - Hill climbing (stagnation)")
    console.print("  • apply_selective_probabilistic() - Probabilistic cleanup")
    console.print()
