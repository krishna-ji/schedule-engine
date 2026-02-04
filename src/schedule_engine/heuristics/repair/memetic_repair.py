"""
Memetic Repair Heuristic

Applies intensive IGLS repair to a percentage of the elite population.
Intended for memetic/Hybrid GA modes where local search is applied to the
best individuals each generation.
"""

import random
from typing import Callable

from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.types import SchedulingContext
from schedule_engine.heuristics.registry import repair_heuristic


@repair_heuristic(
    name="memetic_repair",
    description="Memetic repair (intensive IGLS on elite individuals)",
    priority=6,
    enabled_by_default=False,
    requires_population=True,
    modifies_individual=True,
)
def memetic_repair(
    individual: list[SessionGene],
    population: list[list[SessionGene]],
    context: SchedulingContext,
    evaluator: Callable[[list[SessionGene]], tuple[float, float]] | None = None,
    elite_percentage: float = 0.05,
    memetic_iterations: int = 5,
) -> int:
    """Apply intensive repair to the elite subset of the population.

    Args:
        individual: Current individual (unused, for interface compatibility)
        population: Full population to repair elites from
        context: Scheduling context
        evaluator: Fitness evaluation function. If None, creates a basic evaluator.
        elite_percentage: Fraction of population to repair (default 5%)
        memetic_iterations: Number of repair steps per individual

    Returns:
        Total number of fixes applied across all elite individuals
    """
    from schedule_engine.ga.operators.repair_engine import RepairEngine

    if not population:
        return 0

    # Create a basic evaluator if none provided
    if evaluator is None:
        from schedule_engine.constraints.registry import (
            evaluate_hard_constraints,
            evaluate_soft_constraints,
        )

        def evaluator(ind: list[SessionGene]) -> tuple[float, float]:
            hard = evaluate_hard_constraints(ind, context)
            soft = evaluate_soft_constraints(ind, context)
            return (hard, soft)

    elite_count = max(1, int(len(population) * elite_percentage))
    total_fixes = 0

    # Create repair engine
    engine = RepairEngine(
        context=context,
        evaluator=evaluator,
        policy="round_robin",
        max_steps=memetic_iterations,
        max_candidates=30,
        budget_ms=100.0,
        rng=random.Random(),
    )

    # Population assumed sorted by fitness externally (caller uses selBest)
    for elite_individual in population[:elite_count]:
        stats = engine.repair_individual(elite_individual)
        total_fixes += stats.applied_steps

    return total_fixes
