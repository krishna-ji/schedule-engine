"""
Memetic Repair Heuristic

Applies intensive IGLS repair to a percentage of the elite population.
Intended for memetic/Hybrid GA modes where local search is applied to the
best individuals each generation.
"""

from src.domain.types import SchedulingContext
from src.domain.gene import SessionGene
from src.heuristics.registry import repair_heuristic


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
    elite_percentage: float = 0.05,
    memetic_iterations: int = 5,
) -> int:
    """Apply intensive repair to the elite subset of the population."""
    from src.ga.operators.repair import repair_individual_unified

    if not population:
        return 0

    elite_count = max(1, int(len(population) * elite_percentage))
    total_fixes = 0

    # Population assumed sorted by fitness externally (caller uses selBest)
    for elite_individual in population[:elite_count]:
        stats = repair_individual_unified(
            elite_individual,
            context,
            max_iterations=memetic_iterations,
            selective=True,
        )
        total_fixes += stats.get("total_fixes", 0)

    return total_fixes
