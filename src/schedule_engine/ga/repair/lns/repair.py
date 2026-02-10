"""
LNS Repair Heuristic

Large Neighborhood Search repair operator integrated as a heuristic.
This operator uses conflict detection and subproblem solving for repair.
"""

from schedule_engine.domain.types import SchedulingContext
from schedule_engine.domain.gene import SessionGene

# Import the original LNS repair logic
from schedule_engine.ga.repair.lns.operator import lns_igls_repair


def lns_repair(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_subproblem_size: int = 20,
    min_subproblem_size: int = 4,
    igls_max_iterations: int = 500,
    igls_time_limit: float = 5.0,
) -> int:
    """
    Apply LNS repair to fix constraint violations.

    Args:
        individual: Individual to repair
        context: Scheduling context
        max_subproblem_size: Maximum subproblem size
        min_subproblem_size: Minimum subproblem size
        igls_max_iterations: IGLS iteration limit
        igls_time_limit: IGLS time limit

    Returns:
        Number of modifications made (1 if repair applied, 0 if not)
    """
    # Convert context to the format expected by LNS
    courses_dict = (
        {
            (course.course_code, course.course_type): course
            for course in context.courses.values()
        }
        if hasattr(context.courses, "values")
        else context.courses
    )

    # Apply LNS repair using the original function
    repaired_individual = lns_igls_repair(
        individual=individual,
        courses=courses_dict,
        instructors=context.instructors,
        groups=context.groups,
        rooms=context.rooms,
        max_subproblem_size=max_subproblem_size,
        min_subproblem_size=min_subproblem_size,
        igls_max_iterations=igls_max_iterations,
        igls_time_limit=igls_time_limit,
        enable_diagnostics=False,  # Disable verbose output for heuristic mode
    )

    # Check if any modifications were made
    if repaired_individual is individual:
        return 0  # No changes
    else:
        # Copy the repaired genes back to the original individual
        individual[:] = repaired_individual
        return 1  # Repair was applied
