"""Fitness evaluation for experiment notebooks.

Provides constraint evaluation functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from src.constraints.hard import (
    instructor_exclusivity,
    instructor_qualifications,
    room_exclusivity,
    room_suitability,
    student_group_exclusivity,
)
from src.constraints.registry import get_all_soft_constraints
from src.constraints.soft import (
    instructor_schedule_compactness,
    student_lunch_break,
    student_schedule_compactness,
)
from src.decoder.individual_decoder import decode_individual
from src.ga.sessiongene import SessionGene

if TYPE_CHECKING:
    from src.entities.decoded_session import CourseSession
    from src.notebooks.data_loader import ScheduleData

# Default soft constraints for notebook experiments
DEFAULT_SOFT_CONSTRAINTS: list[Callable[[list[CourseSession]], int]] = [
    student_schedule_compactness,
    instructor_schedule_compactness,
    student_lunch_break,
]


@dataclass
class ConstraintResult:
    """Container for constraint evaluation results."""

    hard_violations: int
    soft_penalty: int
    breakdown: dict[str, int] = field(default_factory=dict)
    soft_breakdown: dict[str, int] = field(default_factory=dict)

    def as_tuple(self) -> tuple[int, int]:
        """Return as (hard, soft) tuple for DEAP fitness."""
        return (self.hard_violations, self.soft_penalty)


def evaluate_constraints(
    individual: list[SessionGene],
    data: ScheduleData,
    soft_constraints: list[Callable[[list[CourseSession]], int]] | None = None,
) -> ConstraintResult:
    """Evaluate all constraints on an individual.

    Args:
        individual: List of SessionGene
        data: Schedule data
        soft_constraints: Optional list of soft constraint functions (uses defaults if None)

    Returns:
        ConstraintResult with violations and breakdown
    """
    # Decode
    sessions = decode_individual(
        individual, data.courses, data.instructors, data.groups, data.rooms
    )

    # Hard constraints
    hard_breakdown = {
        "student_group_exclusivity": student_group_exclusivity(sessions),
        "instructor_exclusivity": instructor_exclusivity(sessions),
        "instructor_qualifications": instructor_qualifications(sessions, data.courses),
        "room_exclusivity": room_exclusivity(sessions),
        "room_suitability": room_suitability(sessions),
    }
    hard_total = sum(hard_breakdown.values())

    # Use default soft constraints if none provided
    soft_fns = (
        soft_constraints if soft_constraints is not None else DEFAULT_SOFT_CONSTRAINTS
    )

    # Soft constraints with breakdown tracking
    soft_breakdown: dict[str, int] = {}
    soft_total = 0
    for constraint_fn in soft_fns:
        # Get constraint name from function
        fn_name = getattr(constraint_fn, "__name__", str(constraint_fn))
        penalty = constraint_fn(sessions)
        soft_breakdown[fn_name] = penalty
        soft_total += penalty

    return ConstraintResult(
        hard_violations=hard_total,
        soft_penalty=soft_total,
        breakdown=hard_breakdown,
        soft_breakdown=soft_breakdown,
    )


def create_evaluator(
    data: ScheduleData,
    soft_constraints: list[Callable[[list[CourseSession]], int]] | None = None,
    use_soft_constraints: bool = True,
) -> Callable[[list[SessionGene]], tuple[int, int]]:
    """Create fitness evaluation function for DEAP.

    Args:
        data: Schedule data
        soft_constraints: Soft constraint functions (uses defaults if None)
        use_soft_constraints: Whether to evaluate soft constraints (default True)

    Returns:
        Evaluation function compatible with DEAP toolbox

    Example:
        >>> evaluate = create_evaluator(data)
        >>> toolbox.register("evaluate", evaluate)
    """
    # Determine which soft constraints to use
    if use_soft_constraints:
        soft_fns = (
            soft_constraints
            if soft_constraints is not None
            else DEFAULT_SOFT_CONSTRAINTS
        )
    else:
        soft_fns = []  # Empty list means no soft constraint evaluation

    def evaluate(individual: list[SessionGene]) -> tuple[int, int]:
        result = evaluate_constraints(individual, data, soft_fns)
        return result.as_tuple()

    return evaluate


def get_constraint_breakdown(
    individual: list[SessionGene],
    data: ScheduleData,
    include_soft: bool = True,
) -> dict[str, int]:
    """Get detailed constraint breakdown for analysis.

    Args:
        individual: Individual to analyze
        data: Schedule data
        include_soft: Whether to include soft constraints in breakdown

    Returns:
        Dictionary mapping constraint names to violation counts
    """
    result = evaluate_constraints(individual, data)
    breakdown = result.breakdown.copy()
    if include_soft:
        # Add soft constraints with "soft_" prefix for clarity
        for name, penalty in result.soft_breakdown.items():
            breakdown[f"soft_{name}"] = penalty
    return breakdown


def get_soft_breakdown(
    individual: list[SessionGene],
    data: ScheduleData,
) -> dict[str, int]:
    """Get detailed soft constraint breakdown.

    Args:
        individual: Individual to analyze
        data: Schedule data

    Returns:
        Dictionary mapping soft constraint names to penalty values
    """
    result = evaluate_constraints(individual, data)
    return result.soft_breakdown
