"""Fitness evaluation for experiment notebooks.

Provides constraint evaluation functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from src.constraints.hard import (
    instructor_exclusivity,
    instructor_qualifications,
    room_exclusivity,
    room_suitability,
    student_group_exclusivity,
)
from src.decoder.individual_decoder import decode_individual
from src.ga.sessiongene import SessionGene

if TYPE_CHECKING:
    from src.entities.decoded_session import CourseSession
    from src.notebooks.data_loader import ScheduleData


@dataclass
class ConstraintResult:
    """Container for constraint evaluation results."""

    hard_violations: int
    soft_penalty: int
    breakdown: dict[str, int]

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
        soft_constraints: Optional list of soft constraint functions

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

    # Soft constraints
    soft_total = 0
    if soft_constraints:
        for constraint_fn in soft_constraints:
            soft_total += constraint_fn(sessions)

    return ConstraintResult(
        hard_violations=hard_total,
        soft_penalty=soft_total,
        breakdown=hard_breakdown,
    )


def create_evaluator(
    data: ScheduleData,
    soft_constraints: list[Callable[[list[CourseSession]], int]] | None = None,
) -> Callable[[list[SessionGene]], tuple[int, int]]:
    """Create fitness evaluation function for DEAP.

    Args:
        data: Schedule data
        soft_constraints: Optional soft constraint functions

    Returns:
        Evaluation function compatible with DEAP toolbox

    Example:
        >>> evaluate = create_evaluator(data)
        >>> toolbox.register("evaluate", evaluate)
    """

    def evaluate(individual: list[SessionGene]) -> tuple[int, int]:
        result = evaluate_constraints(individual, data, soft_constraints)
        return result.as_tuple()

    return evaluate


def get_constraint_breakdown(
    individual: list[SessionGene],
    data: ScheduleData,
) -> dict[str, int]:
    """Get detailed constraint breakdown for analysis.

    Args:
        individual: Individual to analyze
        data: Schedule data

    Returns:
        Dictionary mapping constraint names to violation counts
    """
    result = evaluate_constraints(individual, data)
    return result.breakdown
