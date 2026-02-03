"""
Simple constraint lists - all constraints always enabled.

This module provides direct access to all constraint functions without
the complexity of a registry system. All constraints are always enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schedule_engine.domain.course import Course
    from schedule_engine.domain.session import CourseSession


@dataclass
class ConstraintInfo:
    """Simple constraint metadata."""

    name: str
    function: Callable[..., int]
    weight: float
    needs_courses: bool


# Import all hard constraint functions
from schedule_engine.constraints.hard import (
    course_completeness,
    instructor_exclusivity,
    instructor_qualifications,
    instructor_time_availability,
    room_exclusivity,
    room_suitability,
    room_time_availability,
    student_group_exclusivity,
)

# Import all soft constraint functions
from schedule_engine.constraints.soft import (
    break_placement_compliance,
    instructor_schedule_compactness,
    paired_cohort_practical_alignment,
    session_continuity,
    student_lunch_break,
    student_schedule_compactness,
)

# All hard constraints (always enabled)
HARD_CONSTRAINTS: list[ConstraintInfo] = [
    ConstraintInfo("student_group_exclusivity", student_group_exclusivity, 1.0, False),
    ConstraintInfo("instructor_exclusivity", instructor_exclusivity, 1.0, False),
    ConstraintInfo("instructor_qualifications", instructor_qualifications, 1.0, True),
    ConstraintInfo("room_suitability", room_suitability, 1.0, False),
    ConstraintInfo(
        "instructor_time_availability", instructor_time_availability, 1.0, False
    ),
    ConstraintInfo("room_time_availability", room_time_availability, 1.0, False),
    ConstraintInfo("course_completeness", course_completeness, 1.0, True),
    ConstraintInfo("room_exclusivity", room_exclusivity, 1.0, False),
]

# All soft constraints (always enabled)
SOFT_CONSTRAINTS: list[ConstraintInfo] = [
    ConstraintInfo(
        "student_schedule_compactness", student_schedule_compactness, 1.5, False
    ),
    ConstraintInfo(
        "instructor_schedule_compactness", instructor_schedule_compactness, 1.0, False
    ),
    ConstraintInfo("student_lunch_break", student_lunch_break, 1.0, False),
    ConstraintInfo("session_continuity", session_continuity, 1.0, False),
    ConstraintInfo(
        "paired_cohort_practical_alignment",
        paired_cohort_practical_alignment,
        1.0,
        True,
    ),
    ConstraintInfo(
        "break_placement_compliance", break_placement_compliance, 1.0, False
    ),
]

# Convenience lookups
HARD_CONSTRAINT_NAMES: set[str] = {c.name for c in HARD_CONSTRAINTS}
SOFT_CONSTRAINT_NAMES: set[str] = {c.name for c in SOFT_CONSTRAINTS}


def get_hard_constraints() -> list[ConstraintInfo]:
    """Get all hard constraints."""
    return HARD_CONSTRAINTS


def get_soft_constraints() -> list[ConstraintInfo]:
    """Get all soft constraints."""
    return SOFT_CONSTRAINTS


def evaluate_hard_constraints(
    sessions: list[CourseSession],
    courses: dict[tuple, Course] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Evaluate all hard constraints.

    Args:
        sessions: Decoded course sessions
        courses: Course map (needed for some constraints)

    Returns:
        (total_penalty, breakdown_dict)
    """
    total = 0.0
    breakdown: dict[str, float] = {}

    for c in HARD_CONSTRAINTS:
        if c.needs_courses:
            penalty = c.function(sessions, courses)
        else:
            penalty = c.function(sessions)
        weighted = penalty * c.weight
        breakdown[c.name] = penalty
        total += weighted

    return total, breakdown


def evaluate_soft_constraints(
    sessions: list[CourseSession],
    courses: dict[tuple, Course] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Evaluate all soft constraints.

    Args:
        sessions: Decoded course sessions
        courses: Course map (needed for some constraints)

    Returns:
        (total_penalty, breakdown_dict)
    """
    total = 0.0
    breakdown: dict[str, float] = {}

    for c in SOFT_CONSTRAINTS:
        if c.needs_courses:
            penalty = c.function(sessions, courses)
        else:
            penalty = c.function(sessions)
        weighted = penalty * c.weight
        breakdown[c.name] = penalty
        total += weighted

    return total, breakdown


def evaluate_all(
    sessions: list[CourseSession],
    courses: dict[tuple, Course] | None = None,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """
    Evaluate all constraints (hard and soft).

    Args:
        sessions: Decoded course sessions
        courses: Course map (needed for some constraints)

    Returns:
        (hard_total, soft_total, hard_breakdown, soft_breakdown)
    """
    hard_total, hard_breakdown = evaluate_hard_constraints(sessions, courses)
    soft_total, soft_breakdown = evaluate_soft_constraints(sessions, courses)
    return hard_total, soft_total, hard_breakdown, soft_breakdown
