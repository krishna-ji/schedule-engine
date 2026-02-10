"""GA evaluator functions for fitness evaluation.

Provides functional APIs for evaluating timetable fitness using the constraint system.
For the OOP API, prefer ``schedule_engine.constraints.Evaluator``.

Functions:
    evaluate: Full individual evaluation (decode → build Timetable → score)
    evaluate_from_timetable: Evaluate a pre-built Timetable (avoids re-decoding)
    evaluate_detailed: Per-constraint breakdown of penalties
    evaluate_from_detailed: Convert detailed breakdown to totals
"""

from __future__ import annotations

from schedule_engine.constraints import HARD_CONSTRAINT_CLASSES, SOFT_CONSTRAINT_CLASSES
from schedule_engine.domain.course import Course
from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.group import Group
from schedule_engine.domain.instructor import Instructor
from schedule_engine.domain.room import Room
from schedule_engine.domain.timetable import Timetable
from schedule_engine.domain.types import SchedulingContext
from schedule_engine.io.decoder import decode_individual
from schedule_engine.io.time_system import QuantumTimeSystem


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_from_timetable(tt: Timetable) -> tuple[int, int]:
    """Evaluate fitness using a pre-built Timetable.

    This is the preferred entry point — avoids a redundant
    ``decode_individual()`` call when the caller already has a Timetable.
    """
    hard_penalty = sum(c.weight * c.evaluate(tt) for c in HARD_CONSTRAINT_CLASSES)
    soft_penalty = sum(c.weight * c.evaluate(tt) for c in SOFT_CONSTRAINT_CLASSES)
    return (int(hard_penalty), int(soft_penalty))


def evaluate(
    individual: list[SessionGene],
    courses: dict[tuple, Course],  # Keys are (course_code, course_type) tuples
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room] | None = None,
) -> tuple[int, int]:
    """Evaluate a timetable individual using both hard and soft constraints.

    Hard constraints affect feasibility and must ideally reach zero.
    Soft constraints reflect schedule quality and should be minimized.

    Returns:
        Tuple of ``(hard_penalty_score, soft_penalty_score)``.
    """
    if rooms is None:
        rooms = {}

    decode_individual(individual, courses, instructors, groups, rooms)

    context = SchedulingContext(
        courses=courses,
        instructors=instructors,
        groups=groups,
        rooms=rooms,
        qts=QuantumTimeSystem(),
    )
    tt = Timetable(genes=individual, context=context)
    return evaluate_from_timetable(tt)


# ---------------------------------------------------------------------------
# Detailed (per-constraint) evaluation
# ---------------------------------------------------------------------------


def evaluate_detailed(
    individual: list[SessionGene],
    courses: dict[tuple, Course],  # Keys are (course_code, course_type) tuples
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Evaluate a timetable individual with detailed constraint breakdown.

    Returns:
        Tuple of ``(hard_constraint_details, soft_constraint_details)``
        where each dict maps constraint name → weighted penalty.
    """
    if rooms is None:
        rooms = {}

    context = SchedulingContext(
        courses=courses,
        instructors=instructors,
        groups=groups,
        rooms=rooms,
        qts=QuantumTimeSystem(),
    )
    tt = Timetable(genes=individual, context=context)

    hard_details = {}
    for constraint in HARD_CONSTRAINT_CLASSES:
        penalty = constraint.evaluate(tt)
        hard_details[constraint.name] = int(constraint.weight * penalty)

    soft_details = {}
    for constraint in SOFT_CONSTRAINT_CLASSES:
        penalty = constraint.evaluate(tt)
        soft_details[constraint.name] = constraint.weight * penalty

    return hard_details, soft_details


def evaluate_from_detailed(
    hard_details: dict[str, int], soft_details: dict[str, int]
) -> tuple[int, int]:
    """Convert detailed constraint breakdown to total penalties.

    Returns:
        Tuple of ``(total_hard_penalty, total_soft_penalty)``.
    """
    total_hard = sum(hard_details.values())
    total_soft = sum(soft_details.values())
    return int(total_hard), int(total_soft)
