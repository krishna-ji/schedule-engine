# Constraint System (simplified - all constraints always enabled)
from __future__ import annotations

from typing import TYPE_CHECKING

from schedule_engine.constraints import HARD_CONSTRAINT_CLASSES, SOFT_CONSTRAINT_CLASSES
from schedule_engine.domain.course import Course
from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.group import Group
from schedule_engine.domain.instructor import Instructor
from schedule_engine.domain.room import Room
from schedule_engine.domain.timetable import Timetable
from schedule_engine.io.decoder import decode_individual

if TYPE_CHECKING:
    pass


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
    """
    Evaluates a timetable individual using both hard and soft constraints.

    Hard constraints affect feasibility and must ideally reach zero.
    Soft constraints reflect schedule quality and should be minimized.

    Returns:
        Tuple[int, int]: (hard_penalty_score, soft_penalty_score)
    """
    # Get rooms from context if not provided
    if rooms is None:
        # For backward compatibility, create empty rooms dict
        rooms = {}

    sessions = decode_individual(individual, courses, instructors, groups, rooms)
    # Build Timetable from decoded sessions
    from schedule_engine.domain.types import SchedulingContext
    from schedule_engine.io.time_system import QuantumTimeSystem

    context = SchedulingContext(
        courses=courses,
        instructors=instructors,
        groups=groups,
        rooms=rooms,
        qts=QuantumTimeSystem(),
    )
    tt = Timetable(genes=individual, context=context)
    return evaluate_from_timetable(tt)

