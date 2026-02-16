"""GA evaluator functions for fitness evaluation.

Thin wrappers around ``src.constraints.Evaluator`` that preserve
the functional API used throughout the GA pipeline.

For the OOP API, prefer ``src.constraints.Evaluator`` directly.

Functions:
    evaluate: Full individual evaluation (decode → build Timetable → score)
    evaluate_from_timetable: Evaluate a pre-built Timetable (avoids re-decoding)
    evaluate_detailed: Per-constraint breakdown of penalties
    evaluate_from_detailed: Convert detailed breakdown to totals
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.constraints.evaluator import Evaluator
from src.domain.timetable import Timetable
from src.domain.types import SchedulingContext
from src.io.decoder import decode_individual
from src.io.time_system import QuantumTimeSystem

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.gene import SessionGene
    from src.domain.group import Group
    from src.domain.instructor import Instructor
    from src.domain.room import Room

# Module-level default evaluator (lazy singleton)
_default_evaluator: Evaluator | None = None


def _get_evaluator() -> Evaluator:
    """Return the shared default Evaluator instance."""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = Evaluator()
    return _default_evaluator


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_from_timetable(tt: Timetable) -> tuple[int, int]:
    """Evaluate fitness using a pre-built Timetable.

    This is the preferred entry point — avoids a redundant
    ``decode_individual()`` call when the caller already has a Timetable.
    """
    ev = _get_evaluator()
    hard, soft = ev.fitness_from_timetable(tt)
    return (int(hard), int(soft))


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
        available_quanta=[],
    )
    tt = Timetable(genes=individual, context=context, qts=QuantumTimeSystem())
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
) -> tuple[dict[str, int], dict[str, float]]:
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
        available_quanta=[],
    )
    tt = Timetable(genes=individual, context=context, qts=QuantumTimeSystem())

    ev = _get_evaluator()
    hard_details = {c.name: int(c.weight * c.evaluate(tt)) for c in ev.hard}
    soft_details = {c.name: c.weight * c.evaluate(tt) for c in ev.soft}
    return hard_details, soft_details


def evaluate_from_detailed(
    hard_details: dict[str, int], soft_details: dict[str, float]
) -> tuple[int, int]:
    """Convert detailed constraint breakdown to total penalties.

    Returns:
        Tuple of ``(total_hard_penalty, total_soft_penalty)``.
    """
    total_hard = sum(hard_details.values())
    total_soft = sum(soft_details.values())
    return int(total_hard), int(total_soft)
