"""Shared test fixtures for schedule-engine test suite.

Provides factory functions and pytest fixtures for creating domain objects
(Course, Instructor, Group, Room, SessionGene, SchedulingContext, Timetable)
with full configurability for testing constraints, repairs, heuristics, and
the GA pipeline.

QuantumTimeSystem defaults (reference):
    6 operational days: Sunday-Friday
    7 quanta/day: 10:00-17:00 (each quantum = 60 min)
    42 total quanta: 0-41
    Day offsets: Sun=0, Mon=7, Tue=14, Wed=21, Thu=28, Fri=35
    Midday break: within-day quantum 2 (12:00-13:00, 1 quantum per day)
    Break window: within-day quanta {2, 3} (12:00-14:00)
"""

from __future__ import annotations

import pytest

from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.domain.timetable import Timetable
from src.domain.types import SchedulingContext
from src.io.time_system import QuantumTimeSystem

# ---------------------------------------------------------------------------
# Factory functions — use these directly in tests
# ---------------------------------------------------------------------------


def make_course(
    course_id: str = "CS101",
    course_type: str = "theory",
    quanta: int = 2,
    room_feat: str = "lecture",
    groups: list[str] | None = None,
    instructors: list[str] | None = None,
) -> Course:
    """Create a Course with sensible defaults."""
    return Course(
        course_id=course_id,
        name=f"Course {course_id}",
        quanta_per_week=quanta,
        required_room_features=room_feat,
        enrolled_group_ids=groups if groups is not None else ["G1"],
        qualified_instructor_ids=instructors if instructors is not None else ["I1"],
        course_type=course_type,
    )


def make_instructor(
    instructor_id: str = "I1",
    courses: list | None = None,
    is_full_time: bool = True,
    available_quanta: set[int] | None = None,
) -> Instructor:
    """Create an Instructor (full-time by default)."""
    return Instructor(
        instructor_id=instructor_id,
        name=f"Instructor {instructor_id}",
        qualified_courses=courses or [],
        is_full_time=is_full_time,
        available_quanta=available_quanta or set(),
    )


def make_group(
    group_id: str = "G1",
    students: int = 30,
    courses: list[str] | None = None,
) -> Group:
    """Create a student Group."""
    return Group(
        group_id=group_id,
        name=f"Group {group_id}",
        student_count=students,
        enrolled_courses=courses if courses is not None else ["CS101"],
    )


def make_room(
    room_id: str = "R1",
    capacity: int = 50,
    features: str = "lecture",
    available_quanta: set[int] | None = None,
) -> Room:
    """Create a Room with optional availability constraints.

    Default: available at all 42 quanta (no restrictions).
    Pass available_quanta=set() to test rooms with no availability.
    """
    return Room(
        room_id=room_id,
        name=f"Room {room_id}",
        capacity=capacity,
        room_features=features,
        available_quanta=(
            available_quanta if available_quanta is not None else set(range(42))
        ),
    )


def make_gene(
    course_id: str = "CS101",
    course_type: str = "theory",
    instructor_id: str = "I1",
    group_ids: list[str] | None = None,
    room_id: str = "R1",
    start: int = 0,
    duration: int = 2,
) -> SessionGene:
    """Create a SessionGene with full control over all fields."""
    return SessionGene(
        course_id=course_id,
        course_type=course_type,
        instructor_id=instructor_id,
        group_ids=group_ids or ["G1"],
        room_id=room_id,
        start_quanta=start,
        num_quanta=duration,
    )


def make_context(
    courses: list[Course] | None = None,
    groups: list[Group] | None = None,
    instructors: list[Instructor] | None = None,
    rooms: list[Room] | None = None,
    available_quanta: list[int] | None = None,
    cohort_pairs: list[tuple[str, str]] | None = None,
    family_map: dict[str, set[str]] | None = None,
) -> SchedulingContext:
    """Build a SchedulingContext from lists (auto-keys by ID).

    Unlike _make_context in test_timetable.py, this accepts *lists* and
    builds the required dict structures automatically.
    """
    if courses is None:
        courses = [make_course()]
    if groups is None:
        groups = [make_group()]
    if instructors is None:
        instructors = [make_instructor()]
    if rooms is None:
        rooms = [make_room()]

    return SchedulingContext(
        courses={(c.course_id, c.course_type): c for c in courses},
        groups={g.group_id: g for g in groups},
        instructors={i.instructor_id: i for i in instructors},
        rooms={r.room_id: r for r in rooms},
        available_quanta=(
            available_quanta if available_quanta is not None else list(range(42))
        ),
        cohort_pairs=cohort_pairs,
        family_map=family_map or {},
    )


def make_timetable(
    genes: list[SessionGene],
    ctx: SchedulingContext | None = None,
    **ctx_kwargs,
) -> Timetable:
    """Build a Timetable from genes + optional context kwargs.

    If ctx is None, builds a default context. Extra kwargs are forwarded
    to make_context().
    """
    if ctx is None:
        ctx = make_context(**ctx_kwargs)
    return Timetable(genes, ctx)


def make_violation_free_timetable() -> tuple[Timetable, SchedulingContext]:
    """Create a timetable with zero hard constraint violations.

    Schedule:
        CS101(theory, 2q) → I1, G1, R1 @ Sun q=0-1
        CS102(theory, 2q) → I2, G2, R2 @ Sun q=0-1
    No overlaps: different groups, different instructors, different rooms.
    Both instructors qualified, rooms compatible, quanta match.
    """
    c1 = make_course(
        "CS101",
        "theory",
        quanta=2,
        room_feat="lecture",
        groups=["G1"],
        instructors=["I1"],
    )
    c2 = make_course(
        "CS102",
        "theory",
        quanta=2,
        room_feat="lecture",
        groups=["G2"],
        instructors=["I2"],
    )

    ctx = make_context(
        courses=[c1, c2],
        groups=[
            make_group("G1", courses=["CS101"]),
            make_group("G2", courses=["CS102"]),
        ],
        instructors=[make_instructor("I1"), make_instructor("I2")],
        rooms=[make_room("R1"), make_room("R2")],
    )

    genes = [
        make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
        make_gene("CS102", "theory", "I2", ["G2"], "R2", start=0, duration=2),
    ]

    return Timetable(genes, ctx), ctx


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_constraint_zero(constraint, timetable: Timetable, msg: str = ""):
    """Assert that a constraint evaluates to exactly 0 (no violation)."""
    penalty = constraint.evaluate(timetable)
    assert penalty == 0, f"{constraint.name}: expected 0 penalty, got {penalty}. {msg}"


def assert_constraint_positive(
    constraint, timetable: Timetable, expected: float | None = None, msg: str = ""
):
    """Assert that a constraint evaluates to > 0, optionally checking exact value."""
    penalty = constraint.evaluate(timetable)
    if expected is not None:
        assert penalty == expected, (
            f"{constraint.name}: expected penalty={expected}, got {penalty}. {msg}"
        )
    else:
        assert penalty > 0, (
            f"{constraint.name}: expected positive penalty, got 0. {msg}"
        )


def genes_differ_only_in(
    gene1: SessionGene, gene2: SessionGene, fields: set[str]
) -> bool:
    """Check that two genes are identical except for specified fields.

    Args:
        gene1, gene2: Genes to compare.
        fields: Set of field names that ARE allowed to differ
                (e.g. {"start_quanta", "room_id"}).

    Returns:
        True if genes match on all fields except those in `fields`.
    """
    all_fields = {
        "course_id",
        "course_type",
        "instructor_id",
        "group_ids",
        "room_id",
        "start_quanta",
        "num_quanta",
    }
    for f in all_fields - fields:
        v1 = getattr(gene1, f)
        v2 = getattr(gene2, f)
        if v1 != v2:
            return False
    return True


def structural_fields_preserved(before: SessionGene, after: SessionGene) -> bool:
    """Verify immutable structural fields are unchanged after an operator.

    Structural invariants: course_id, course_type, group_ids, num_quanta
    must NEVER be modified by crossover, mutation, repair, or perturbation.
    """
    return (
        before.course_id == after.course_id
        and before.course_type == after.course_type
        and before.group_ids == after.group_ids
        and before.num_quanta == after.num_quanta
    )


def count_hard_violations(timetable: Timetable) -> float:
    """Sum all hard constraint violations for a timetable."""
    from src.constraints.constraints import HARD_CONSTRAINT_CLASSES

    return sum(c.evaluate(timetable) for c in HARD_CONSTRAINT_CLASSES)


def count_soft_violations(timetable: Timetable) -> float:
    """Sum all soft constraint violations for a timetable."""
    from src.constraints.constraints import SOFT_CONSTRAINT_CLASSES

    return sum(c.evaluate(timetable) for c in SOFT_CONSTRAINT_CLASSES)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qts() -> QuantumTimeSystem:
    """Default QuantumTimeSystem: 6 days x 7 quanta = 42 total."""
    return QuantumTimeSystem()


@pytest.fixture
def clean_timetable() -> tuple[Timetable, SchedulingContext]:
    """A violation-free timetable for tests that need a clean starting point."""
    return make_violation_free_timetable()
