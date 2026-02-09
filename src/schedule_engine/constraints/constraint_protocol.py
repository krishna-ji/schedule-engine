"""Constraint Protocol and OOP constraint wrappers.

Phase 2 of the OOP redesign. Each constraint is a class with:
- ``name``: human-readable identifier
- ``weight``: penalty multiplier
- ``kind``: "hard" or "soft"
- ``evaluate(tt)``: accepts a ``Timetable``, returns a numeric penalty

This eliminates:
- The ``needs_courses`` branching in all_constraints.py, fitness.py,
  detailed_fitness.py, run_helpers.py (7+ sites)
- The ``ConstraintInfo`` dataclass and its function/flag coupling
- The ``constraint_needs_courses()`` lookup function

Backward compatibility: The existing function-based constraints in
hard.py and soft.py are preserved. These wrapper classes call them
internally, using ``Timetable`` to provide the right arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from schedule_engine.domain.timetable import Timetable

__all__ = [
    "Constraint",
    "HARD_CONSTRAINT_CLASSES",
    "SOFT_CONSTRAINT_CLASSES",
    "ALL_CONSTRAINTS",
]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class Constraint(Protocol):
    """Protocol for all constraints (hard and soft)."""

    name: str
    weight: float
    kind: Literal["hard", "soft"]

    def evaluate(self, tt: Timetable) -> float:
        """Evaluate the constraint against a timetable.

        Returns:
            Penalty score (0 = no violation).
        """
        ...


# ---------------------------------------------------------------------------
# Hard constraint classes
# ---------------------------------------------------------------------------
class StudentGroupExclusivity:
    name = "student_group_exclusivity"
    weight = 1.0
    kind: Literal["hard"] = "hard"

    def evaluate(self, tt: Timetable) -> float:
        return tt.count_group_violations()


class InstructorExclusivity:
    name = "instructor_exclusivity"
    weight = 1.0
    kind: Literal["hard"] = "hard"

    def evaluate(self, tt: Timetable) -> float:
        return tt.count_instructor_violations()


class RoomExclusivity:
    name = "room_exclusivity"
    weight = 1.0
    kind: Literal["hard"] = "hard"

    def evaluate(self, tt: Timetable) -> float:
        return tt.count_room_violations()


class InstructorQualifications:
    name = "instructor_qualifications"
    weight = 1.0
    kind: Literal["hard"] = "hard"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.hard import instructor_qualifications

        return instructor_qualifications(tt.sessions, tt.context.courses)


class RoomSuitability:
    name = "room_suitability"
    weight = 1.0
    kind: Literal["hard"] = "hard"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.hard import room_suitability

        return room_suitability(tt.sessions)


class InstructorTimeAvailability:
    name = "instructor_time_availability"
    weight = 1.0
    kind: Literal["hard"] = "hard"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.hard import instructor_time_availability

        return instructor_time_availability(tt.sessions)


class RoomTimeAvailability:
    name = "room_time_availability"
    weight = 1.0
    kind: Literal["hard"] = "hard"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.hard import room_time_availability

        return room_time_availability(tt.sessions)


class CourseCompleteness:
    name = "course_completeness"
    weight = 1.0
    kind: Literal["hard"] = "hard"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.hard import course_completeness

        return course_completeness(tt.sessions, tt.context.courses)


# ---------------------------------------------------------------------------
# Soft constraint classes
# ---------------------------------------------------------------------------
class StudentScheduleCompactness:
    name = "student_schedule_compactness"
    weight = 1.0
    kind: Literal["soft"] = "soft"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.soft import student_schedule_compactness

        return student_schedule_compactness(tt.sessions)


class InstructorScheduleCompactness:
    name = "instructor_schedule_compactness"
    weight = 1.0
    kind: Literal["soft"] = "soft"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.soft import instructor_schedule_compactness

        return instructor_schedule_compactness(tt.sessions)


class StudentLunchBreak:
    name = "student_lunch_break"
    weight = 1.0
    kind: Literal["soft"] = "soft"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.soft import student_lunch_break

        return student_lunch_break(tt.sessions)


class SessionContinuity:
    name = "session_continuity"
    weight = 1.0
    kind: Literal["soft"] = "soft"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.soft import session_continuity

        return session_continuity(tt.sessions)


class PairedCohortPracticalAlignment:
    name = "paired_cohort_practical_alignment"
    weight = 1.0
    kind: Literal["soft"] = "soft"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.soft import paired_cohort_practical_alignment

        return paired_cohort_practical_alignment(tt.sessions, tt.context.courses)


class BreakPlacementCompliance:
    name = "break_placement_compliance"
    weight = 1.0
    kind: Literal["soft"] = "soft"

    def evaluate(self, tt: Timetable) -> float:
        from schedule_engine.constraints.soft import break_placement_compliance

        return break_placement_compliance(tt.sessions)


# ---------------------------------------------------------------------------
# Registries (instances, not types — ready to use)
# ---------------------------------------------------------------------------
HARD_CONSTRAINT_CLASSES: list[Constraint] = [
    StudentGroupExclusivity(),
    InstructorExclusivity(),
    RoomExclusivity(),
    InstructorQualifications(),
    RoomSuitability(),
    InstructorTimeAvailability(),
    RoomTimeAvailability(),
    CourseCompleteness(),
]

SOFT_CONSTRAINT_CLASSES: list[Constraint] = [
    StudentScheduleCompactness(),
    InstructorScheduleCompactness(),
    StudentLunchBreak(),
    SessionContinuity(),
    PairedCohortPracticalAlignment(),
    BreakPlacementCompliance(),
]

ALL_CONSTRAINTS: list[Constraint] = HARD_CONSTRAINT_CLASSES + SOFT_CONSTRAINT_CLASSES
