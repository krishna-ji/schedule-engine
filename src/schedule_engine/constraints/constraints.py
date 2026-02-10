"""Self-contained OOP constraint system.

Each constraint class:
- Contains its own evaluation logic (no delegation)
- Accepts weight and magic values as __init__ params
- Implements the Constraint protocol
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from schedule_engine.domain.timetable import Timetable

from schedule_engine.io.time_system import QuantumTimeSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (DRY for schedule-compactness & lunch-break constraints)
# ---------------------------------------------------------------------------


def _compute_gap_penalty(
    daily_map: dict[str, dict[str, set[int]]],
    break_quanta_by_day: dict[str, set[int]],
    gap_penalty: float,
) -> float:
    """Sum gap penalties across all entities/days, excluding break quanta.

    Parameters
    ----------
    daily_map:
        ``{entity_id: {day_name: set[within_day_quantum]}}``
        — works for both group_daily and instructor_daily.
    break_quanta_by_day:
        ``{day_name: set[within_day_quantum]}`` from ``qts.get_midday_break_quanta()``.
    gap_penalty:
        Penalty added per non-break gap quantum.
    """
    penalty = 0.0
    for days in daily_map.values():
        for day_name, quanta in days.items():
            if len(quanta) < 2:
                continue
            sorted_q = sorted(quanta)
            min_q, max_q = sorted_q[0], sorted_q[-1]
            break_q = break_quanta_by_day.get(day_name, set())
            for q in range(min_q, max_q + 1):
                if q not in quanta and q not in break_q:
                    penalty += gap_penalty
    return penalty


def _group_daily_map(tt: Timetable, qts: QuantumTimeSystem) -> dict[str, dict[str, set[int]]]:
    """Return group daily map — prefer Timetable's pre-built index, else compute."""
    if tt.group_daily:
        return tt.group_daily
    # Fallback: build on the fly (happens when Timetable was created without QTS)
    result: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for gene in tt.genes:
        for group_id in gene.group_ids:
            for q in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
                day, within_day = qts.quantum_to_day_and_within_day(q)
                result[group_id][day].add(within_day)
    return dict(result)


def _instructor_daily_map(tt: Timetable, qts: QuantumTimeSystem) -> dict[str, dict[str, set[int]]]:
    """Return instructor daily map — prefer Timetable's pre-built index, else compute."""
    if tt.instructor_daily:
        return tt.instructor_daily
    result: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for gene in tt.genes:
        for q in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
            day, within_day = qts.quantum_to_day_and_within_day(q)
            result[gene.instructor_id][day].add(within_day)
    return dict(result)

__all__ = [
    "Constraint",
    "ALL_CONSTRAINTS",
    "HARD_CONSTRAINT_CLASSES",
    "SOFT_CONSTRAINT_CLASSES",
    "build_constraints",
]


# Protocol


@runtime_checkable
class Constraint(Protocol):
    """Protocol that all constraints must implement."""

    name: str
    weight: float
    kind: Literal["hard", "soft"]

    def evaluate(self, tt: Timetable) -> float:
        """Evaluate constraint against timetable. Returns penalty (0 = no violation)."""
        ...


# HARD CONSTRAINTS


class StudentGroupExclusivity:
    """Groups cannot be in two places at the same time."""

    kind: Literal["hard"] = "hard"

    def __init__(self, weight: float = 1.0):
        self.name = "student_group_exclusivity"
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Use Timetable's pre-built conflict detection."""
        return tt.count_group_violations()


class InstructorExclusivity:
    """Instructors cannot teach two classes simultaneously."""

    kind: Literal["hard"] = "hard"

    def __init__(self, weight: float = 1.0):
        self.name = "instructor_exclusivity"
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Use Timetable's pre-built conflict detection."""
        return tt.count_instructor_violations()


class RoomExclusivity:
    """Rooms cannot host two sessions simultaneously."""

    kind: Literal["hard"] = "hard"

    def __init__(self, weight: float = 1.0):
        self.name = "room_exclusivity"
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Use Timetable's pre-built conflict detection."""
        return tt.count_room_violations()


class InstructorQualifications:
    """Instructors must be qualified to teach assigned courses."""

    kind: Literal["hard"] = "hard"

    def __init__(self, weight: float = 1.0):
        self.name = "instructor_qualifications"
        self.weight = weight
        self._warned_missing: set[tuple[str, str]] = set()
        self._warned_empty: set[tuple[str, str]] = set()

    def evaluate(self, tt: Timetable) -> float:
        """Check instructor qualification for each session."""
        violations = 0
        missing_courses = set()
        empty_qualifications = set()

        for gene in tt.genes:
            course_key = (gene.course_id, gene.course_type)

            # Missing course definition = violation
            if course_key not in tt.context.courses:
                violations += 1
                missing_courses.add(course_key)
                continue

            course = tt.context.courses[course_key]
            qualified = course.qualified_instructor_ids

            # Empty qualification list = violation
            if not qualified:
                violations += 1
                empty_qualifications.add(course_key)
                continue

            # Instructor not qualified = violation
            if gene.instructor_id not in qualified:
                violations += 1

        # Warn about data issues (once per constraint instance)
        if missing_courses:
            unseen = missing_courses - self._warned_missing
            if unseen:
                logger.warning(
                    f"Missing course definitions: {len(unseen)} courses {list(unseen)[:3]}"
                )
                self._warned_missing.update(unseen)

        if empty_qualifications:
            unseen = empty_qualifications - self._warned_empty
            if unseen:
                logger.warning(
                    f"Courses without qualified instructors: {len(unseen)} courses {list(unseen)[:3]}"
                )
                self._warned_empty.update(unseen)

        return violations


class RoomSuitability:
    """Rooms must be suitable for course type (lecture/lab/etc)."""

    kind: Literal["hard"] = "hard"

    def __init__(self, weight: float = 1.0):
        self.name = "room_suitability"
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Check room type compatibility."""
        from schedule_engine.utils.room_compatibility import is_room_type_compatible

        violations = 0

        for gene in tt.genes:
            course = tt.course_for_gene(gene)
            room = tt.room_for_gene(gene)

            if not room:
                continue

            required = getattr(course, "required_room_features", "lecture")
            room_type = getattr(room, "room_features", "lecture")

            # Normalize to lowercase
            required_str = (
                (required if isinstance(required, str) else str(required))
                .lower()
                .strip()
            )
            room_str = (
                (room_type if isinstance(room_type, str) else str(room_type))
                .lower()
                .strip()
            )

            if not is_room_type_compatible(required_str, room_str):
                violations += 1

        return violations


class InstructorTimeAvailability:
    """Instructors only teach during their available time slots."""

    kind: Literal["hard"] = "hard"

    def __init__(self, weight: float = 1.0):
        self.name = "instructor_time_availability"
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Check instructor availability for each session."""
        violations = 0

        for gene in tt.genes:
            instructor = tt.instructor_for_gene(gene)
            if not instructor:
                continue

            # Full-time = always available
            if instructor.is_full_time:
                continue

            # Part-time: check availability
            for q in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
                if q not in instructor.available_quanta:
                    violations += 1

        return violations


class RoomTimeAvailability:
    """Rooms only used during their available time slots."""

    kind: Literal["hard"] = "hard"

    def __init__(self, weight: float = 1.0):
        self.name = "room_time_availability"
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Check room availability for each session."""
        violations = 0

        for gene in tt.genes:
            room = tt.room_for_gene(gene)
            if not room:
                continue

            for q in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
                if q not in room.available_quanta:
                    violations += 1

        return violations


class CourseCompleteness:
    """Each course scheduled for exactly required quanta per week."""

    kind: Literal["hard"] = "hard"

    def __init__(self, weight: float = 1.0):
        self.name = "course_completeness"
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Check each (course, group) has correct total quanta."""
        # Count quanta per (course_key, group_id)
        course_group_quanta: dict[tuple[tuple[str, str], str], int] = defaultdict(int)

        for gene in tt.genes:
            course_key = (gene.course_id, gene.course_type)
            for group_id in gene.group_ids:
                key = (course_key, group_id)
                course_group_quanta[key] += gene.num_quanta

        # Check against expected
        violations = 0
        for course_key, course in tt.context.courses.items():
            expected = course.quanta_per_week
            for group_id in course.enrolled_group_ids:
                key = (course_key, group_id)
                actual = course_group_quanta.get(key, 0)
                if actual != expected:
                    violations += 1

        return violations


# SOFT CONSTRAINTS


class StudentScheduleCompactness:
    """Minimize idle time gaps in student schedules."""

    kind: Literal["soft"] = "soft"

    def __init__(
        self,
        weight: float = 1.0,
        gap_penalty_per_quantum: float = 1.0,
    ):
        self.name = "student_schedule_compactness"
        self.weight = weight
        self.gap_penalty = gap_penalty_per_quantum

    def evaluate(self, tt: Timetable) -> float:
        """Penalize gaps between first and last session per group per day."""
        qts = tt.qts or QuantumTimeSystem()
        break_quanta_by_day = qts.get_midday_break_quanta()
        return _compute_gap_penalty(_group_daily_map(tt, qts), break_quanta_by_day, self.gap_penalty)


class InstructorScheduleCompactness:
    """Minimize idle time gaps in instructor schedules."""

    kind: Literal["soft"] = "soft"

    def __init__(
        self,
        weight: float = 1.0,
        gap_penalty_per_quantum: float = 1.0,
    ):
        self.name = "instructor_schedule_compactness"
        self.weight = weight
        self.gap_penalty = gap_penalty_per_quantum

    def evaluate(self, tt: Timetable) -> float:
        """Penalize gaps between first and last session per instructor per day."""
        qts = tt.qts or QuantumTimeSystem()
        break_quanta_by_day = qts.get_midday_break_quanta()
        return _compute_gap_penalty(_instructor_daily_map(tt, qts), break_quanta_by_day, self.gap_penalty)


class StudentLunchBreak:
    """Students should have free time during lunch window."""

    kind: Literal["soft"] = "soft"

    def __init__(
        self,
        weight: float = 1.0,
        break_min_quanta: int = 2,
        penalty_per_missing_quantum: float = 5.0,
    ):
        self.name = "student_lunch_break"
        self.weight = weight
        self.break_min_quanta = break_min_quanta
        self.penalty_per_missing = penalty_per_missing_quantum

    def evaluate(self, tt: Timetable) -> float:
        """Penalize groups without sufficient lunch break."""
        penalty = 0.0
        qts = tt.qts or QuantumTimeSystem()
        break_quanta_by_day = qts.get_midday_break_quanta()

        # Re-use pre-built group_daily index (or compute fallback)
        for days in _group_daily_map(tt, qts).values():
            for day_name, occupied_quanta in days.items():
                if day_name not in break_quanta_by_day:
                    continue
                lunch_quanta = break_quanta_by_day[day_name]
                free_quanta = lunch_quanta - occupied_quanta
                if len(free_quanta) < self.break_min_quanta:
                    missing = self.break_min_quanta - len(free_quanta)
                    penalty += missing * self.penalty_per_missing

        return penalty


class SessionContinuity:
    """Penalize fragmented schedules (isolated single slots, bad block sizes)."""

    kind: Literal["soft"] = "soft"

    def __init__(
        self,
        weight: float = 1.0,
        isolated_slot_penalty: float = 10.0,
        preferred_block_sizes: tuple[int, int] = (2, 3),
    ):
        self.name = "session_continuity"
        self.weight = weight
        self.isolated_slot_penalty = isolated_slot_penalty
        self.preferred_block_min = preferred_block_sizes[0]
        self.preferred_block_max = preferred_block_sizes[1]

    def evaluate(self, tt: Timetable) -> float:
        """Penalize isolated slots and non-preferred block sizes."""
        penalty = 0
        qts = tt.qts or QuantumTimeSystem()

        # Group sessions by (course_id, course_type, day)
        course_day_quanta: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        course_type_map: dict[tuple[str, str], str] = {}

        for gene in tt.genes:
            course_key = (gene.course_id, gene.course_type)
            course_type_map[course_key] = gene.course_type

            for q in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
                day, _ = qts.quantum_to_day_and_within_day(q)
                course_day_quanta[course_key][day].append(q)

        # Analyze block sizes for each course on each day
        for course_key, course_days in course_day_quanta.items():
            course_type = course_type_map[course_key]

            # Practical = always contiguous (enforced by SessionGene)
            if course_type == "practical":
                continue

            # Theory = check fragmentation
            for quanta in course_days.values():
                quanta.sort()
                blocks = self._find_blocks(quanta)

                # Penalize isolated single slots (first one excused)
                isolated_count = sum(1 for block in blocks if len(block) == 1)
                if isolated_count > 1:
                    penalty += (isolated_count - 1) * self.isolated_slot_penalty

        return penalty

    def _find_blocks(self, quanta: list[int]) -> list[list[int]]:
        """Split quanta into contiguous blocks."""
        if not quanta:
            return []

        blocks = []
        current = [quanta[0]]

        for q in quanta[1:]:
            if q == current[-1] + 1:
                current.append(q)
            else:
                blocks.append(current)
                current = [q]

        blocks.append(current)
        return blocks


class PairedCohortPracticalAlignment:
    """Paired cohorts should have parallel practical schedules."""

    kind: Literal["soft"] = "soft"

    def __init__(self, weight: float = 1.0):
        self.name = "paired_cohort_practical_alignment"
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Measure misalignment of practical sessions for paired cohorts."""
        penalty = 0

        # Get cohort pairs from context
        cohort_pairs = tt.context.cohort_pairs or []

        # Index quanta per (course_id, course_type, group_id)
        course_group_quanta: dict[tuple[str, str, str], set[int]] = defaultdict(set)

        for gene in tt.genes:
            for group_id in gene.group_ids:
                key = (gene.course_id, gene.course_type, group_id)
                for q in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
                    course_group_quanta[key].add(q)

        # For each cohort pair, measure misalignment on shared practical courses
        for left_id, right_id in cohort_pairs:
            # Find shared practical courses
            left_courses = {
                (cid, ctype)
                for cid, ctype, gid in course_group_quanta.keys()
                if gid == left_id and ctype == "practical"
            }
            right_courses = {
                (cid, ctype)
                for cid, ctype, gid in course_group_quanta.keys()
                if gid == right_id and ctype == "practical"
            }
            shared_practicals = left_courses & right_courses

            # Measure misalignment (symmetric difference)
            for course_id, course_type in shared_practicals:
                left_quanta = course_group_quanta[(course_id, course_type, left_id)]
                right_quanta = course_group_quanta[(course_id, course_type, right_id)]
                penalty += len(left_quanta ^ right_quanta)  # Symmetric difference

        return penalty


class BreakPlacementCompliance:
    """Groups should have breaks during designated windows."""

    kind: Literal["soft"] = "soft"

    def __init__(
        self,
        weight: float = 1.0,
        break_min_quanta: int = 2,
    ):
        self.name = "break_placement_compliance"
        self.weight = weight
        self.break_min_quanta = break_min_quanta

    def evaluate(self, tt: Timetable) -> float:
        """Penalize insufficient break time during designated windows."""
        qts = tt.qts or QuantumTimeSystem()

        # Check if breaks are enforced
        if not qts.enforce_break_placement:
            return 0

        violation_count = 0
        break_windows = self._get_break_windows(qts)

        # Re-use pre-built group_daily index (or compute fallback)
        for group_id, days in _group_daily_map(tt, qts).items():
            for day_name, occupied_quanta in days.items():
                if day_name not in break_windows:
                    continue
                break_quanta = break_windows[day_name]
                free_in_window = break_quanta - occupied_quanta
                if len(free_in_window) < self.break_min_quanta:
                    violation_count += 1

        return violation_count

    def _get_break_windows(self, qts: QuantumTimeSystem) -> dict[str, set[int]]:
        """Get break window quanta per day."""
        windows = {}
        for day in qts.DAY_NAMES:
            if not qts.is_operational(day):
                continue

            try:
                break_start_q = qts.time_to_quanta(day, qts.break_window_start)
                break_end_q = qts.time_to_quanta(day, qts.break_window_end)

                day_offset = qts.day_quanta_offset[day]
                if day_offset is None:
                    continue

                within_day_start = break_start_q - day_offset
                within_day_end = break_end_q - day_offset

                windows[day] = set(range(within_day_start, within_day_end))
            except ValueError:
                continue

        return windows


# Registries


# Default constraint instances (all weights = 1.0, default params)
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


# Factory / Builder


def build_constraints(
    # Global scaling
    hard_weight: float = 1.0,
    soft_weight: float = 1.0,
    # Individual constraint weights (None = use global weight)
    student_group_exclusivity_weight: float | None = None,
    instructor_exclusivity_weight: float | None = None,
    room_exclusivity_weight: float | None = None,
    instructor_qualifications_weight: float | None = None,
    room_suitability_weight: float | None = None,
    instructor_time_availability_weight: float | None = None,
    room_time_availability_weight: float | None = None,
    course_completeness_weight: float | None = None,
    student_schedule_compactness_weight: float | None = None,
    instructor_schedule_compactness_weight: float | None = None,
    student_lunch_break_weight: float | None = None,
    session_continuity_weight: float | None = None,
    paired_cohort_practical_alignment_weight: float | None = None,
    break_placement_compliance_weight: float | None = None,
    # Constraint-specific params
    gap_penalty_per_quantum: float = 1.0,
    break_min_quanta: int = 2,
    lunch_penalty_per_missing: float = 5.0,
    isolated_slot_penalty: float = 10.0,
    preferred_block_sizes: tuple[int, int] = (2, 3),
) -> list[Constraint]:
    """
    Build constraint set with custom weights and parameters.

    Args:
        hard_weight: Global weight for all hard constraints (default 1.0)
        soft_weight: Global weight for all soft constraints (default 1.0)
        *_weight: Individual constraint weights (override global weight)
        gap_penalty_per_quantum: Penalty per gap quantum in compactness constraints
        break_min_quanta: Minimum free quanta required during lunch
        lunch_penalty_per_missing: Penalty per missing lunch quantum
        isolated_slot_penalty: Penalty for each isolated single slot
        preferred_block_sizes: (min, max) preferred session block sizes

    Returns:
        List of configured constraint instances

    Examples:
        # Default (all weights = 1.0)
        constraints = build_constraints()

        # Make hard constraints 10x more important
        constraints = build_constraints(hard_weight=10.0, soft_weight=1.0)

        # Disable soft constraints
        constraints = build_constraints(soft_weight=0.0)

        # Custom per-constraint weights
        constraints = build_constraints(
            student_group_exclusivity_weight=100.0,  # Critical!
            student_lunch_break_weight=0.5,          # Less important
        )

        # Custom magic values
        constraints = build_constraints(
            isolated_slot_penalty=50.0,  # Heavy penalty
            break_min_quanta=4,          # Require 4-quantum lunch
        )
    """
    return [
        # Hard constraints
        StudentGroupExclusivity(weight=student_group_exclusivity_weight or hard_weight),
        InstructorExclusivity(weight=instructor_exclusivity_weight or hard_weight),
        RoomExclusivity(weight=room_exclusivity_weight or hard_weight),
        InstructorQualifications(
            weight=instructor_qualifications_weight or hard_weight
        ),
        RoomSuitability(weight=room_suitability_weight or hard_weight),
        InstructorTimeAvailability(
            weight=instructor_time_availability_weight or hard_weight
        ),
        RoomTimeAvailability(weight=room_time_availability_weight or hard_weight),
        CourseCompleteness(weight=course_completeness_weight or hard_weight),
        # Soft constraints
        StudentScheduleCompactness(
            weight=student_schedule_compactness_weight or soft_weight,
            gap_penalty_per_quantum=gap_penalty_per_quantum,
        ),
        InstructorScheduleCompactness(
            weight=instructor_schedule_compactness_weight or soft_weight,
            gap_penalty_per_quantum=gap_penalty_per_quantum,
        ),
        StudentLunchBreak(
            weight=student_lunch_break_weight or soft_weight,
            break_min_quanta=break_min_quanta,
            penalty_per_missing_quantum=lunch_penalty_per_missing,
        ),
        SessionContinuity(
            weight=session_continuity_weight or soft_weight,
            isolated_slot_penalty=isolated_slot_penalty,
            preferred_block_sizes=preferred_block_sizes,
        ),
        PairedCohortPracticalAlignment(
            weight=paired_cohort_practical_alignment_weight or soft_weight
        ),
        BreakPlacementCompliance(
            weight=break_placement_compliance_weight or soft_weight,
            break_min_quanta=break_min_quanta,
        ),
    ]
