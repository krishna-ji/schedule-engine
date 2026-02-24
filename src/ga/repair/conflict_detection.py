"""
Conflict detection for LNS repair.

This module identifies sessions involved in hard constraint violations
and provides detailed violation information for targeted repair.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from src.constraints import HARD_CONSTRAINT_CLASSES
from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.domain.session import CourseSession
from src.io.decoder import decode_individual


class ViolationInfo:
    """
    Detailed information about constraint violations.

    Attributes:
        constraint_name: Name of the violated constraint
        violation_count: Number of violations for this constraint
        affected_sessions: Indices of sessions involved in violations
        conflict_details: Additional details about the conflict (optional)
    """

    def __init__(
        self,
        constraint_name: str,
        violation_count: int,
        affected_sessions: set[int],
        conflict_details: dict[str, Any] | None = None,
    ):
        self.constraint_name = constraint_name
        self.violation_count = violation_count
        self.affected_sessions = affected_sessions
        self.conflict_details = conflict_details or {}


def find_hard_conflict_sessions(
    individual: list[SessionGene],
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
) -> tuple[list[int], list[ViolationInfo]]:
    """
    Identifies sessions involved in hard constraint violations.

    This function evaluates all enabled hard constraints and tracks which
    sessions are involved in violations. Used by LNS to determine which
    sessions to extract for repair.

    Args:
        individual: GA individual (list of SessionGene)
        courses: Course dictionary
        instructors: Instructor dictionary
        groups: Group dictionary
        rooms: Room dictionary

    Returns:
        Tuple of:
            - List of session indices involved in conflicts (sorted)
            - List of ViolationInfo objects with detailed violation data
    """
    # Decode individual to CourseSession objects
    sessions = decode_individual(individual, courses, instructors, groups, rooms)

    # Track which sessions are involved in violations
    conflicted_session_indices: set[int] = set()
    violations: list[ViolationInfo] = []

    # Get enabled hard constraints
    # Evaluate each hard constraint and identify conflicted sessions
    for constraint in HARD_CONSTRAINT_CLASSES:
        constraint_name = constraint.name

        # Determine affected sessions using tracking functions
        affected_indices, violation_count, details = _evaluate_constraint_with_tracking(
            constraint_name, None, sessions, courses
        )

        if violation_count > 0:
            conflicted_session_indices.update(affected_indices)
            violations.append(
                ViolationInfo(
                    constraint_name=constraint_name,
                    violation_count=violation_count,
                    affected_sessions=affected_indices,
                    conflict_details=details,
                )
            )

    # Return sorted list of indices
    return sorted(conflicted_session_indices), violations


def _evaluate_constraint_with_tracking(
    constraint_name: str,
    constraint_func: Callable[..., int] | None,
    sessions: list[CourseSession],
    courses: dict[tuple, Course] | None = None,
) -> tuple[set[int], int, dict[str, Any]]:
    """
    Evaluates a constraint and tracks which sessions are involved.

    Args:
        constraint_name: Name of the constraint
        constraint_func: Constraint evaluation function
        sessions: List of decoded sessions
        courses: Course dictionary (optional, for constraints that need it)

    Returns:
        Tuple of:
            - Set of session indices involved in violations
            - Total violation count
            - Dict with additional details
    """
    # Dispatch to specific tracking function based on constraint type
    if constraint_name == "CTE":  # Cohort Time Exclusivity
        return _track_student_group_conflicts(sessions)
    if constraint_name == "FTE":  # Faculty Time Exclusivity
        return _track_instructor_conflicts(sessions)
    if constraint_name == "SRE":  # Space Resource Exclusivity
        return _track_room_conflicts(sessions)
    if constraint_name == "FPC":  # Faculty-Program Compliance
        # Courses is required for this constraint
        if courses is None:
            return set(), 0, {}
        return _track_qualification_violations(sessions, courses)
    if constraint_name in ("room_capacity", "FFC"):  # Facility-Format Compliance
        return _track_room_capacity_violations(sessions)
    if constraint_name == "room_features":
        return _track_room_feature_violations(sessions)
    if constraint_name in (
        "instructor_availability",
        "FCA",
    ):  # Faculty Chronometric Availability
        return _track_instructor_availability_violations(sessions)
    if constraint_name in ("group_availability", "room_time_availability"):
        return _track_group_availability_violations(sessions)
    if constraint_name == "CQF":  # Curriculum Quantum Fulfillment
        # Course completeness is a global constraint, not per-session trackable
        return set(), 0, {}
    # Generic fallback: skip if no constraint function provided
    if constraint_func is None:
        return set(), 0, {}
    if courses:
        violation_count = constraint_func(sessions, courses)
    else:
        violation_count = constraint_func(sessions)

    # If violations exist but we can't track specifically, mark all sessions
    if violation_count > 0:
        return set(range(len(sessions))), violation_count, {}
    return set(), 0, {}


def _track_student_group_conflicts(
    sessions: list[CourseSession],
) -> tuple[set[int], int, dict[str, Any]]:
    """Track student group exclusivity violations."""
    conflict_count = 0
    conflicted_indices = set()
    group_time_map: dict[tuple[str, int], list[int]] = defaultdict(
        list
    )  # Maps (group_id, quanta) to list of session indices

    for idx, session in enumerate(sessions):
        for gid in session.group_ids:
            for q in session.session_quanta:
                key = (gid, q)
                if key in group_time_map:
                    # Conflict detected
                    conflict_count += 1
                    conflicted_indices.add(idx)
                    conflicted_indices.update(group_time_map[key])
                group_time_map[key].append(idx)

    return (
        conflicted_indices,
        conflict_count,
        {"conflict_type": "student_group_overlap"},
    )


def _track_instructor_conflicts(
    sessions: list[CourseSession],
) -> tuple[set[int], int, dict[str, Any]]:
    """Track instructor exclusivity violations."""
    conflict_count = 0
    conflicted_indices = set()
    instructor_time_map: dict[tuple[str, int], list[int]] = defaultdict(list)

    for idx, session in enumerate(sessions):
        iid = session.instructor_id
        for q in session.session_quanta:
            key = (iid, q)
            if key in instructor_time_map:
                conflict_count += 1
                conflicted_indices.add(idx)
                conflicted_indices.update(instructor_time_map[key])
            instructor_time_map[key].append(idx)

    return conflicted_indices, conflict_count, {"conflict_type": "instructor_overlap"}


def _track_room_conflicts(
    sessions: list[CourseSession],
) -> tuple[set[int], int, dict[str, Any]]:
    """Track room exclusivity violations."""
    conflict_count = 0
    conflicted_indices = set()
    room_time_map: dict[tuple[str, int], list[int]] = defaultdict(list)

    for idx, session in enumerate(sessions):
        rid = session.room_id
        for q in session.session_quanta:
            key = (rid, q)
            if key in room_time_map:
                conflict_count += 1
                conflicted_indices.add(idx)
                conflicted_indices.update(room_time_map[key])
            room_time_map[key].append(idx)

    return conflicted_indices, conflict_count, {"conflict_type": "room_overlap"}


def _track_qualification_violations(
    sessions: list[CourseSession],
    courses: dict[tuple, Course],
) -> tuple[set[int], int, dict[str, Any]]:
    """Track instructor qualification violations."""
    violations = 0
    conflicted_indices = set()

    for idx, session in enumerate(sessions):
        course_key = (session.course_id, session.course_type)
        course = courses.get(course_key)

        if not course or not course.qualified_instructor_ids:
            violations += 1
            conflicted_indices.add(idx)
            continue

        if session.instructor_id not in course.qualified_instructor_ids:
            violations += 1
            conflicted_indices.add(idx)

    return conflicted_indices, violations, {"conflict_type": "unqualified_instructor"}


def _track_room_capacity_violations(
    sessions: list[CourseSession],
) -> tuple[set[int], int, dict[str, Any]]:
    """Track room capacity violations."""
    violations = 0
    conflicted_indices = set()

    for idx, session in enumerate(sessions):
        if (
            session.room
            and session.group
            and session.group.student_count > session.room.capacity
        ):
            violations += 1
            conflicted_indices.add(idx)

    return conflicted_indices, violations, {"conflict_type": "room_capacity_exceeded"}


def _track_room_feature_violations(
    sessions: list[CourseSession],
) -> tuple[set[int], int, dict[str, Any]]:
    """Track room feature violations."""
    from src.utils.room_compatibility import is_room_suitable_for_course

    violations = 0
    conflicted_indices = set()

    for idx, session in enumerate(sessions):
        if session.room and session.required_room_features:
            required_features = session.required_room_features.strip().lower()
            room_features = session.room.room_features.strip().lower()
            course_lab_feats = getattr(session, "specific_lab_features", None)
            room_spec_feats = getattr(session.room, "specific_features", None)
            if required_features and not is_room_suitable_for_course(
                required_features, room_features, course_lab_feats, room_spec_feats
            ):
                violations += 1
                conflicted_indices.add(idx)

    return conflicted_indices, violations, {"conflict_type": "room_feature_mismatch"}


def _track_instructor_availability_violations(
    sessions: list[CourseSession],
) -> tuple[set[int], int, dict[str, Any]]:
    """Track instructor availability violations."""
    violations = 0
    conflicted_indices = set()

    for idx, session in enumerate(sessions):
        # Full-time instructors are always available during operating hours
        if session.instructor and not session.instructor.is_full_time:
            for q in session.session_quanta:
                if q not in session.instructor.available_quanta:
                    violations += 1
                    conflicted_indices.add(idx)
                    break  # Count once per session

    return conflicted_indices, violations, {"conflict_type": "instructor_unavailable"}


def _track_group_availability_violations(
    sessions: list[CourseSession],
) -> tuple[set[int], int, dict[str, Any]]:
    """Track group availability violations."""
    violations = 0
    conflicted_indices = set()

    for idx, session in enumerate(sessions):
        if session.group:
            for q in session.session_quanta:
                if q not in session.group.available_quanta:
                    violations += 1
                    conflicted_indices.add(idx)
                    break  # Count once per session

    return conflicted_indices, violations, {"conflict_type": "group_unavailable"}


def select_worst_conflicts(
    conflicted_indices: list[int],
    violations: list[ViolationInfo],
    max_sessions: int = 20,
) -> list[int]:
    """
    Select up to max_sessions most problematic sessions for repair.

    Prioritizes sessions involved in multiple violations or high-weight
    constraint violations.

    Args:
        conflicted_indices: List of all session indices with conflicts
        violations: List of ViolationInfo objects
        max_sessions: Maximum number of sessions to select

    Returns:
        List of selected session indices (at most max_sessions)
    """
    if len(conflicted_indices) <= max_sessions:
        return conflicted_indices

    # Count how many violations each session is involved in
    session_violation_count: dict[int, int] = defaultdict(int)
    for violation in violations:
        for idx in violation.affected_sessions:
            session_violation_count[idx] += violation.violation_count

    # Sort by violation count (descending) and take top max_sessions
    sorted_indices = sorted(
        conflicted_indices,
        key=lambda idx: session_violation_count[idx],
        reverse=True,
    )

    return sorted_indices[:max_sessions]
