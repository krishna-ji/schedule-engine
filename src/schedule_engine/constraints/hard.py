from __future__ import annotations

import logging
from collections import defaultdict

from schedule_engine.domain.course import Course
from schedule_engine.domain.session import CourseSession
from schedule_engine.io.time_system import QuantumTimeSystem

# Time system singleton
_QTS = QuantumTimeSystem()
_WARNED_MISSING_COURSES: set[tuple[str, str]] = set()
_WARNED_EMPTY_QUALIFICATIONS: set[tuple[str, str]] = set()
logger = logging.getLogger(__name__)


def student_group_exclusivity(sessions: list[CourseSession]) -> int:
    """
    Ensures each student group can only be in one session at a time.

    Counts violations where a group is scheduled in multiple sessions simultaneously.

    Args:
        sessions: List of all decoded sessions.

    Returns:
        Number of group double-booking conflicts.
    """
    conflict_count = 0
    group_time_map = {}  # Maps (group_id, time_quanta) to count of sessions

    for session in sessions:
        for gid in session.group_ids:
            for q in session.session_quanta:
                key = (gid, q)
                if key in group_time_map:
                    conflict_count += 1
                else:
                    # The 'else' block is executed when the (group_id, time_quanta) key is not already in the group_time_map.
                    # It adds this key to the map and associates it with the session's course_id.
                    # This helps track which group is scheduled at which time, so future overlaps can be detected.
                    group_time_map[key] = session.course_id  #

    return conflict_count


def instructor_exclusivity(sessions: list[CourseSession]) -> int:
    """
    Ensures each instructor can only teach one session at a time.

    Counts violations where an instructor is scheduled in multiple sessions simultaneously.
    """
    conflicts = 0
    instructor_time_map = {}

    for session in sessions:
        iid = session.instructor_id
        for q in session.session_quanta:
            key = (iid, q)
            if key in instructor_time_map:
                conflicts += 1
            else:
                instructor_time_map[key] = session.course_id

    return conflicts


def instructor_qualifications(
    sessions: list[CourseSession], course_map: dict[tuple, Course]
) -> int:
    """
    Ensures instructors are qualified to teach their assigned courses.

    Counts violations where an instructor is assigned to a course they are not qualified for.
    Treats missing course definitions and empty qualification lists as violations.

    Args:
        sessions: List of decoded course sessions
        course_map: Mapping from (course_id, course_type) to Course entity

    Returns:
        Number of unqualified instructor assignments
    """
    violations = 0
    missing_courses = set()
    empty_qualifications = set()

    for session in sessions:
        course_key = (session.course_id, session.course_type)

        # Missing course definition = violation (stricter policy)
        if course_key not in course_map:
            violations += 1
            missing_courses.add(course_key)
            continue

        course = course_map[course_key]
        qualified = getattr(course, "qualified_instructor_ids", None)

        # Empty/None qualification list = violation (no one qualified)
        if not qualified:
            violations += 1
            empty_qualifications.add(course_key)
            continue

        # Instructor not in qualified list = violation
        if session.instructor_id not in qualified:
            violations += 1

    # Warn about data issues (helps debugging)
    if missing_courses:
        unseen = missing_courses - _WARNED_MISSING_COURSES
        if unseen:
            logger.warning(
                "Missing course definitions for %d course(s): %s",
                len(unseen),
                list(unseen)[:3],
            )
            _WARNED_MISSING_COURSES.update(unseen)
    if empty_qualifications:
        unseen = empty_qualifications - _WARNED_EMPTY_QUALIFICATIONS
        if unseen:
            logger.warning(
                "Courses without qualified instructors: %d course(s): %s",
                len(unseen),
                list(unseen)[:3],
            )
            _WARNED_EMPTY_QUALIFICATIONS.update(unseen)

    return violations


def room_suitability(sessions: list[CourseSession]) -> int:
    """
    Ensures rooms are suitable for the type of course being taught.

    Counts violations where a course is scheduled in an incompatible room type.
    Allows flexible compatibility (e.g., lectures can use auditoriums).

    Args:
        sessions: List of decoded course sessions

    Returns:
        Number of room type incompatibilities
    """
    violations = 0

    for session in sessions:
        # Both should be simple strings now (not lists)
        required = getattr(session, "required_room_features", "lecture")
        room_type = getattr(session.room, "room_features", "lecture")

        # Normalize to lowercase strings
        required_str = (
            (required if isinstance(required, str) else str(required)).lower().strip()
        )
        room_str = (
            (room_type if isinstance(room_type, str) else str(room_type))
            .lower()
            .strip()
        )

        # Check if room type matches (with flexibility)
        from schedule_engine.utils.room_compatibility import is_room_type_compatible

        if not is_room_type_compatible(required_str, room_str):
            violations += 1

    return violations


def instructor_time_availability(sessions: list[CourseSession]) -> int:
    """
    Ensures instructors only teach during their available time slots.

    For part-time instructors or those with scheduling restrictions,
    sessions must fit within their specified availability windows.
    Full-time instructors are available during all operating hours.

    Args:
        sessions: List of all decoded sessions.

    Returns:
        Number of violations where instructors are scheduled outside their availability.
    """
    violations = 0

    for session in sessions:
        instructor = session.instructor

        # Full-time instructors are always available during operating hours
        if instructor and instructor.is_full_time:
            continue

        # Part-time: check if session quanta are within available_quanta
        if instructor:
            for q in session.session_quanta:
                if q not in instructor.available_quanta:
                    violations += 1
                    break  # Only count one violation per session

    return violations


def room_time_availability(sessions: list[CourseSession]) -> int:
    """
    Ensures rooms are only used during their available time slots.

    Some rooms may have restricted availability (e.g., labs under maintenance,
    rooms reserved for other purposes). If no availability is specified,
    the room is assumed available during all operating hours.

    Args:
        sessions: List of all decoded sessions.

    Returns:
        Number of violations where rooms are used outside their availability.
    """
    violations = 0

    for session in sessions:
        room = session.room

        # Check if any session quantum is outside room's available quanta
        if room:
            for q in session.session_quanta:
                if q not in room.available_quanta:
                    violations += 1
                    break  # Only count one violation per session

    return violations


def course_completeness(
    sessions: list[CourseSession], course_map: dict[tuple, Course]
) -> int:
    """
    Ensures each course is scheduled for exactly the required number of sessions.

    Verifies that each (course, group) combination has the correct number of quanta per week.
    Courses are taught per group - theory may use parent groups, practicals use subgroups.

    Args:
        sessions: List of decoded course sessions
        course_map: Mapping from (course_id, course_type) to Course entity

    Returns:
        Number of (course, group) combinations that are under- or over-scheduled.

    Example:
        If BAE2 is enrolled in ENME 151 (5 quanta/week),
        we must have exactly 5 quanta for (ENME 151, BAE2) combination.
    """
    # Count quanta per (course_code, course_type, group_id) combination
    # Use (course_code, course_type) to distinguish theory from practical
    course_group_quanta: dict[tuple[tuple[str, str], str], int] = defaultdict(int)

    for session in sessions:
        course_code = session.course_id  # This is just the course code string
        course_type = session.course_type  # This is "theory" or "practical"

        # Each session can have multiple groups (multi-group sessions)
        # Count quanta for each group separately
        for group_id in session.group_ids:
            # Key must match course_map key structure: (course_code, course_type)
            key = ((course_code, course_type), group_id)
            course_group_quanta[key] += len(session.session_quanta)

    violations = 0

    # Check each course's enrolled groups
    # course_key is (course_code, course_type) tuple
    for course_key, course in course_map.items():
        expected_quanta = course.quanta_per_week
        enrolled_groups = course.enrolled_group_ids

        # For each group enrolled in this course
        for group_id in enrolled_groups:
            # Use same key structure as counting above
            key = (course_key, group_id)
            actual_quanta = course_group_quanta.get(key, 0)

            # Check if scheduled correctly for this (course, group) pair
            if actual_quanta != expected_quanta:
                violations += 1

    return violations


def room_exclusivity(sessions: list[CourseSession]) -> int:
    """
    Ensures each room can only host one session at a time.

    Counts violations where a room is scheduled for multiple sessions simultaneously.
    Rooms are physical resources that cannot be shared.

    Args:
        sessions: List of all decoded sessions.

    Returns:
        Number of room double-booking conflicts.
    """
    conflicts = 0
    room_time_map = {}  # Maps (room_id, time_quanta) to course_id

    for session in sessions:
        if not session.room:
            continue
        room_id = session.room.room_id
        for q in session.session_quanta:
            key = (room_id, q)
            if key in room_time_map:
                conflicts += 1
            else:
                room_time_map[key] = session.course_id

    return conflicts
