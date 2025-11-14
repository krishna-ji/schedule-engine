from typing import Dict, List
from src.entities.course import Course
from src.entities.decoded_session import CourseSession
from collections import defaultdict
from src.config import get_config
from src.encoder.quantum_time_system import QuantumTimeSystem

# Time system singleton
_QTS = QuantumTimeSystem()


def student_group_exclusivity(sessions: List[CourseSession]) -> int:
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


def instructor_exclusivity(sessions: List[CourseSession]) -> int:
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
    sessions: List[CourseSession], course_map: Dict[tuple, Course]
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
        print(
            f"⚠ WARNING: {len(missing_courses)} course(s) missing from course_map: "
            f"{list(missing_courses)[:3]}{'...' if len(missing_courses) > 3 else ''}"
        )
    if empty_qualifications:
        print(
            f"⚠ WARNING: {len(empty_qualifications)} course(s) have no qualified instructors: "
            f"{list(empty_qualifications)[:3]}{'...' if len(empty_qualifications) > 3 else ''}"
        )

    return violations


def room_suitability(sessions: List[CourseSession]) -> int:
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
        if not _room_type_matches(required_str, room_str):
            violations += 1

    return violations


def _room_type_matches(required: str, room_type: str) -> bool:
    """
    Check if room type satisfies requirement with flexible compatibility.

    Args:
        required: Required room type (e.g., "lecture", "practical")
        room_type: Actual room type (e.g., "lecture", "practical")

    Returns:
        True if compatible, False otherwise
    """
    # Exact match
    if required == room_type:
        return True

    # Lecture/theory courses: Accept lecture, classroom, auditorium
    if required in ["lecture", "classroom", "theory"]:
        if room_type in ["lecture", "classroom", "auditorium", "seminar", "tutorial"]:
            return True

    # Practical/lab courses: Accept practical, lab variants
    if required in ["practical", "lab", "laboratory"]:
        if room_type in [
            "practical",
            "lab",
            "laboratory",
            "computer_lab",
            "science_lab",
        ]:
            return True

    return False


def instructor_time_availability(sessions: List[CourseSession]) -> int:
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
        if instructor.is_full_time:
            continue

        # Part-time: check if session quanta are within available_quanta
        for q in session.session_quanta:
            if q not in instructor.available_quanta:
                violations += 1
                break  # Only count one violation per session

    return violations


def room_time_availability(sessions: List[CourseSession]) -> int:
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
        for q in session.session_quanta:
            if q not in room.available_quanta:
                violations += 1
                break  # Only count one violation per session

    return violations


def course_completeness(
    sessions: List[CourseSession], course_map: Dict[tuple, Course]
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
    course_group_quanta = defaultdict(int)

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


def room_exclusivity(sessions: List[CourseSession]) -> int:
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
        room_id = session.room.room_id
        for q in session.session_quanta:
            key = (room_id, q)
            if key in room_time_map:
                conflicts += 1
            else:
                room_time_map[key] = session.course_id

    return conflicts


# ---------------------------
# Hard Constraint Registry
# ---------------------------
def get_all_hard_constraints():
    """
    Returns a dictionary of all available hard constraint functions.

    Hard Constraints (7 total):
    1. student_group_exclusivity - Groups can only be in one place at a time
    2. instructor_exclusivity - Instructors can only teach one session at a time
    3. instructor_qualifications - Instructors must be qualified for their courses
    4. instructor_time_availability - Instructors teach within their available hours
    5. room_suitability - Rooms must match course requirements
    6. room_exclusivity - Rooms can only host one session at a time
    7. room_time_availability - Rooms used within their available hours
    8. course_completeness - Courses have correct number of sessions per group

    Returns:
        Dict[str, callable]: Mapping of constraint names to their functions.
    """
    return {
        "student_group_exclusivity": student_group_exclusivity,
        "instructor_exclusivity": instructor_exclusivity,
        "instructor_qualifications": instructor_qualifications,
        "instructor_time_availability": instructor_time_availability,
        "room_suitability": room_suitability,
        "room_exclusivity": room_exclusivity,
        "room_time_availability": room_time_availability,
        "course_completeness": course_completeness,
    }


def get_enabled_hard_constraints():
    """
    Returns only the enabled hard constraints based on config.

    Returns:
        Dict[str, dict]: Mapping of enabled constraint names to their config (function, weight).
    """

    all_constraints = get_all_hard_constraints()
    enabled = {}

    cfg = get_config().hard_constraints
    for name, func in all_constraints.items():
        constraint_cfg = getattr(cfg, name, None)
        if constraint_cfg and constraint_cfg.enabled:
            enabled[name] = {
                "function": func,
                "weight": constraint_cfg.weight,
            }

    return enabled
