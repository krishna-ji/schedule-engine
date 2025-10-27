from typing import Dict, List
from src.entities.course import Course
from src.entities.decoded_session import CourseSession
from collections import defaultdict
from src.config import get_config
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.utils.time_helpers import quantum_to_day_and_within_day

# Time system singleton
_QTS = QuantumTimeSystem()


def no_group_overlap(sessions: List[CourseSession]) -> int:
    """
    Counts how many times a group is assigned to multiple sessions at the same time.

    Args:
        sessions (List[CourseSession]): List of all decoded sessions.

    Returns:
        int: Total number of group-time conflicts.
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


def no_instructor_conflict(sessions: List[CourseSession]) -> int:
    """
    Counts how many times an instructor is assigned to multiple sessions at the same time.
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


def instructor_not_qualified(
    sessions: List[CourseSession], course_map: Dict[tuple, Course]
) -> int:
    """
    Counts sessions assigned to unqualified instructors.

    Treats missing course definitions and empty qualification lists as violations
    (stricter than silently skipping).

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


def room_type_mismatch(sessions: List[CourseSession]) -> int:
    """
    Counts how many sessions are scheduled in rooms that don't match required type.

    Simple string matching with flexible compatibility rules:
    - Exact match: required == room_type
    - Compatible: "lecture" courses can use lecture/classroom/auditorium rooms
    - Compatible: "practical" courses can use practical/lab rooms

    Args:
        sessions: List of decoded course sessions

    Returns:
        Number of room type mismatches
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


def availability_violations(sessions: List[CourseSession]) -> int:
    """
    Counts how many sessions are scheduled during unavailable time slots for
    the group, instructor, or room.

    For multi-group sessions, checks availability for all assigned groups.
    """
    violations = 0

    for session in sessions:
        for q in session.session_quanta:
            # Check if quantum is unavailable for instructor or room
            if (
                q not in session.instructor.available_quanta
                or q not in session.room.available_quanta
            ):
                violations += 1
                break  # Only count one violation per session

        # For multi-group sessions, check all groups' availability
        # If primary group exists, use it; otherwise skip group availability check
        if session.group:
            for q in session.session_quanta:
                if q not in session.group.available_quanta:
                    violations += 1
                    break  # Only count one violation per session

    return violations


def incomplete_or_extra_sessions(
    sessions: List[CourseSession], course_map: Dict[tuple, Course]
) -> int:
    """
    Verifies that each course is scheduled for exactly the required number of quanta
    PER GROUP that is enrolled in that course.

    New Architecture:
    - Courses are taught per group (not globally)
    - Theory sessions may use parent groups (whole class)
    - Practical sessions may use subgroups
    - Must check: each (course, group) combination has correct quanta

    IMPORTANT: course_map keys are tuples (course_code, course_type), but
    session.course_id is just the course_code string. Must extract course_code
    from tuple keys for proper comparison.

    Returns:
        Number of (course, group) combinations that are under- or over-scheduled.

    Example:
        If BAE2 is enrolled in ENME 151 (5 quanta/week),
        we should have exactly 5 quanta for (ENME 151, BAE2) combination.
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


def session_block_clustering_penalty(sessions: List[CourseSession]) -> int:
    """
    Penalizes course sessions that are fragmented into undesirable block sizes.

    Theory courses:
    - Penalty of 1 for blocks greater than 3
    - Penalty of 2 for isolated blocks (length 1), but first isolated slot is excused
    - Preferred block sizes: 2-3 consecutive quanta

    Practical courses:
    - All sessions must be in a single coalesced block (no splits)
    - Heavy penalty for fragmentation

    Example (Theory):
    - 6 quanta as [3,3] → 0 penalty (ideal)
    - 6 quanta as [2,2,2] → 0 penalty (acceptable)
    - 6 quanta as [1,2,3] → 0 penalty (first isolated slot excused)
    - 6 quanta as [1,1,4] → 2 + 1 = 3 penalty (second isolated slot + oversized block)
    - 6 quanta as [6] → 3 penalty (oversized by 3)

    Example (Practical):
    - 3 quanta as [3] → 0 penalty (ideal - single block)
    - 3 quanta as [2,1] → 20 penalty (fragmented practical)

    Args:
        sessions: List of course sessions to evaluate.

    Returns:
        Total penalty for non-preferred block sizes.
    """
    cfg = get_config().time

    penalty = 0

    # Group sessions by (course_id, course_type, day) to find blocks
    course_day_quanta = defaultdict(lambda: defaultdict(list))
    course_type_map = {}  # Track course types

    for session in sessions:
        # Use course_id + course_type as unique identifier
        course_key = (session.course_id, session.course_type)
        course_type_map[course_key] = session.course_type

        for q in session.session_quanta:
            day, within_day = quantum_to_day_and_within_day(q, _QTS)
            course_day_quanta[course_key][day].append(within_day)

    # Analyze block sizes for each course on each day
    for course_key, course_days in course_day_quanta.items():
        course_type = course_type_map[course_key]

        for day_quanta in course_days.values():
            # Sort quanta to identify consecutive blocks
            sorted_quanta = sorted(day_quanta)

            # Find consecutive blocks
            blocks = []
            if sorted_quanta:
                current_block = [sorted_quanta[0]]

                for i in range(1, len(sorted_quanta)):
                    if sorted_quanta[i] == sorted_quanta[i - 1] + 1:
                        # Consecutive - add to current block
                        current_block.append(sorted_quanta[i])
                    else:
                        # Gap - start new block
                        blocks.append(len(current_block))
                        current_block = [sorted_quanta[i]]

                # Don't forget the last block
                blocks.append(len(current_block))

            # Apply penalties based on course type
            if course_type.lower() == "practical":
                # Practical courses: must be in a single block
                if len(blocks) > 1:
                    # Heavy penalty for fragmentation
                    penalty += cfg.practical_fragmentation_penalty * (len(blocks) - 1)
            else:
                # Theory courses: apply refined penalty logic
                isolated_count = 0

                for block_size in blocks:
                    if block_size == 1:
                        # Isolated single quantum
                        isolated_count += 1
                        if isolated_count > cfg.theory_max_excused_isolated:
                            # Excused slots exceeded, penalize subsequent ones
                            penalty += cfg.theory_isolated_penalty
                    elif block_size > cfg.preferred_block_size_max:
                        # Oversized block - penalty per quantum beyond max
                        excess = block_size - cfg.preferred_block_size_max
                        penalty += excess * cfg.theory_oversized_penalty_per_quantum
                    # Block sizes within preferred range have no penalty

    return penalty


# ---------------------------
# Hard Constraint Registry
# ---------------------------
def get_all_hard_constraints():
    """
    Returns a dictionary of all available hard constraint functions.

    Returns:
        Dict[str, callable]: Mapping of constraint names to their functions.
    """
    return {
        "no_group_overlap": no_group_overlap,
        "no_instructor_conflict": no_instructor_conflict,
        "instructor_not_qualified": instructor_not_qualified,
        "room_type_mismatch": room_type_mismatch,
        "availability_violations": availability_violations,
        "incomplete_or_extra_sessions": incomplete_or_extra_sessions,
        "session_block_clustering_penalty": session_block_clustering_penalty,
    }


def get_enabled_hard_constraints():
    """
    Returns only the enabled hard constraints based on config.

    Returns:
        Dict[str, dict]: Mapping of enabled constraint names to their config (function, weight).
    """
    from src.config import get_config

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
