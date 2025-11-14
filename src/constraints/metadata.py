"""
Single source of truth for constraint metadata.

This module defines which constraints need special parameters (like course_map)
and provides centralized constraint information to avoid duplication.
"""

from typing import Set, Dict, Any

# ============================================================================
# CONSTRAINT PARAMETER REQUIREMENTS
# ============================================================================

# Hard constraints that require the 'courses' parameter (course_map)
HARD_CONSTRAINTS_NEEDING_COURSES: Set[str] = {
    "instructor_qualifications",
    "course_completeness",
}

# All other hard constraints only need 'sessions' parameter
# (student_group_exclusivity, instructor_exclusivity, instructor_time_availability,
#  room_suitability, room_exclusivity, room_time_availability)


# ============================================================================
# CONSTRAINT DESCRIPTIONS
# ============================================================================

HARD_CONSTRAINT_DESCRIPTIONS: Dict[str, str] = {
    "student_group_exclusivity": "Ensures each student group can only be in one session at a time",
    "instructor_exclusivity": "Ensures each instructor can only teach one session at a time",
    "instructor_qualifications": "Ensures instructors are qualified to teach their assigned courses",
    "instructor_time_availability": "Ensures instructors are only scheduled during their available time windows",
    "room_suitability": "Ensures rooms are suitable for the type of course being taught",
    "room_exclusivity": "Ensures rooms are not double-booked",
    "room_time_availability": "Ensures rooms are only used during their available time windows",
    "course_completeness": "Ensures courses have the correct number of sessions per group",
}

SOFT_CONSTRAINT_DESCRIPTIONS: Dict[str, str] = {
    "student_schedule_compactness": "Minimizes gaps in student schedules",
    "instructor_schedule_compactness": "Minimizes gaps in instructor schedules",
    "student_lunch_break": "Encourages students to have midday break time",
    "session_continuity": "Encourages sessions to be in appropriate continuous blocks",
}


# ============================================================================
# VIOLATION TYPE MAPPINGS (for violation_detector.py)
# ============================================================================

# Map old violation type names to new constraint names for consistency
VIOLATION_TYPE_TO_CONSTRAINT_NAME: Dict[str, str] = {
    "instructor_qualifications": "instructor_qualifications",
    "room_suitability": "room_suitability",
    "instructor_availability": "instructor_time_availability",
    "room_availability": "room_time_availability",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def constraint_needs_courses(constraint_name: str) -> bool:
    """
    Check if a hard constraint needs the 'courses' parameter.

    Args:
        constraint_name: Name of the constraint

    Returns:
        True if constraint needs courses parameter, False otherwise
    """
    return constraint_name in HARD_CONSTRAINTS_NEEDING_COURSES


def get_constraint_description(constraint_name: str) -> str:
    """
    Get human-readable description of a constraint.

    Args:
        constraint_name: Name of the constraint

    Returns:
        Description string, or "Unknown constraint" if not found
    """
    return (
        HARD_CONSTRAINT_DESCRIPTIONS.get(constraint_name)
        or SOFT_CONSTRAINT_DESCRIPTIONS.get(constraint_name)
        or "Unknown constraint"
    )


def get_all_constraint_names() -> Dict[str, Any]:
    """
    Get all constraint names organized by type.

    Returns:
        Dict with 'hard' and 'soft' keys containing lists of constraint names
    """
    return {
        "hard": list(HARD_CONSTRAINT_DESCRIPTIONS.keys()),
        "soft": list(SOFT_CONSTRAINT_DESCRIPTIONS.keys()),
    }
