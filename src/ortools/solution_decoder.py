"""
CP-SAT Solution Decoder

Converts CP-SAT solver solutions into CourseSession objects.

Decoding Process:
    CP-SAT Solution (variable assignments)
        ↓
    Extract: start_quantum, instructor_idx, room_idx for each session
        ↓
    Convert indices to entity IDs
        ↓
    Create CourseSession objects with resolved entities
"""

from typing import Dict, List, Tuple
from ortools.sat.python import cp_model

from src.core.types import SchedulingContext
from src.entities.decoded_session import CourseSession
from src.ortools.variable_factory import VariableFactory


def decode_cp_solution(
    solution: cp_model.CpSolver,
    session_vars: Dict[Tuple[str, str, str, int], Dict],
    var_factory: VariableFactory,
    context: SchedulingContext,
) -> List[CourseSession]:
    """
    Decode a CP-SAT solution into list of CourseSession objects.

    Args:
        solution: CP-SAT solver with solution
        session_vars: Dictionary of session variables
        var_factory: VariableFactory with ID mappings
        context: SchedulingContext with entity references

    Returns:
        List of CourseSession objects
    """
    sessions = []

    for session_key, vars_dict in session_vars.items():
        course_code, course_type, group_id, session_idx = session_key

        # Extract variable values
        start_quantum = solution.Value(vars_dict["start_quantum"])
        instructor_idx = solution.Value(vars_dict["instructor"])
        room_idx = solution.Value(vars_dict["room"])

        # Convert indices to IDs
        instructor_id = var_factory.idx_to_instructor[instructor_idx]
        room_id = var_factory.idx_to_room[room_idx]

        # Get course information
        course_key = (course_code, course_type)
        course = context.courses[course_key]

        # Create CourseSession
        session = CourseSession(
            course_id=course_code,
            instructor_id=instructor_id,
            group_ids=[group_id],  # Single group per session in this model
            room_id=room_id,
            session_quanta=[start_quantum],  # Single quantum per session variable
            required_room_features=course.required_room_features,
            course_type=course_type,
            instructor=context.instructors[instructor_id],
            group=context.groups[group_id],
            room=context.rooms[room_id],
        )

        sessions.append(session)

    return sessions


def merge_consecutive_sessions(sessions: List[CourseSession]) -> List[CourseSession]:
    """
    Merge consecutive single-quantum sessions into multi-quantum blocks.

    This post-processing step combines sessions for the same course/group/instructor/room
    that occur in consecutive time slots.

    Args:
        sessions: List of single-quantum CourseSession objects

    Returns:
        List of merged CourseSession objects with consecutive quanta combined
    """
    if not sessions:
        return []

    # Sort sessions by course, group, start quantum
    sorted_sessions = sorted(
        sessions,
        key=lambda s: (s.course_id, s.course_type, s.group_ids[0], s.session_quanta[0]),
    )

    merged = []
    current_session = None

    for session in sorted_sessions:
        if current_session is None:
            # Start new session
            current_session = session
            continue

        # Check if this session can be merged with current
        can_merge = (
            current_session.course_id == session.course_id
            and current_session.course_type == session.course_type
            and current_session.group_ids == session.group_ids
            and current_session.instructor_id == session.instructor_id
            and current_session.room_id == session.room_id
            and
            # Consecutive quantum check
            session.session_quanta[0] == current_session.session_quanta[-1] + 1
        )

        if can_merge:
            # Merge: extend quanta list
            current_session.session_quanta.extend(session.session_quanta)
        else:
            # Cannot merge: save current, start new
            merged.append(current_session)
            current_session = session

    # Don't forget last session
    if current_session is not None:
        merged.append(current_session)

    return merged
