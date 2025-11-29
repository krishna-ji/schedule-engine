# Constraint Registry
from src.constraints.registry import (
    constraint_needs_courses,
    get_enabled_hard_constraints,
    get_enabled_soft_constraints,
)
from src.decoder.individual_decoder import decode_individual
from src.entities.course import Course
from src.entities.group import Group
from src.entities.instructor import Instructor
from src.entities.room import Room
from src.ga.sessiongene import SessionGene


def evaluate(
    individual: list[SessionGene],
    courses: dict[tuple, Course],  # Keys are (course_code, course_type) tuples
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room] | None = None,
) -> tuple[int, int]:
    """
    Evaluates a timetable individual using both hard and soft constraints.

    Hard constraints affect feasibility and must ideally reach zero.
    Soft constraints reflect schedule quality and should be minimized.

    Returns:
        Tuple[int, int]: (hard_penalty_score, soft_penalty_score)
    """
    # Get rooms from context if not provided
    if rooms is None:
        # For backward compatibility, create empty rooms dict
        rooms = {}

    sessions = decode_individual(individual, courses, instructors, groups, rooms)

    # Hard constraint penalty (using registry)
    hard_penalty = 0
    enabled_hard_constraints = get_enabled_hard_constraints()

    for constraint_name, constraint_info in enabled_hard_constraints.items():
        constraint_func = constraint_info["function"]
        weight = constraint_info["weight"]

        # Some hard constraints need courses parameter (centralized in metadata.py)
        if constraint_needs_courses(constraint_name):
            penalty = constraint_func(sessions, courses)
        else:
            penalty = constraint_func(sessions)

        hard_penalty += weight * penalty

    # Soft constraint penalty (using registry)
    soft_penalty = 0
    enabled_soft_constraints = get_enabled_soft_constraints()

    for _constraint_name, constraint_info in enabled_soft_constraints.items():
        constraint_func = constraint_info["function"]
        weight = constraint_info["weight"]
        soft_penalty += weight * constraint_func(sessions)

    return (hard_penalty, soft_penalty)
