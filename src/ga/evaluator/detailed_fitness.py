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


def evaluate_detailed(
    individual: list[SessionGene],
    courses: dict[tuple, Course],  # Keys are (course_code, course_type) tuples
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Evaluates a timetable individual with detailed constraint breakdown.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: (hard_constraint_details, soft_constraint_details)
    """
    # Get rooms from context if not provided
    if rooms is None:
        rooms = {}

    sessions = decode_individual(individual, courses, instructors, groups, rooms)

    # Hard constraint penalties (individual breakdown using registry)
    hard_details = {}
    enabled_hard_constraints = get_enabled_hard_constraints()

    for constraint_name, constraint_info in enabled_hard_constraints.items():
        constraint_func = constraint_info["function"]
        weight = constraint_info["weight"]

        # Some hard constraints need courses parameter (centralized in metadata.py)
        if constraint_needs_courses(constraint_name):
            penalty = constraint_func(sessions, courses)
        else:
            penalty = constraint_func(sessions)

        hard_details[constraint_name] = weight * penalty

    # Soft constraint penalties (individual breakdown using registry)
    soft_details = {}
    enabled_soft_constraints = get_enabled_soft_constraints()

    for constraint_name, constraint_info in enabled_soft_constraints.items():
        constraint_func = constraint_info["function"]
        weight = constraint_info["weight"]

        # Check if soft constraint needs courses parameter
        if constraint_needs_courses(constraint_name):
            penalty = constraint_func(sessions, courses)
        else:
            penalty = constraint_func(sessions)

        soft_details[constraint_name] = weight * penalty

    return hard_details, soft_details


def evaluate_from_detailed(
    hard_details: dict[str, int], soft_details: dict[str, int]
) -> tuple[int, int]:
    """
    Convert detailed constraint breakdown to total penalties.

    Returns:
        Tuple[int, int]: (total_hard_penalty, total_soft_penalty)
    """
    total_hard = sum(hard_details.values())
    total_soft = sum(soft_details.values())
    return total_hard, total_soft
