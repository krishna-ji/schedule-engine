from typing import List, Dict, Tuple
from src.decoder.individual_decoder import decode_individual
from src.ga.sessiongene import SessionGene
from src.entities.course import Course
from src.entities.instructor import Instructor
from src.entities.group import Group
from src.entities.room import Room

# Constraint Registries
from src.constraints.hard import get_enabled_hard_constraints
from src.constraints.soft import get_enabled_soft_constraints


def evaluate_detailed(
    individual: List[SessionGene],
    courses: Dict[tuple, Course],  # Keys are (course_code, course_type) tuples
    instructors: Dict[str, Instructor],
    groups: Dict[str, Group],
    rooms: Dict[str, Room] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
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

        # Some hard constraints need courses parameter
        if constraint_name in [
            "instructor_qualifications",
            "course_completeness",
        ]:
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
        penalty = constraint_func(sessions)
        soft_details[constraint_name] = weight * penalty

    return hard_details, soft_details


def evaluate_from_detailed(
    hard_details: Dict[str, int], soft_details: Dict[str, int]
) -> Tuple[int, int]:
    """
    Convert detailed constraint breakdown to total penalties.

    Returns:
        Tuple[int, int]: (total_hard_penalty, total_soft_penalty)
    """
    total_hard = sum(hard_details.values())
    total_soft = sum(soft_details.values())
    return total_hard, total_soft
