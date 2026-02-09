# Constraint System (simplified - all constraints always enabled)
from schedule_engine.constraints import HARD_CONSTRAINT_CLASSES, SOFT_CONSTRAINT_CLASSES
from schedule_engine.domain.course import Course
from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.group import Group
from schedule_engine.domain.instructor import Instructor
from schedule_engine.domain.room import Room
from schedule_engine.domain.timetable import Timetable
from schedule_engine.domain.types import SchedulingContext
from schedule_engine.io.time_system import QuantumTimeSystem


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

    # Build Timetable
    context = SchedulingContext(
        courses=courses,
        instructors=instructors,
        groups=groups,
        rooms=rooms,
        qts=QuantumTimeSystem(),
    )
    tt = Timetable(genes=individual, context=context)

    # Hard constraint penalties (individual breakdown)
    hard_details = {}
    for constraint in HARD_CONSTRAINT_CLASSES:
        penalty = constraint.evaluate(tt)
        hard_details[constraint.name] = int(constraint.weight * penalty)

    # Soft constraint penalties (individual breakdown)
    soft_details = {}
    for constraint in SOFT_CONSTRAINT_CLASSES:
        penalty = constraint.evaluate(tt)
        soft_details[constraint.name] = constraint.weight * penalty

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
    return int(total_hard), int(total_soft)

