"""Utility helpers shared across heuristic modules."""

from __future__ import annotations

from typing import Iterable, Set

from src.core.types import SchedulingContext
from src.ga.sessiongene import SessionGene


def get_course_for_gene(context: SchedulingContext, gene: SessionGene):
    """Return the Course entity associated with a gene, regardless of key format."""

    course_id = gene.course_id
    course_type = getattr(gene, "course_type", None)

    courses = context.courses

    # Try tuple key first (course_id, course_type)
    if course_type is not None:
        key = (course_id, course_type)
        if key in courses:
            return courses[key]

    # Direct string key (legacy contexts)
    if course_id in courses:
        return courses[course_id]

    # Fall back to first matching course_id regardless of type
    for key, course in courses.items():
        if isinstance(key, tuple) and key[0] == course_id:
            if course_type is None or key[1] == course_type:
                return course

    raise KeyError(course_id)


def get_available_quanta(context: SchedulingContext) -> list[int]:
    """Return a sorted list of available quanta from the context."""

    quanta = getattr(context, "available_quanta", None)
    if not quanta:
        return []
    return sorted(quanta)


def get_room_feature(room) -> str:
    """Normalize room type/feature attribute used by heuristics."""

    if hasattr(room, "room_type"):
        return getattr(room, "room_type")
    if hasattr(room, "room_features"):
        return getattr(room, "room_features")
    return "lecture"


def get_course_room_requirement(course) -> str:
    """Return the normalized room requirement string for a course."""

    if hasattr(course, "required_room_features"):
        return getattr(course, "required_room_features")
    return getattr(course, "required_room_type", "lecture")


def estimate_session_student_count(
    gene: SessionGene, context: SchedulingContext
) -> int:
    """Estimate student count for a gene based on its assigned groups."""

    total = 0
    for group_id in getattr(gene, "group_ids", []) or []:
        group = context.groups.get(group_id)
        if group and hasattr(group, "student_count"):
            total += group.student_count
    return total or 0


def is_instructor_available(
    instructor,
    time_range: Iterable[int],
) -> bool:
    """Check instructor availability across the specified time quanta."""

    if instructor is None:
        return False

    if getattr(instructor, "is_full_time", True):
        return True

    available = getattr(instructor, "available_quanta", set())
    return all(q in available for q in time_range)


def move_gene_to_time_if_valid(
    gene: SessionGene,
    new_start: int,
    valid_quanta: Set[int],
) -> bool:
    """Shift a gene to a new start time only if all quanta remain valid."""

    if not gene.quanta:
        if new_start not in valid_quanta:
            return False
        gene.quanta = [new_start]
        return True

    shift_delta = new_start - gene.quanta[0]
    shifted_quanta = [q + shift_delta for q in gene.quanta]

    if any(q not in valid_quanta for q in shifted_quanta):
        return False

    gene.quanta = shifted_quanta
    return True
