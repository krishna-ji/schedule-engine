"""Decoded course session entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.group import Group
    from src.domain.instructor import Instructor
    from src.domain.room import Room

__all__ = ["CourseSession"]


@dataclass(slots=True)
class CourseSession:
    """
    Represents a fully decoded session of a course within the university timetabling system.

    This object is used after decoding a GA individual (chromosome) for constraint checking,
    visualization, and scheduling logic. Unlike the gene, this class includes semantic
    information like resolved Instructor and Group objects.

    Attributes:
        course_id (str): Unique identifier for the course.
        instructor_id (str): ID of the instructor assigned to this session.
        group_ids (List[str]): List of group IDs attending this session.
        room_id (str): ID of the room assigned to this session.
        session_quanta (List[int]): List of time quanta (e.g., 15-min blocks) during which the session is scheduled.
        required_room_features (str): Room feature type required (e.g., "lab", "lecture").
        course_type (str): Type of course - 'theory' or 'practical'.
        instructor (Instructor, optional): Reference to the assigned Instructor object.
        group (Group, optional): Reference to the primary Group object (if applicable).
        room (Room, optional): Reference to the assigned Room object.
    """

    course_id: str
    instructor_id: str
    group_ids: list[str]
    room_id: str
    session_quanta: list[int]
    required_room_features: str | None = None
    course_type: str = "theory"
    instructor: Instructor | None = None
    group: Group | None = None
    room: Room | None = None
    co_instructor_ids: list[str] = field(default_factory=list)
