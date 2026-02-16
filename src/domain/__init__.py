"""Domain layer: Core data models and types.

This module contains all domain entities, gene representation, and types.

Usage:
    from src.domain import Course, Group, Instructor, Room, CourseSession
    from src.domain import SessionGene, SchedulingContext, Individual
"""

from __future__ import annotations

from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.domain.session import CourseSession
from src.domain.timetable import ConflictPair, Timetable
from src.domain.types import Individual, SchedulingContext

__all__ = [
    # Timetable (pre-indexed schedule view)
    "ConflictPair",
    # Entities
    "Course",
    "CourseSession",
    "Group",
    # Types
    "Individual",
    "Instructor",
    "Room",
    "SchedulingContext",
    # Gene
    "SessionGene",
    "Timetable",
]
