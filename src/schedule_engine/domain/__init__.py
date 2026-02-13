"""Domain layer: Core data models and types.

This module contains all domain entities, gene representation, and types.

Usage:
    from schedule_engine.domain import Course, Group, Instructor, Room, CourseSession
    from schedule_engine.domain import SessionGene, SchedulingContext, Individual
"""

from __future__ import annotations

from schedule_engine.domain.course import Course
from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.group import Group
from schedule_engine.domain.instructor import Instructor
from schedule_engine.domain.room import Room
from schedule_engine.domain.session import CourseSession
from schedule_engine.domain.timetable import ConflictPair, Timetable
from schedule_engine.domain.types import Individual, SchedulingContext

__all__ = [
    # Entities
    "Course",
    "CourseSession",
    "Group",
    "Instructor",
    "Room",
    # Gene
    "SessionGene",
    # Timetable (pre-indexed schedule view)
    "ConflictPair",
    "Timetable",
    # Types
    "Individual",
    "SchedulingContext",
]
