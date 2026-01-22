"""
Core Type Definitions

This module contains type-safe data structures used throughout the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from schedule_engine.domain.course import Course
from schedule_engine.domain.group import Group
from schedule_engine.domain.instructor import Instructor
from schedule_engine.domain.room import Room

if TYPE_CHECKING:
    from schedule_engine.domain.gene import SessionGene

__all__ = ["Individual", "SchedulingContext"]

# Alias used across GA and RL modules; DEAP wraps this list at runtime.
Individual = list["SessionGene"]


@dataclass
class SchedulingContext:
    """
    Type-safe container for scheduling context.

    Replaces the previously used Dict[str, Any] context parameter
    with a strongly-typed dataclass for better IDE support, type checking,
    and documentation.

    Attributes:
        courses: Dictionary mapping (course_code, course_type) tuples to Course objects
        groups: Dictionary mapping group IDs to Group objects
        instructors: Dictionary mapping instructor IDs to Instructor objects
        rooms: Dictionary mapping room IDs to Room objects
        available_quanta: List of available time quantum indices
    """

    courses: dict[tuple[str, str], Course]  # Keys are (course_code, course_type) tuples
    groups: dict[str, Group]
    instructors: dict[str, Instructor]
    rooms: dict[str, Room]
    available_quanta: list[int]
    config: Any | None = None
    cohort_pairs: list[tuple[str, str]] | None = None

    def validate(self) -> list[str]:
        """
        Validate the scheduling context for consistency.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.courses:
            errors.append("No courses loaded")

        if not self.groups:
            errors.append("No groups loaded")

        if not self.instructors:
            errors.append("No instructors loaded")

        if not self.rooms:
            errors.append("No rooms loaded")

        if not self.available_quanta:
            errors.append("No available time quanta defined")

        return errors
