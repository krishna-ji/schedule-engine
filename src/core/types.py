"""
Core Type Definitions

This module contains type-safe data structures used throughout the system.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.entities.course import Course
from src.entities.group import Group
from src.entities.instructor import Instructor
from src.entities.room import Room
from src.ga.sessiongene import SessionGene

# Alias used across GA and RL modules; DEAP wraps this list at runtime.
Individual = List[SessionGene]


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

    courses: Dict[tuple, Course]  # Keys are (course_code, course_type) tuples
    groups: Dict[str, Group]
    instructors: Dict[str, Instructor]
    rooms: Dict[str, Room]
    available_quanta: List[int]
    config: Optional[Any] = None

    def validate(self) -> List[str]:
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
