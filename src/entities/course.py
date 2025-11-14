"""
Course entity model for the timetabling system.
Represents a course with its attributes and requirements.
"""

from typing import List, Set
from dataclasses import dataclass


@dataclass
class Course:
    """
    Represents a course in the university timetabling system.

    Attributes:
        course_id: Unique identifier for the course
        name: Display name of the course
        quanta_per_week: Number of sessions required per week
        required_room_features: Type of room required (e.g., 'lecture', 'lab', 'seminar')
        enrolled_group_ids: List of student group IDs enrolled in this course
        qualified_instructor_ids: List of instructor IDs qualified to teach this course
        course_type: Type of course - 'theory' or 'practical'
    """

    course_id: str
    name: str
    quanta_per_week: int
    required_room_features: str
    enrolled_group_ids: List[str]
    qualified_instructor_ids: List[str]
    course_type: str = "theory"  # Default to theory

    def __post_init__(self):
        """Validate course data after initialization."""
        if self.quanta_per_week <= 0:
            raise ValueError(
                f"Course {self.course_id}: quanta_per_week must be positive"
            )
        # Note: enrolled_group_ids can be empty for certain courses/semesters
        # Note: qualified_instructor_ids can be empty for certain courses/semesters

    def is_instructor_qualified(self, instructor_id: str) -> bool:
        """Check if an instructor is qualified to teach this course."""
        return instructor_id in self.qualified_instructor_ids

    def has_group(self, group_id: str) -> bool:
        """Check if a group is enrolled in this course."""
        return group_id in self.enrolled_group_ids

    def get_enrolled_groups(self) -> Set[str]:
        """Get set of enrolled group IDs."""
        return set(self.enrolled_group_ids)
