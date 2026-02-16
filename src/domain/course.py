"""Course entity model for the timetabling system."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Course"]


@dataclass(slots=True)
class Course:
    """
    Represents a course in the university timetabling system.

    Attributes:
        course_id: Unique identifier for the course
        name: Display name of the course
        quanta_per_week: Number of sessions required per week
        required_room_features: Type of room required (e.g., 'lecture', 'lab', 'seminar')
                               NOTE: Currently str, but may change to str | None = None
                               to support courses without specific room requirements.
        enrolled_group_ids: List of student group IDs enrolled in this course
        qualified_instructor_ids: List of instructor IDs qualified to teach this course
        specific_lab_features: Specific lab features required (e.g., 'networking lab')
                               from PracticalRoomFeatures in Course.json
        course_type: Type of course - 'theory' or 'practical'
        course_code: Course code (may differ from course_id if split into theory/practical)
        department: Department offering the course
        semester: Semester the course is offered
        credits: Number of credits for the course
        lecture_hours: Total lecture hours (L+T for theory courses)
        practical_hours: Total practical hours (P for practical courses)
    """

    course_id: str
    name: str
    quanta_per_week: int
    required_room_features: str
    enrolled_group_ids: list[str] = field(default_factory=list)
    qualified_instructor_ids: list[str] = field(default_factory=list)
    course_type: str = "theory"  # Default to theory
    L: int = 0  # Lecture hours (for theory type)
    T: int = 0  # Tutorial hours (for theory type)
    P: int = 0  # Practical hours (for practical type)
    course_code: str = ""  # Course code from JSON
    department: str = ""  # Department
    semester: str = ""  # Semester
    credits: int = 0  # Credits
    lecture_hours: int = 0  # L+T for theory
    practical_hours: int = 0  # P for practical
    specific_lab_features: list[str] = field(
        default_factory=list
    )  # e.g. ["networking lab", "general programming lab"]

    def __post_init__(self) -> None:
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

    def get_enrolled_groups(self) -> set[str]:
        """Get set of enrolled group IDs."""
        return set(self.enrolled_group_ids)
