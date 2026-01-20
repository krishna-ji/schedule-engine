"""Data loader for experiment notebooks.

Provides single-function data loading with all entity linking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.encoder.input_encoder import (
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.ga.course_group_pairs import generate_course_group_pairs
from src.ga.group_hierarchy import analyze_group_hierarchy

if TYPE_CHECKING:
    from src.entities.course import Course
    from src.entities.group import Group
    from src.entities.instructor import Instructor
    from src.entities.room import Room


@dataclass
class ScheduleData:
    """Container for all loaded schedule data."""

    courses: dict[tuple[str, str], Course]
    instructors: dict[str, Instructor]
    rooms: dict[str, Room]
    groups: dict[str, Group]
    course_group_pairs: list[tuple[tuple[str, str], list[str], str, int]]
    qts: QuantumTimeSystem

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Courses: {len(self.courses)}, "
            f"Instructors: {len(self.instructors)}, "
            f"Rooms: {len(self.rooms)}, "
            f"Groups: {len(self.groups)}, "
            f"Pairs: {len(self.course_group_pairs)}, "
            f"Quanta: {self.qts.total_quanta}"
        )


def load_data(
    data_dir: str | Path = "data",
    opening_time: str = "10:00",
    closing_time: str = "17:00",
    closed_days: list[str] | None = None,
) -> ScheduleData:
    """Load all scheduling data with entity linking.

    Args:
        data_dir: Path to data directory containing JSON files
        opening_time: Daily start time (default: "10:00")
        closing_time: Daily end time (default: "17:00")
        closed_days: Days with no classes (default: ["Saturday"])

    Returns:
        ScheduleData containing all entities and course-group pairs

    Example:
        >>> data = load_data("../data")
        >>> print(data.summary())
        Courses: 668, Instructors: 181, Rooms: 67, Groups: 74, Pairs: 527
    """
    if closed_days is None:
        closed_days = ["Saturday"]

    data_path = Path(data_dir)

    # Build operating hours dict for QuantumTimeSystem
    operating_hours: dict[str, tuple[str, str] | None] = {}
    for day in [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]:
        if day in closed_days:
            operating_hours[day] = None
        else:
            operating_hours[day] = (opening_time, closing_time)

    # Initialize time system
    qts = QuantumTimeSystem(operating_hours=operating_hours)

    # Load entities
    courses = load_courses(str(data_path / "Course.json"))
    instructors = load_instructors(str(data_path / "Instructors.json"), qts)
    rooms = load_rooms(str(data_path / "Rooms.json"), qts)
    groups = load_groups(str(data_path / "Groups.json"), qts)

    # Link entities
    link_courses_and_instructors(courses, instructors)
    link_courses_and_groups(courses, groups)

    # Generate course-group pairs
    hierarchy = analyze_group_hierarchy(groups)
    course_group_pairs = generate_course_group_pairs(
        courses, groups, hierarchy, silent=True
    )

    return ScheduleData(
        courses=courses,
        instructors=instructors,
        rooms=rooms,
        groups=groups,
        course_group_pairs=course_group_pairs,
        qts=qts,
    )
