"""Example module demonstrating strict typing enforcement.

This file serves as a live demonstration of the strict typing requirements
enforced by the project's coding standards.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.entities.course import Course
from src.entities.instructor import Instructor

logger = logging.getLogger(__name__)


def filter_courses_by_instructor(
    courses: list[Course],
    instructor: Instructor,
) -> list[Course]:
    """Filter courses taught by a specific instructor.

    Args:
        courses: List of courses to filter
        instructor: Instructor to filter by

    Returns:
        List of courses taught by the instructor
    """
    filtered: list[Course] = []
    for course in courses:
        if instructor.instructor_id in course.qualified_instructor_ids:
            filtered.append(course)
    return filtered


def calculate_instructor_workload(
    instructor: Instructor,
    courses: list[Course],
) -> dict[str, int | float]:
    """Calculate workload statistics for an instructor.

    Args:
        instructor: The instructor to analyze
        courses: All available courses

    Returns:
        Dictionary with workload metrics (total_courses, total_quanta, avg_quanta)
    """
    instructor_courses = filter_courses_by_instructor(courses, instructor)
    total_courses = len(instructor_courses)
    total_quanta = sum(course.quanta_per_week for course in instructor_courses)
    avg_quanta = total_quanta / total_courses if total_courses > 0 else 0.0

    return {
        "total_courses": total_courses,
        "total_quanta": total_quanta,
        "avg_quanta": avg_quanta,
    }


class CourseAnalyzer:
    """Analyzer for course scheduling metrics with strict typing."""

    # Class attributes
    default_threshold: float = 0.8

    def __init__(
        self,
        courses: list[Course],
        instructors: list[Instructor],
        threshold: float | None = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            courses: List of courses to analyze
            instructors: List of available instructors
            threshold: Optional custom threshold (uses default if None)
        """
        self.courses: list[Course] = courses
        self.instructors: list[Instructor] = instructors
        self.threshold: float = (
            threshold if threshold is not None else self.default_threshold
        )
        self._cache: dict[str, dict[str, int | float]] = {}

    def get_workload(self, instructor_id: str) -> dict[str, int | float]:
        """Get workload for an instructor with caching.

        Args:
            instructor_id: ID of the instructor

        Returns:
            Workload metrics dictionary

        Raises:
            ValueError: If instructor_id not found
        """
        if instructor_id in self._cache:
            return self._cache[instructor_id]

        instructor = self._find_instructor(instructor_id)
        if instructor is None:
            raise ValueError(f"Instructor {instructor_id} not found")

        workload = calculate_instructor_workload(instructor, self.courses)
        self._cache[instructor_id] = workload
        return workload

    def _find_instructor(self, instructor_id: str) -> Instructor | None:
        """Find instructor by ID.

        Args:
            instructor_id: ID to search for

        Returns:
            Instructor if found, None otherwise
        """
        for instructor in self.instructors:
            if instructor.instructor_id == instructor_id:
                return instructor
        return None

    def filter_by_predicate(
        self,
        predicate: Callable[[Course], bool],
    ) -> list[Course]:
        """Filter courses using a custom predicate function.

        Args:
            predicate: Function that returns True for courses to include

        Returns:
            Filtered list of courses
        """
        return [course for course in self.courses if predicate(course)]


def main() -> None:
    """Example usage demonstrating strict typing throughout."""
    # This function shows that even simple scripts are strictly typed
    logger.info("CourseAnalyzer example - all types enforced")


if __name__ == "__main__":
    main()
