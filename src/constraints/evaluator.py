"""
Individual constraint evaluator for per-constraint breakdown.

ENHANCEMENT #2: Provides fine-grained constraint violation analysis
for RL state representation and targeted repair strategies.
"""

from __future__ import annotations

from src.constraints.hard import (
    course_completeness,
    instructor_exclusivity,
    instructor_qualifications,
    instructor_time_availability,
    room_exclusivity,
    room_suitability,
    room_time_availability,
    student_group_exclusivity,
)
from src.constraints.soft import (
    instructor_schedule_compactness,
    session_continuity,
    student_lunch_break,
    student_schedule_compactness,
)
from src.entities.course import Course
from src.entities.decoded_session import CourseSession


class ConstraintEvaluator:
    """
    Evaluates individual constraints and provides per-constraint breakdown.

    Used by RL state encoder to provide fine-grained constraint information
    for targeted repair strategies.
    """

    # Hard constraint function mapping
    HARD_CONSTRAINTS = {
        "student_group_exclusivity": student_group_exclusivity,
        "instructor_exclusivity": instructor_exclusivity,
        "instructor_qualifications": instructor_qualifications,
        "room_suitability": room_suitability,
        "instructor_time_availability": instructor_time_availability,
        "room_time_availability": room_time_availability,
        "course_completeness": course_completeness,
        "room_exclusivity": room_exclusivity,
    }

    # Soft constraint function mapping
    SOFT_CONSTRAINTS = {
        "student_schedule_compactness": student_schedule_compactness,
        "instructor_schedule_compactness": instructor_schedule_compactness,
        "student_lunch_break": student_lunch_break,
        "session_continuity": session_continuity,
    }

    def __init__(self, course_map: dict[tuple, Course] | None = None):
        """
        Initialize constraint evaluator.

        Args:
            course_map: Mapping from (course_id, course_type) to Course entity.
                       Required for constraints that need course information.
        """
        self.course_map = course_map or {}

    def evaluate_hard_breakdown(self, sessions: list[CourseSession]) -> dict[str, int]:
        """
        Evaluate all hard constraints individually.

        Args:
            sessions: List of decoded course sessions

        Returns:
            Dictionary mapping constraint names to violation counts
        """
        breakdown = {}

        for name, func in self.HARD_CONSTRAINTS.items():
            try:
                # Check if constraint needs course information
                if name in ["instructor_qualifications", "room_suitability"]:
                    violations = func(sessions, self.course_map)
                else:
                    violations = func(sessions)
                breakdown[name] = violations
            except Exception as e:
                # Log error but don't crash - return 0 for failed constraint
                print(f"Warning: Failed to evaluate constraint {name}: {e}")
                breakdown[name] = 0

        return breakdown

    def evaluate_soft_breakdown(
        self, sessions: list[CourseSession]
    ) -> dict[str, float]:
        """
        Evaluate all soft constraints individually.

        Args:
            sessions: List of decoded course sessions

        Returns:
            Dictionary mapping constraint names to penalty counts
        """
        breakdown = {}

        for name, func in self.SOFT_CONSTRAINTS.items():
            try:
                # Check if constraint needs course information
                if name == "session_continuity":
                    penalties = func(sessions, self.course_map)
                else:
                    penalties = func(sessions)
                breakdown[name] = penalties
            except Exception as e:
                # Log error but don't crash - return 0 for failed constraint
                print(f"Warning: Failed to evaluate constraint {name}: {e}")
                breakdown[name] = 0

        return breakdown

    def evaluate_full_breakdown(
        self, sessions: list[CourseSession]
    ) -> dict[str, int | float]:
        """
        Evaluate all constraints and return combined breakdown.

        Args:
            sessions: List of decoded course sessions

        Returns:
            Dictionary with all constraint violations (hard + soft)
        """
        hard_breakdown = self.evaluate_hard_breakdown(sessions)
        soft_breakdown = self.evaluate_soft_breakdown(sessions)

        # Combine both dictionaries
        return {**hard_breakdown, **soft_breakdown}

    def get_top_violators(
        self, sessions: list[CourseSession], top_n: int = 3
    ) -> list[tuple[str, int | float]]:
        """
        Get the top N most violated constraints.

        Useful for prioritizing repair efforts.

        Args:
            sessions: List of decoded course sessions
            top_n: Number of top violators to return

        Returns:
            List of (constraint_name, violation_count) tuples, sorted by violations
        """
        breakdown = self.evaluate_full_breakdown(sessions)

        # Sort by violation count (descending)
        sorted_violations = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)

        return sorted_violations[:top_n]

    def get_constraint_priorities(
        self, sessions: list[CourseSession]
    ) -> dict[str, float]:
        """
        Calculate constraint priority scores for targeted repair.

        Priority = violation_count * weight (from config)

        Args:
            sessions: List of decoded course sessions

        Returns:
            Dictionary mapping constraint names to priority scores
        """
        breakdown = self.evaluate_full_breakdown(sessions)
        priorities = {}

        # Apply weights (from config or defaults)
        # Hard constraints typically have weight 2.0-3.0
        # Soft constraints typically have weight 0.5-2.0
        for constraint_name, violations in breakdown.items():
            if constraint_name in self.HARD_CONSTRAINTS:
                weight = 3.0  # Default hard constraint weight
            else:
                weight = 1.0  # Default soft constraint weight

            priorities[constraint_name] = violations * weight

        return priorities
