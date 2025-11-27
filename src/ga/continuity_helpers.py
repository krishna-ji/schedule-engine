"""
Continuity enforcement helpers for population initialization.

Key Functions:
- calculate_session_durations(): Convert course L/P values to duration list
- find_contiguous_window(): Locate valid time slots for given duration
- build_availability_grid(): Track resource usage across quanta
"""

from dataclasses import dataclass

from src.config import get_config
from src.core.types import SchedulingContext
from src.entities.course import Course


@dataclass
class SessionDuration:
    """Represents required duration for a single session."""

    course_id: str
    course_type: str  # "theory" or "practical"
    session_index: int  # 0-based within course type
    num_quanta: int  # Required contiguous quanta

    def __repr__(self):
        return f"{self.course_id}[{self.course_type[0].upper()}{self.session_index}]:{self.num_quanta}q"


class ContinuityHelper:
    """Helper class for enforcing session continuity in population initialization."""

    def __init__(self, context: SchedulingContext):
        self.context = context
        self.qts = context.quantum_time_system
        self.config = get_config()

    def calculate_session_durations(self, course: Course) -> list[SessionDuration]:
        """
        Calculate required durations for all sessions of a course.

        Rules:
        - Theory (L hours):
            - Odd L: [2, 2, ..., 1] (e.g., L=5 → [2, 2, 1])
            - Even L: [2, 2, ...] (e.g., L=4 → [2, 2])
        - Practical (P hours):
            - Single contiguous block: [P] (e.g., P=3 → [3])

        Args:
            course: Course entity with L and P values

        Returns:
            List of SessionDuration objects (one per subsession)

        Example:
            course.L = 5, course.P = 3
            → [
                SessionDuration(type="theory", index=0, num_quanta=2),
                SessionDuration(type="theory", index=1, num_quanta=2),
                SessionDuration(type="theory", index=2, num_quanta=1),
                SessionDuration(type="practical", index=0, num_quanta=3),
            ]
        """
        durations = []
        block_size = 2  # Default: 2-quantum blocks for theory

        # Theory sessions (L hours)
        if course.L > 0:
            theory_quanta = course.L
            full_blocks = theory_quanta // block_size
            remainder = theory_quanta % block_size

            for i in range(full_blocks):
                durations.append(
                    SessionDuration(
                        course_id=course.id,
                        course_type="theory",
                        session_index=i,
                        num_quanta=block_size,
                    )
                )

            if remainder > 0:
                durations.append(
                    SessionDuration(
                        course_id=course.id,
                        course_type="theory",
                        session_index=full_blocks,
                        num_quanta=remainder,
                    )
                )

        # Practical session (P hours) - single contiguous block
        if course.P > 0:
            max_practical = 10  # Safety limit (avoid 10-hour marathons)
            practical_quanta = min(course.P, max_practical)

            durations.append(
                SessionDuration(
                    course_id=course.id,
                    course_type="practical",
                    session_index=0,
                    num_quanta=practical_quanta,
                )
            )

        return durations

    def find_contiguous_window(
        self,
        duration: SessionDuration,
        used_quanta: dict[str, set[int]],
        instructor_id: str,
        room_id: str,
        group_ids: list[str],
    ) -> int | None:
        """
        Find a valid starting quantum for a session with given duration.

        Validation:
        - All quanta in [start, start+duration) must be free for:
            - Instructor
            - Room
            - All groups
        - No day boundary crossing

        Args:
            duration: Session duration requirements
            used_quanta: Dictionary of already-allocated quanta per resource
            instructor_id: Assigned instructor
            room_id: Assigned room
            group_ids: Enrolled groups

        Returns:
            Starting quantum index if found, else None
        """
        candidates = []

        for start_q in range(self.qts.total_quanta - duration.num_quanta + 1):
            if self._is_window_valid(
                start_q,
                duration.num_quanta,
                used_quanta,
                instructor_id,
                room_id,
                group_ids,
            ):
                score = self._score_window(start_q, duration.num_quanta)
                candidates.append((start_q, score))

        if not candidates:
            return None

        # Return best candidate (lowest score = best)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _is_window_valid(
        self,
        start_q: int,
        duration: int,
        used_quanta: dict[str, set[int]],
        instructor_id: str,
        room_id: str,
        group_ids: list[str],
    ) -> bool:
        """Check if time window is available for all resources."""
        window = range(start_q, start_q + duration)

        # Day boundary check
        start_day = start_q // self.qts.quanta_per_day
        end_day = (start_q + duration - 1) // self.qts.quanta_per_day
        if start_day != end_day:
            return False

        # Resource availability checks
        for q in window:
            if q in used_quanta.get(instructor_id, set()):
                return False
            if q in used_quanta.get(room_id, set()):
                return False
            for group_id in group_ids:
                if q in used_quanta.get(group_id, set()):
                    return False

        return True

    def _score_window(self, start_q: int, duration: int) -> float:
        """
        Heuristic scoring for window placement (lower is better).

        Factors:
        - Earlier times preferred (morning > afternoon)
        """
        # Time preference (favor earlier slots)
        time_of_day = start_q % self.qts.quanta_per_day
        time_score = time_of_day * 0.1

        return time_score
