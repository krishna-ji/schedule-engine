from dataclasses import dataclass
from typing import List


@dataclass
class SessionGene:
    """
    Each SessionGene Represents a single session in the timetable.
    This is for the purpose of DEAP Encoding and Genetic Algorithm Engine (GAE)
    It contains the course, instructor, group(s), room, and quanta information.

    A single session can be scheduled for multiple groups simultaneously
    (e.g., a lecture for BAE2 and BAE4 at the same time in the same room).

    Clean architecture: course_id is plain code (e.g., "ENME 103"),
    course_type distinguishes "theory" vs "practical".
    """

    course_id: str
    course_type: str  # "theory" or "practical"
    instructor_id: str
    group_ids: List[str]  # Changed from group_id to support multiple groups
    room_id: str
    quanta: List[int]

    def __post_init__(self):
        """Normalize internal state for compatibility with heuristic operators."""
        # Ensure quanta are always sorted so derived properties remain consistent.
        if self.quanta:
            self.quanta = sorted(self.quanta)

    @property
    def time_quantum(self) -> int:
        """Return the starting quantum for this session (first slot)."""

        if not self.quanta:
            return 0
        return self.quanta[0]

    @time_quantum.setter
    def time_quantum(self, new_start: int) -> None:
        """Shift the session so that it starts at ``new_start`` quantum."""

        if not self.quanta:
            self.quanta = [new_start]
            return

        delta = new_start - self.quanta[0]
        self.quanta = [q + delta for q in self.quanta]

    @property
    def duration_quanta(self) -> int:
        """Number of quanta occupied by this session."""

        return len(self.quanta)
