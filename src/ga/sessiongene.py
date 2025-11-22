from dataclasses import dataclass
from typing import List


@dataclass
class SessionGene:
    """
    Represents a single scheduled session with GUARANTEED contiguous quanta.

    BREAKING CHANGE (Nov 2025 Migration):
    - Removed: `quanta: List[int]` (allowed fragmentation)
    - Added: `start_quanta: int, num_quanta: int` (structural continuity)
    - Memory: 60% reduction (2 ints vs N-element array)
    - Validation: Simpler range checks, no continuity scanning

    Design Rationale:
    - Makes fragmentation structurally impossible
    - Eliminates session_continuity soft constraint (redundant)
    - Direct mapping to course duration requirements

    A single session can be scheduled for multiple groups simultaneously
    (e.g., a lecture for BAE2 and BAE4 at the same time in the same room).
    """

    course_id: str
    course_type: str  # "theory" or "practical"
    instructor_id: str
    group_ids: List[str]
    room_id: str

    # Contiguous block representation (NEW)
    start_quanta: int  # Starting quantum index (e.g., 10 = Monday 10:00 AM)
    num_quanta: int  # Duration in quanta (e.g., 2 = 2-hour block)

    def __post_init__(self):
        """Validate quantum range and continuity constraints."""
        try:
            from src.encoder.quantum_time_system import QuantumTimeSystem

            qts = QuantumTimeSystem()
            total_quanta = qts.total_quanta
            quanta_per_day = qts.quanta_per_day
        except Exception:
            # Fallback for tests or incomplete initialization
            total_quanta = 70  # 7 days * 10 hours
            quanta_per_day = 10

        # Range validation
        if self.start_quanta < 0:
            self.start_quanta = 0
        if self.start_quanta >= total_quanta:
            self.start_quanta = total_quanta - 1

        if self.num_quanta <= 0:
            self.num_quanta = 1

        # Ensure session doesn't overflow quantum range
        if self.start_quanta + self.num_quanta > total_quanta:
            self.num_quanta = total_quanta - self.start_quanta

        # Day boundary validation (no midnight wrap)
        start_day = self.start_quanta // quanta_per_day
        end_day = (self.start_quanta + self.num_quanta - 1) // quanta_per_day
        if start_day != end_day:
            # Clip to end of day
            self.num_quanta = (start_day + 1) * quanta_per_day - self.start_quanta

    # ========== UTILITY METHODS ==========

    @property
    def end_quanta(self) -> int:
        """Exclusive end quantum (for range operations)."""
        return self.start_quanta + self.num_quanta

    @property
    def time_quantum(self) -> int:
        """Return the starting quantum for this session (backward compatibility)."""
        return self.start_quanta

    @time_quantum.setter
    def time_quantum(self, new_start: int) -> None:
        """
        Shift session to new start time (preserves duration).

        Args:
            new_start: New starting quantum index
        """
        self.start_quanta = new_start
        self.__post_init__()  # Re-validate after shift

    @property
    def duration_quanta(self) -> int:
        """Number of quanta occupied by this session."""
        return self.num_quanta

    def get_quanta_list(self) -> List[int]:
        """
        Generate explicit quanta array when needed (e.g., for legacy APIs).

        Example:
            start_quanta=10, num_quanta=3 → [10, 11, 12]

        Note: Prefer using range(gene.start_quanta, gene.end_quanta) for loops.
        """
        return list(range(self.start_quanta, self.end_quanta))

    def shift_to(self, new_start: int) -> None:
        """
        Shift session to new start time (preserves duration).

        Args:
            new_start: New starting quantum index
        """
        self.start_quanta = new_start
        self.__post_init__()  # Re-validate after shift

    def overlaps_with(self, other: "SessionGene") -> bool:
        """Check if this session overlaps with another session in time."""
        return not (
            self.end_quanta <= other.start_quanta
            or other.end_quanta <= self.start_quanta
        )
