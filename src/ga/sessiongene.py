from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.encoder.quantum_time_system import QuantumTimeSystem


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
    group_ids: list[str]
    room_id: str

    # Contiguous block representation (NEW)
    start_quanta: int  # Starting quantum index (e.g., 10 = Monday 10:00 AM)
    num_quanta: int  # Duration in quanta (e.g., 2 = 2-hour block)

    def __post_init__(self) -> None:
        """Validate quantum range and continuity constraints."""
        qts, total_quanta = _get_time_system_metadata()

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

        # Smart day boundary validation: Only allow multi-day if duration exceeds single day capacity
        # Prevents illogical midnight wraps (e.g., 2-hour session spanning Mon 4PM - Tue 8AM)
        # Allows multi-day courses when necessary (e.g., AR701=30hrs needs ~4-5 days)
        if qts is not None:
            day_bounds = _get_day_bounds(qts, self.start_quanta)
            if day_bounds is not None:
                day_offset, day_quanta = day_bounds
                end_of_day = day_offset + day_quanta
                if (
                    self.num_quanta <= day_quanta
                    and self.start_quanta + self.num_quanta > end_of_day
                ):
                    self.start_quanta = max(day_offset, end_of_day - self.num_quanta)

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

    def get_quanta_list(self) -> list[int]:
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

    def overlaps_with(self, other: SessionGene) -> bool:
        """Check if this session overlaps with another session in time."""
        return not (
            self.end_quanta <= other.start_quanta
            or other.end_quanta <= self.start_quanta
        )


def _get_time_system_metadata() -> tuple[QuantumTimeSystem | None, int]:  # type: ignore[name-defined]
    """Fetch QuantumTimeSystem info with safe fallbacks."""
    global _SESSION_GENE_QTS

    if _SESSION_GENE_QTS is None:
        try:
            from src.encoder.quantum_time_system import QuantumTimeSystem

            _SESSION_GENE_QTS = QuantumTimeSystem()
        except Exception:
            _SESSION_GENE_QTS = None

    if _SESSION_GENE_QTS is None:
        # Fallback for tests or incomplete initialization
        return None, 70

    qts = _SESSION_GENE_QTS
    return qts, qts.total_quanta


def _get_day_bounds(qts: QuantumTimeSystem, quantum: int) -> tuple[int, int] | None:  # type: ignore[name-defined]
    """Return (day_offset, day_quanta_count) for the given quantum."""
    for day in qts.DAY_NAMES:
        day_offset = qts.day_quanta_offset.get(day)
        day_quanta = qts.day_quanta_count.get(day, 0)
        if day_offset is None or day_quanta <= 0:
            continue
        if day_offset <= quantum < day_offset + day_quanta:
            return day_offset, day_quanta
    return None


_SESSION_GENE_QTS: QuantumTimeSystem | None = None  # type: ignore[name-defined]
