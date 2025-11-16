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
            # Validate quanta on initialization
            self._validate_and_fix_quanta()

    def __setattr__(self, name, value):
        """Intercept quanta assignments to validate bounds."""
        if name == "quanta" and value is not None:
            # Validate quanta before setting
            from src.encoder.quantum_time_system import QuantumTimeSystem

            MAX_VALID_QUANTUM = QuantumTimeSystem().total_quanta
            if value and isinstance(value, list) and max(value) >= MAX_VALID_QUANTUM:
                # Clip invalid quanta
                value = [q for q in value if q < MAX_VALID_QUANTUM]
                if not value:
                    value = [0]  # Fallback to first valid quantum
        super().__setattr__(name, value)

    def _validate_and_fix_quanta(self):
        """Validate and fix quanta if they exceed valid range."""
        from src.encoder.quantum_time_system import QuantumTimeSystem

        MAX_VALID_QUANTUM = QuantumTimeSystem().total_quanta
        if self.quanta and max(self.quanta) >= MAX_VALID_QUANTUM:
            # Clip to valid range
            self.quanta = [q for q in self.quanta if q < MAX_VALID_QUANTUM]
            if not self.quanta:
                self.quanta = [0]

    @property
    def time_quantum(self) -> int:
        """Return the starting quantum for this session (first slot)."""

        if not self.quanta:
            return 0
        return self.quanta[0]

    @time_quantum.setter
    def time_quantum(self, new_start: int) -> None:
        """Shift the session so that it starts at ``new_start`` quantum.

        Note: This shifts ALL quanta in the session by the delta.
        Validates that the entire session fits within valid quantum range (0-43).
        If invalid, clips to the maximum valid start position.
        """

        if not self.quanta:
            self.quanta = [new_start]
            return

        delta = new_start - self.quanta[0]
        new_quanta = [q + delta for q in self.quanta]

        # Validate: check if any quantum exceeds valid range
        from src.encoder.quantum_time_system import QuantumTimeSystem

        MAX_VALID_QUANTUM = QuantumTimeSystem().total_quanta
        if new_quanta and max(new_quanta) >= MAX_VALID_QUANTUM:
            # Clip to max valid start position
            max_valid_start = MAX_VALID_QUANTUM - len(self.quanta)
            if max_valid_start >= 0 and new_start > max_valid_start:
                # Adjust to fit within bounds
                delta = max_valid_start - self.quanta[0]
                new_quanta = [q + delta for q in self.quanta]
            elif max_valid_start < 0:
                # Session too long to fit anywhere - clip quanta list
                new_quanta = list(range(0, MAX_VALID_QUANTUM))

        # Final safety clip: remove any quanta still >= MAX_VALID_QUANTUM
        new_quanta = [q for q in new_quanta if q < MAX_VALID_QUANTUM]
        if not new_quanta:
            new_quanta = [0]

        self.quanta = new_quanta

    @property
    def duration_quanta(self) -> int:
        """Number of quanta occupied by this session."""

        return len(self.quanta)
