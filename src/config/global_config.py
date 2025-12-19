"""
Global Time and Quanta Configuration

Shared across all experiments. Should rarely change.
"""

from dataclasses import dataclass


@dataclass
class GlobalTimeConfig:
    """Time and quantum settings shared across all experiments."""

    # Time system
    quantum_minutes: int = 60
    opening_time: str = "10:00"
    closing_time: str = "17:00"
    closed_days: list[str] | None = None

    # Time constraint parameters
    midday_break_start: str = "12:00"
    midday_break_end: str = "13:00"
    max_session_coalescence: int = 3
    max_sessions_per_day: int = 6
    preferred_block_size_min: int = 2
    preferred_block_size_max: int = 3

    # Soft constraint penalties
    theory_isolated_penalty: int = 5
    theory_oversized_penalty_per_quantum: int = 2
    theory_max_excused_isolated: int = 1
    practical_fragmentation_penalty: int = 10

    def __post_init__(self) -> None:
        if self.closed_days is None:
            self.closed_days = ["Saturday"]


# Singleton instance
GLOBAL_TIME_CONFIG = GlobalTimeConfig()
