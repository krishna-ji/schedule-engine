"""Shared time and quanta scheduling settings.

This module isolates time-related configuration so experiment configs can
import shared defaults without redefining values.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimeSettingsConfig:
    """Time and quanta parameters shared by all profiles and experiments."""

    quantum_minutes: int = 60
    opening_time: str = "10:00"
    closing_time: str = "17:00"
    closed_days: list[str] = field(default_factory=lambda: ["Saturday"])

    midday_break_start: str = "12:00"
    midday_break_end: str = "13:00"
    max_session_coalescence: int = 3
    max_sessions_per_day: int = 6
    preferred_block_size_min: int = 2
    preferred_block_size_max: int = 3

    enforce_break_placement: bool = True
    break_window_start: str = "12:00"
    break_window_end: str = "14:00"
    break_min_quanta: int = 1
    break_violation_penalty: int = 8

    cohort_pairs: list[tuple[str, str]] = field(default_factory=list)

    theory_isolated_penalty: int = 5
    theory_oversized_penalty_per_quantum: int = 2
    theory_max_excused_isolated: int = 1
    practical_fragmentation_penalty: int = 10
