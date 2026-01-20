"""Time configuration helper functions used across the codebase."""

from __future__ import annotations

from src.config import get_config
from src.io.time_system import QuantumTimeSystem


def get_midday_break_quanta(qts: QuantumTimeSystem) -> dict[str, set[int]]:
    """
    Get quantum indices for midday break period.

    Args:
        qts: QuantumTimeSystem instance

    Returns:
        Dict mapping day_name -> set of quantum indices (within-day) for break period
    """
    cfg = get_config()
    break_quanta: dict[str, set[int]] = {}

    for day in qts.DAY_NAMES:
        if not qts.is_operational(day):
            continue

        try:
            break_start_q = qts.time_to_quanta(day, cfg.time.midday_break_start)
            break_end_q = qts.time_to_quanta(day, cfg.time.midday_break_end)

            day_offset = qts.day_quanta_offset[day]
            if day_offset is None:
                continue

            within_day_start = break_start_q - day_offset
            within_day_end = break_end_q - day_offset

            break_quanta[day] = set(range(within_day_start, within_day_end))
        except ValueError:
            continue

    return break_quanta


def quantum_to_day_and_within_day(
    quantum: int, qts: QuantumTimeSystem
) -> tuple[str, int]:
    """
    Convert continuous quantum to (day_name, within_day_quantum).

    Args:
        quantum: Continuous quantum index
        qts: QuantumTimeSystem instance

    Returns:
        Tuple of (day_name, within_day_quantum_index)
    """
    for day in qts.DAY_NAMES:
        if qts.day_quanta_offset[day] is None:
            continue

        day_offset = qts.day_quanta_offset[day]
        day_count = qts.day_quanta_count[day]

        if day_offset is None or day_count is None:
            continue

        if day_offset <= quantum < day_offset + day_count:
            within_day = quantum - day_offset
            return day, within_day

    raise ValueError(f"Quantum {quantum} out of valid range")
