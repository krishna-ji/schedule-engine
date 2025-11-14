"""
Time configuration helper functions.
These are used across the codebase for time calculations.
"""

from src.encoder.quantum_time_system import QuantumTimeSystem
from src.config import get_config


def get_midday_break_quanta(qts: QuantumTimeSystem):
    """
    Get quantum indices for midday break period.

    Args:
        qts: QuantumTimeSystem instance

    Returns:
        Dict mapping day_name -> set of quantum indices (within-day) for break period
    """
    cfg = get_config()
    break_quanta = {}

    for day in qts.DAY_NAMES:
        if not qts.is_operational(day):
            continue

        try:
            break_start_q = qts.time_to_quanta(day, cfg.time.midday_break_start)
            break_end_q = qts.time_to_quanta(day, cfg.time.midday_break_end)

            day_offset = qts.day_quanta_offset[day]
            within_day_start = break_start_q - day_offset
            within_day_end = break_end_q - day_offset

            break_quanta[day] = set(range(within_day_start, within_day_end))
        except ValueError:
            continue

    return break_quanta


def quantum_to_day_and_within_day(quantum, qts: QuantumTimeSystem):
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

        if day_offset <= quantum < day_offset + day_count:
            within_day = quantum - day_offset
            return day, within_day

    raise ValueError(f"Quantum {quantum} out of valid range")
