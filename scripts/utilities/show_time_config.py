"""
Display Time Configuration Settings

Shows time-related parameters and quantum conversions for verification.
"""

from src.encoder.quantum_time_system import QuantumTimeSystem
from src.utils.console import write_header, write_separator, write_info
from src.utils.time_helpers import (
    QUANTUM_MINUTES,
    QUANTA_PER_HOUR,
    MAX_SESSION_COALESCENCE,
    MAX_SESSIONS_PER_DAY,
    EARLIEST_PREFERRED_TIME,
    LATEST_PREFERRED_TIME,
    MIDDAY_BREAK_START_TIME,
    MIDDAY_BREAK_END_TIME,
    get_midday_break_quanta,
    get_preferred_time_range_quanta,
    quantum_to_day_and_within_day,
)


def main():
    write_header("TIME CONFIGURATION SETTINGS")
    write_info("")

    # Basic quantum parameters
    write_info("QUANTUM TIME SYSTEM PARAMETERS")
    write_separator("-")
    write_info(f"  Quantum Duration:        {QUANTUM_MINUTES} minutes")
    write_info(f"  Quanta per Hour:         {QUANTA_PER_HOUR}")
    write_info("")

    # Session preferences
    write_info("SESSION PREFERENCES")
    write_separator("-")
    write_info(f"  Max Session Coalescence: {MAX_SESSION_COALESCENCE} quanta")
    write_info(f"  Max Sessions per Day:    {MAX_SESSIONS_PER_DAY}")
    write_info("")

    # Preferred hours
    write_info("PREFERRED OPERATING HOURS (Wall-Clock)")
    write_separator("-")
    write_info(f"  Earliest Preferred:      {EARLIEST_PREFERRED_TIME}")
    write_info(f"  Latest Preferred:        {LATEST_PREFERRED_TIME}")
    write_info("")

    # Break settings
    write_info("MIDDAY BREAK SETTINGS (Wall-Clock)")
    write_separator("-")
    write_info(f"  Break Start:             {MIDDAY_BREAK_START_TIME}")
    write_info(f"  Break End:               {MIDDAY_BREAK_END_TIME}")
    write_info("")

    # Initialize QuantumTimeSystem
    qts = QuantumTimeSystem()

    write_separator()
    write_info("QUANTUM CONVERSIONS (Per Day)".center(80))
    write_separator()
    write_info("")

    # Display operating days and their quantum ranges
    write_info("OPERATING DAYS & QUANTUM RANGES")
    write_separator("-")
    for day in qts.DAY_NAMES:
        if qts.is_operational(day):
            offset = qts.day_quanta_offset[day]
            count = qts.day_quanta_count[day]
            hours = qts.operating_hours[day]
            write_info(
                f"  {day:12} {hours[0]}-{hours[1]}  "
                f"-> Quanta {offset:3d}-{offset+count-1:3d} ({count:2d} total)"
            )
        else:
            write_info(f"  {day:12} CLOSED")
    write_info("")

    # Get break quanta for each day
    write_info("MIDDAY BREAK QUANTUM INDICES (Within-Day)")
    write_separator("-")
    break_quanta = get_midday_break_quanta(qts)
    for day in qts.DAY_NAMES:
        if day in break_quanta:
            quanta_set = break_quanta[day]
            if quanta_set:
                min_q = min(quanta_set)
                max_q = max(quanta_set)
                write_info(f"  {day:12} Within-day quanta {min_q:2d}-{max_q:2d}")
        elif qts.is_operational(day):
            write_info(f"  {day:12} Break time outside operating hours")
    write_info("")

    # Get preferred time range quanta
    write_info("PREFERRED HOURS QUANTUM INDICES (Within-Day)")
    write_separator("-")
    earliest_quanta, latest_quanta = get_preferred_time_range_quanta(qts)
    for day in qts.DAY_NAMES:
        if day in earliest_quanta and day in latest_quanta:
            earliest = earliest_quanta[day]
            latest = latest_quanta[day]
            write_info(
                f"  {day:12} Preferred within-day quanta {earliest:2d}-{latest:2d}"
            )
        elif qts.is_operational(day):
            write_info(f"  {day:12} Preferred hours outside operating hours")
    write_info("")

    # Example conversions
    write_info("EXAMPLE QUANTUM CONVERSIONS")
    write_separator("-")
    example_quanta = [0, 5, 12, 24, 36]
    for q in example_quanta:
        if q < qts.total_quanta:
            day, within_day = quantum_to_day_and_within_day(q, qts)
            day_str, time_str = qts.quanta_to_time(q)
            write_info(
                f"  Quantum {q:3d} -> {day:12} offset {within_day:2d} ({time_str})"
            )
    write_info("")

    write_separator()
    write_info(f"Total Operational Quanta: {qts.total_quanta}")
    write_separator()
    write_info("")
    write_info("All time configurations aligned with QuantumTimeSystem")
    write_info("No hardcoded QUANTA_PER_DAY or magic numbers")
    write_info("")
    write_info("To modify: Edit config/time_config.py")
    write_info("")


if __name__ == "__main__":
    main()
