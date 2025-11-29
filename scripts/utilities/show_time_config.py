"""
Display Time Configuration Settings

Shows time-related parameters and quantum conversions for verification.
"""

from rich.console import Console

from src.config import get_config
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.utils.time_helpers import (
    get_midday_break_quanta,
    quantum_to_day_and_within_day,
)

console = Console()


def _time_to_minutes(time_str: str) -> int:
    hour, minute = map(int, time_str.split(":"))
    return hour * 60 + minute


def _within_day_quantum(
    day: str, qts: QuantumTimeSystem, time_str: str
) -> int | None:
    start_minutes = qts.day_start_time.get(day)
    quanta_count = qts.day_quanta_count.get(day, 0)
    if start_minutes is None or quanta_count == 0:
        return None

    end_minutes = start_minutes + quanta_count * QuantumTimeSystem.QUANTUM_MINUTES
    target_minutes = _time_to_minutes(time_str)
    if target_minutes < start_minutes or target_minutes >= end_minutes:
        return None

    return (target_minutes - start_minutes) // QuantumTimeSystem.QUANTUM_MINUTES


def _get_preferred_time_range_quanta(
    qts: QuantumTimeSystem, earliest: str, latest: str
) -> tuple[dict[str, int], dict[str, int]]:
    earliest_map: dict[str, int] = {}
    latest_map: dict[str, int] = {}

    for day in qts.DAY_NAMES:
        if not qts.is_operational(day):
            continue

        earliest_idx = _within_day_quantum(day, qts, earliest)
        latest_idx = _within_day_quantum(day, qts, latest)

        if earliest_idx is not None and latest_idx is not None:
            earliest_map[day] = earliest_idx
            latest_map[day] = latest_idx

    return earliest_map, latest_map


def main():
    console.print("\n[bold cyan]" + "═" * 60 + "[/bold cyan]")
    console.print("[bold cyan]TIME CONFIGURATION SETTINGS[/bold cyan]".center(110))
    console.print("[bold cyan]" + "═" * 60 + "[/bold cyan]\n")

    config = get_config()
    time_cfg = config.time

    # Basic quantum parameters
    console.print("[bold yellow]QUANTUM TIME SYSTEM PARAMETERS[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    console.print(
        f"  Quantum Duration:        [cyan]{time_cfg.quantum_minutes} minutes[/cyan]"
    )
    quanta_per_hour = max(1, 60 // time_cfg.quantum_minutes)
    console.print(f"  Quanta per Hour:         [cyan]{quanta_per_hour}[/cyan]")
    console.print(
        f"  Operating Window:        [cyan]{time_cfg.opening_time}-{time_cfg.closing_time}[/cyan]"
    )
    closed_days = ", ".join(sorted(time_cfg.closed_days)) or "None"
    console.print(f"  Closed Days:             [cyan]{closed_days}[/cyan]")
    console.print()

    # Session preferences
    console.print("[bold yellow]SESSION PREFERENCES[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    console.print(
        f"  Max Session Coalescence: [cyan]{time_cfg.max_session_coalescence} quanta[/cyan]"
    )
    console.print(
        f"  Max Sessions per Day:    [cyan]{time_cfg.max_sessions_per_day}[/cyan]"
    )
    console.print(
        "  Preferred Block Size:     [cyan]"
        f"{time_cfg.preferred_block_size_min}-{time_cfg.preferred_block_size_max} quanta"
        "[/cyan]"
    )
    console.print()

    # Preferred hours
    console.print("[bold yellow]PREFERRED OPERATING HOURS (Wall-Clock)[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    console.print(
        f"  Earliest Preferred:      [cyan]{time_cfg.earliest_preferred_time}[/cyan]"
    )
    console.print(
        f"  Latest Preferred:        [cyan]{time_cfg.latest_preferred_time}[/cyan]"
    )
    console.print()

    # Break settings
    console.print("[bold yellow]MIDDAY BREAK SETTINGS (Wall-Clock)[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    console.print(
        f"  Break Start:             [cyan]{time_cfg.midday_break_start}[/cyan]"
    )
    console.print(
        f"  Break End:               [cyan]{time_cfg.midday_break_end}[/cyan]"
    )
    console.print()

    # Initialize QuantumTimeSystem
    qts = QuantumTimeSystem()

    console.print("[bold cyan]" + "─" * 60 + "[/bold cyan]")
    console.print("[bold cyan]QUANTUM CONVERSIONS (Per Day)[/bold cyan]".center(110))
    console.print("[bold cyan]" + "─" * 60 + "[/bold cyan]\n")

    # Display operating days and their quantum ranges
    console.print("[bold yellow]OPERATING DAYS & QUANTUM RANGES[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    for day in qts.DAY_NAMES:
        if qts.is_operational(day):
            offset = qts.day_quanta_offset.get(day)
            count = qts.day_quanta_count.get(day, 0)
            hours = qts.operating_hours.get(day)

            if offset is None or count == 0 or not hours:
                console.print(f"  {day:12} [dim]Configuration incomplete[/dim]")
                continue

            console.print(
                f"  {day:12} [cyan]{hours[0]}-{hours[1]}[/cyan]  "
                f"-> Quanta [green]{offset:3d}-{offset+count-1:3d}[/green] ([yellow]{count:2d}[/yellow] total)"
            )
        else:
            console.print(f"  {day:12} [dim]CLOSED[/dim]")
    console.print()

    # Get break quanta for each day
    console.print(
        "[bold yellow]MIDDAY BREAK QUANTUM INDICES (Within-Day)[/bold yellow]"
    )
    console.print("[dim]" + "-" * 60 + "[/dim]")
    break_quanta = get_midday_break_quanta(qts)
    for day in qts.DAY_NAMES:
        if day in break_quanta:
            quanta_set = break_quanta[day]
            if quanta_set:
                min_q = min(quanta_set)
                max_q = max(quanta_set)
                console.print(
                    f"  {day:12} Within-day quanta [cyan]{min_q:2d}-{max_q:2d}[/cyan]"
                )
        elif qts.is_operational(day):
            console.print(f"  {day:12} [dim]Break time outside operating hours[/dim]")
    console.print()

    # Get preferred time range quanta
    console.print(
        "[bold yellow]PREFERRED HOURS QUANTUM INDICES (Within-Day)[/bold yellow]"
    )
    console.print("[dim]" + "-" * 60 + "[/dim]")
    earliest_quanta, latest_quanta = _get_preferred_time_range_quanta(
        qts, time_cfg.earliest_preferred_time, time_cfg.latest_preferred_time
    )
    for day in qts.DAY_NAMES:
        if day in earliest_quanta and day in latest_quanta:
            earliest = earliest_quanta[day]
            latest = latest_quanta[day]
            console.print(
                f"  {day:12} Preferred within-day quanta [cyan]{earliest:2d}-{latest:2d}[/cyan]"
            )
        elif qts.is_operational(day):
            console.print(
                f"  {day:12} [dim]Preferred hours outside operating hours[/dim]"
            )
    console.print()

    # Example conversions
    console.print("[bold yellow]EXAMPLE QUANTUM CONVERSIONS[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    example_quanta = [0, 5, 12, 24, 36]
    for q in example_quanta:
        if q < qts.total_quanta:
            day, within_day = quantum_to_day_and_within_day(q, qts)
            day_str, time_str = qts.quanta_to_time(q)
            console.print(
                f"  Quantum [yellow]{q:3d}[/yellow] -> {day:12} offset [cyan]{within_day:2d}[/cyan] ([green]{time_str}[/green])"
            )
    console.print()

    console.print("[bold cyan]" + "═" * 60 + "[/bold cyan]")
    console.print(
        f"[bold]Total Operational Quanta: [cyan]{qts.total_quanta}[/cyan][/bold]"
    )
    console.print("[bold cyan]" + "═" * 60 + "[/bold cyan]\n")
    console.print("[dim]All time configurations aligned with QuantumTimeSystem[/dim]")
    console.print("[dim]No hardcoded QUANTA_PER_DAY or magic numbers[/dim]")
    console.print(
        "[dim]To modify: Update Python presets in src/config/presets/data.py or profile overrides.[/dim]\n"
    )


if __name__ == "__main__":
    main()
