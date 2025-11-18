"""
Display Time Configuration Settings

Shows time-related parameters and quantum conversions for verification.
"""

from src.encoder.quantum_time_system import QuantumTimeSystem
from rich.console import Console
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

console = Console()


def main():
    console.print("\n[bold cyan]" + "═" * 60 + "[/bold cyan]")
    console.print("[bold cyan]TIME CONFIGURATION SETTINGS[/bold cyan]".center(110))
    console.print("[bold cyan]" + "═" * 60 + "[/bold cyan]\n")

    # Basic quantum parameters
    console.print("[bold yellow]QUANTUM TIME SYSTEM PARAMETERS[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    console.print(f"  Quantum Duration:        [cyan]{QUANTUM_MINUTES} minutes[/cyan]")
    console.print(f"  Quanta per Hour:         [cyan]{QUANTA_PER_HOUR}[/cyan]")
    console.print()

    # Session preferences
    console.print("[bold yellow]SESSION PREFERENCES[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    console.print(
        f"  Max Session Coalescence: [cyan]{MAX_SESSION_COALESCENCE} quanta[/cyan]"
    )
    console.print(f"  Max Sessions per Day:    [cyan]{MAX_SESSIONS_PER_DAY}[/cyan]")
    console.print()

    # Preferred hours
    console.print("[bold yellow]PREFERRED OPERATING HOURS (Wall-Clock)[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    console.print(f"  Earliest Preferred:      [cyan]{EARLIEST_PREFERRED_TIME}[/cyan]")
    console.print(f"  Latest Preferred:        [cyan]{LATEST_PREFERRED_TIME}[/cyan]")
    console.print()

    # Break settings
    console.print("[bold yellow]MIDDAY BREAK SETTINGS (Wall-Clock)[/bold yellow]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
    console.print(f"  Break Start:             [cyan]{MIDDAY_BREAK_START_TIME}[/cyan]")
    console.print(f"  Break End:               [cyan]{MIDDAY_BREAK_END_TIME}[/cyan]")
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
            offset = qts.day_quanta_offset[day]
            count = qts.day_quanta_count[day]
            hours = qts.operating_hours[day]
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
    earliest_quanta, latest_quanta = get_preferred_time_range_quanta(qts)
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
    console.print("[dim]To modify: Edit configs/base.yaml[/dim]\n")


if __name__ == "__main__":
    main()
