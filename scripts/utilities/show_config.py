"""
Display current constraint configuration (both hard and soft).
Quick utility to see which constraints are enabled and their weights.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table

from src.config import get_config

console = Console()


def main():
    config = get_config()

    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]CONSTRAINT CONFIGURATION[/bold cyan]".center(110))
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")

    # Hard Constraints Section
    console.print("\n[bold yellow]HARD CONSTRAINTS (Feasibility)[/bold yellow]\n")

    hard_table = Table(show_header=True, header_style="bold magenta")
    hard_table.add_column("Constraint", style="cyan", width=45)
    hard_table.add_column("Enabled", justify="center", width=10)
    hard_table.add_column("Weight", justify="right", width=10)

    hard_constraints = [
        (
            "student_group_exclusivity",
            config.hard_constraints.student_group_exclusivity,
        ),
        ("instructor_exclusivity", config.hard_constraints.instructor_exclusivity),
        (
            "instructor_qualifications",
            config.hard_constraints.instructor_qualifications,
        ),
        (
            "instructor_time_availability",
            config.hard_constraints.instructor_time_availability,
        ),
        ("room_suitability", config.hard_constraints.room_suitability),
        ("room_exclusivity", config.hard_constraints.room_exclusivity),
        ("room_time_availability", config.hard_constraints.room_time_availability),
        ("course_completeness", config.hard_constraints.course_completeness),
    ]

    hard_enabled_count = sum(1 for _, cfg in hard_constraints if cfg.enabled)
    hard_total_count = len(hard_constraints)

    for name, cfg in hard_constraints:
        status = "[green]✓[/green]" if cfg.enabled else "[dim]✗[/dim]"
        hard_table.add_row(name, status, f"{cfg.weight:.2f}")

    console.print(hard_table)
    console.print(f"\n[green]● {hard_enabled_count}/{hard_total_count} enabled[/green]")

    total_hard_weight = sum(cfg.weight for _, cfg in hard_constraints if cfg.enabled)
    console.print(f"[bold]Total weight: [cyan]{total_hard_weight:.2f}[/cyan][/bold]")

    # Soft Constraints Section
    console.print("\n[bold yellow]SOFT CONSTRAINTS (Quality)[/bold yellow]\n")

    soft_table = Table(show_header=True, header_style="bold magenta")
    soft_table.add_column("Constraint", style="cyan", width=40)
    soft_table.add_column("Enabled", justify="center", width=10)
    soft_table.add_column("Weight", justify="right", width=10)
    soft_table.add_column("Penalty Info", style="dim", width=20)

    soft_constraints = [
        (
            "student_schedule_compactness",
            config.soft_constraints.student_schedule_compactness,
        ),
        (
            "instructor_schedule_compactness",
            config.soft_constraints.instructor_schedule_compactness,
        ),
        ("student_lunch_break", config.soft_constraints.student_lunch_break),
        ("session_continuity", config.soft_constraints.session_continuity),
    ]

    soft_enabled_count = sum(1 for _, cfg in soft_constraints if cfg.enabled)
    soft_total_count = len(soft_constraints)

    for name, cfg in soft_constraints:
        status = "[green]✓[/green]" if cfg.enabled else "[dim]✗[/dim]"
        penalty_info = ""
        if cfg.gap_penalty_per_quantum:
            penalty_info = f"gap: {cfg.gap_penalty_per_quantum}"
        elif cfg.distance_penalty_per_quantum:
            penalty_info = f"dist: {cfg.distance_penalty_per_quantum}"
        soft_table.add_row(name, status, f"{cfg.weight:.2f}", penalty_info)

    console.print(soft_table)
    console.print(f"\n[green]● {soft_enabled_count}/{soft_total_count} enabled[/green]")
    console.print(
        f"[bold]Soft weight factor: [cyan]{config.soft_constraints.soft_weight_factor:.2f}[/cyan][/bold]"
    )

    total_soft_weight = sum(cfg.weight for _, cfg in soft_constraints if cfg.enabled)
    console.print(
        f"[bold]Total soft weight: [cyan]{total_soft_weight:.2f}[/cyan][/bold]"
    )

    # Summary
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print(
        f"[bold]TOTAL: [green]{hard_enabled_count + soft_enabled_count}/{hard_total_count + soft_total_count}[/green] constraints enabled[/bold]"
    )
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print(
        "\n[dim]To modify: Update Python presets in src/config/presets/data.py or profile overrides.[/dim]\n"
    )


if __name__ == "__main__":
    main()
