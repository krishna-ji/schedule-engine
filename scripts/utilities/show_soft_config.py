"""
Display current soft constraint configuration.
Quick utility to see which constraints are enabled and their weights.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from rich.console import Console

from schedule_engine.config import get_config  # get_config().soft_constraints
from schedule_engine.constraints.registry import get_enabled_soft_constraints

console = Console()


def main():
    console.print("\n[bold cyan]" + "═" * 60 + "[/bold cyan]")
    console.print("[bold cyan]SOFT CONSTRAINT CONFIGURATION[/bold cyan]".center(110))
    console.print("[bold cyan]" + "═" * 60 + "[/bold cyan]\n")

    enabled_count = sum(
        1 for cfg in get_config().soft_constraints.values() if cfg["enabled"]
    )
    total_count = len(get_config().soft_constraints)

    console.print(
        f"[green]Status: {enabled_count}/{total_count} constraints enabled[/green]\n"
    )

    # Show enabled constraints
    enabled = get_enabled_soft_constraints()
    if enabled:
        console.print("[bold]ENABLED CONSTRAINTS:[/bold]")
        console.print("[dim]" + "-" * 60 + "[/dim]")
        for name, info in enabled.items():
            console.print(f"  {name:<40} weight = [cyan]{info['weight']:.2f}[/cyan]")
        console.print()

    # Show disabled constraints
    disabled = [
        name
        for name, cfg in get_config().soft_constraints.items()
        if not cfg["enabled"]
    ]
    if disabled:
        console.print("[bold]DISABLED CONSTRAINTS:[/bold]")
        console.print("[dim]" + "-" * 60 + "[/dim]")
        for name in disabled:
            console.print(
                f"  {name:<40} [dim](weight = {get_config().soft_constraints[name]['weight']:.2f})[/dim]"
            )
        console.print()

    # Show total weight
    total_weight = sum(info["weight"] for info in enabled.values())
    console.print(f"[bold]Total enabled weight: [cyan]{total_weight:.2f}[/cyan][/bold]")
    console.print("[bold cyan]" + "═" * 60 + "[/bold cyan]")
    console.print(
        "[dim]To modify: Update defaults in src/schedule_engine/config/models.py or supply overrides via schedule_engine.config.loader.[/dim]\n"
    )


if __name__ == "__main__":
    main()
