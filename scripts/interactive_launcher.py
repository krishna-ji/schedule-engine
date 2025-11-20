"""
Interactive UV Command Launcher - FLAT LIST (KISS)

Simple flat list of all commands - just select by number!
Run with: uv run launcher
"""

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich import box
import subprocess
import sys
from typing import List, Tuple

console = Console()


# Flat list of ALL commands (category, command, description, runtime)
ALL_COMMANDS: List[Tuple[str, str, str, str]] = [
    # 1. RL Guided Hyperheuristic
    (" RL-Guided", "train-rl", "Train RL agent", "2-4h GPU"),
    (" RL-Guided", "train-curriculum", "Curriculum learning", "Progressive"),
    (" RL-Guided", "test-rl", "Test RL-guided (Mode 5)", "30 gens"),
    (" RL-Guided", "prod-rl", "Prod RL-guided (Mode 5)", "2000 gens"),
    # 2. NSGA-II Only
    (" NSGA-Only", "test-nsga", "Test Pure NSGA-II (Mode 1)", "30 gens"),
    (" NSGA-Only", "prod-nsga", "Prod Pure NSGA-II (Mode 1)", "2000 gens"),
    # 3. NSGA-II + Repair
    (" NSGA+Repair", "test-repair", "Test NSGA + Repairs (Mode 2)", "30 gens"),
    (" NSGA+Repair", "prod-repair", "Prod NSGA + Repairs (Mode 2)", "2000 gens"),
    # 4. NSGA-II + Repair + Local Search
    (" NSGA+LS", "test-ls", "Test Full (Mode 4)", "30 gens"),
    (" NSGA+LS", "prod-ls", "Prod Full (Mode 4)", "2000 gens"),
    # 5. Round Robin Heuristics
    (" RoundRobin", "test-roundrobin", "Test Round Robin (Mode 6)", "30 gens"),
    (" RoundRobin", "prod-roundrobin", "Prod Round Robin (Mode 6)", "2000 gens"),
    # Diagnostics & Utilities
    (" Utils", "diagnose-system", "Full system diagnostics", "Check all"),
    (" Utils", "clean-output", "Clean old output files", "Cleanup"),
    (" Utils", "list-experiments", "List available experiments", "Manifest"),
]


def display_all_commands() -> None:
    """Display all commands in categorized list format (no tables)."""
    console.clear()
    console.print()
    console.print("[bold cyan]Schedule Engine - All Commands[/bold cyan]")
    console.print("[dim]Simple list - select by number or command name[/dim]")
    console.print()

    # Group commands by category
    current_category = None
    cmd_number = 1

    for category, cmd, desc, runtime in ALL_COMMANDS:
        # Print category header when it changes
        if category != current_category:
            console.print()
            console.print(f"[bold magenta]{category}[/bold magenta]")
            console.print("[dim]" + "-" * 80 + "[/dim]")
            current_category = category

        # Print command line
        console.print(
            f"[yellow]{cmd_number:2}.[/yellow] "
            f"[green]uv run {cmd:22}[/green] "
            f"{desc:38} "
            f"[dim]({runtime})[/dim]"
        )
        cmd_number += 1

    console.print()
    console.print("[dim]" + "-" * 80 + "[/dim]")
    console.print(f"[dim]Total: {len(ALL_COMMANDS)} commands[/dim]")
    console.print()


def run_command(cmd: str):
    """Execute UV command with live output."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Running:[/bold green] [cyan]uv run {cmd}[/cyan]",
            border_style="green",
        )
    )
    console.print()

    try:
        # Run command with live output
        result = subprocess.run(["uv", "run", cmd], check=False, text=True)

        console.print()
        if result.returncode == 0:
            console.print("[bold green]✓ Command completed successfully[/bold green]")
        else:
            console.print(
                f"[bold red]✗ Command failed with exit code {result.returncode}[/bold red]"
            )

        return result.returncode

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        return 1


def main():
    """Main interactive launcher loop."""
    try:
        while True:
            # Display all commands
            display_all_commands()

            console.print("[dim]Options:[/dim]")
            console.print("  • [cyan]Enter command number[/cyan] (1-52) to run")
            console.print(
                "  • [cyan]Type command name[/cyan] directly (e.g., 'exp1', 'test')"
            )
            console.print("  • [yellow]'q' or 'quit'[/yellow] to exit")
            console.print()

            choice = (
                Prompt.ask("[bold cyan]Select command[/bold cyan]", default="q")
                .strip()
                .lower()
            )

            # Exit
            if choice in ["q", "quit", "exit"]:
                console.print("[yellow] Goodbye![/yellow]")
                break

            # Number selection
            try:
                cmd_idx = int(choice) - 1
                if 0 <= cmd_idx < len(ALL_COMMANDS):
                    _, selected_cmd, _, _ = ALL_COMMANDS[cmd_idx]
                    run_command(selected_cmd)

                    console.print()
                    if (
                        not Prompt.ask(
                            "[cyan]Run another command?[/cyan]",
                            choices=["y", "n"],
                            default="y",
                        )
                        == "y"
                    ):
                        break
                else:
                    console.print(
                        f"[red]Invalid number: {choice} (must be 1-{len(ALL_COMMANDS)})[/red]"
                    )
                    input("\nPress Enter to continue...")
            except ValueError:
                # Direct command name
                cmd_map = {cmd: cmd for _, cmd, _, _ in ALL_COMMANDS}
                if choice in cmd_map:
                    run_command(choice)

                    console.print()
                    if (
                        not Prompt.ask(
                            "[cyan]Run another command?[/cyan]",
                            choices=["y", "n"],
                            default="y",
                        )
                        == "y"
                    ):
                        break
                else:
                    console.print(f"[red]Unknown command: {choice}[/red]")
                    console.print(
                        "[dim]Tip: Use command number or exact name like 'exp1', 'test', etc.[/dim]"
                    )
                    input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        console.print("\n[yellow] Interrupted. Goodbye![/yellow]")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
