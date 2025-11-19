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
    # Thesis Experiments
    (" Thesis", "exp1", "Pure NSGA-II baseline", "30m test / 1-2h prod"),
    (" Thesis", "exp2", "+ IGLS repair system", "35m test / 1.5-2.5h prod"),
    (" Thesis", "exp3", "+ 19 heuristics (no LS)", "40m test / 2-3h prod"),
    (" Thesis", "exp4", "+ LNS local search", "45m test / 2.5-3.5h prod"),
    (" Thesis", "exp5", "+ RL adaptive selection", "50m test / 3-4h prod"),
    # Quick Tests
    (" Test", "test", "Smoke test (30 gens)", "~5 min"),
    (" Test", "test-baseline", "Test Mode 1: Pure NSGA-II", "~5 min"),
    (" Test", "test-repairs", "Test Mode 2: + Repairs", "~6 min"),
    (" Test", "test-heuristics", "Test Mode 3: + Heuristics", "~7 min"),
    # Production Runs
    (" Prod", "prod", "Full production (2000 gens)", "1.5-2.5h GPU"),
    (" Prod", "prod-baseline", "Prod Mode 1: Pure NSGA-II", "1-2h"),
    (" Prod", "prod-repairs", "Prod Mode 2: + Repairs", "1.5-2.5h"),
    (" Prod", "prod-heuristics", "Prod Mode 3: + Heuristics", "2-3h"),
    # Runtime Modes
    (" Mode", "baseline", "Mode 1: Pure NSGA-II", "Baseline"),
    (" Mode", "repairs", "Mode 2: + IGLS repairs", "Repair sys"),
    (" Mode", "heuristics", "Mode 3: + 19 heuristics", "Heuristics"),
    (" Mode", "full", "Mode 4: Full GA (best non-RL)", "Complete"),
    (" Mode", "rl", "Mode 5: RL-guided selection", "RL adaptive"),
    (" Mode", "roundrobin", "Mode 6: Fixed round-robin", "Fixed"),
    (" Mode", "specialists", "Mode 7: RL specialist agents", "Specialists"),
    (" Mode", "archive", "Mode 8: Archive diversity", "Diversity"),
    (" Mode", "hierarchical", "Mode 9: Hierarchical RL", "2-level RL"),
    (" Mode", "multiagent", "Mode 10: Multi-agent RL", "Multi-agent"),
    # RL Training
    (" RL", "train-rl", "Train RL agent (100K steps)", "2-4h GPU"),
    (" RL", "train-rl-quick", "Quick training (10K steps)", "15-30 min"),
    (" RL", "train-curriculum", "Curriculum learning", "Progressive"),
    (" RL", "select-checkpoint", "Choose best checkpoint", "Interactive"),
    (" RL", "promote-model", "Promote model to prod", "Promotion"),
    (" RL", "validate-rl", "Validate RL model", "Validation"),
    # Analysis
    (" Analysis", "compare-experiments", "Compare thesis experiments", "Comparison"),
    (" Analysis", "generate-thesis-plots", "Generate thesis plots", "Plots"),
    (" Analysis", "export-thesis-data", "Export metrics (CSV/LaTeX)", "Export"),
    (" Analysis", "analyze-convergence", "Convergence analysis", "Evolution"),
    (" Analysis", "analyze-diversity", "Diversity metrics", "Diversity"),
    # Diagnostics
    (" Diag", "diagnose-system", "Full system diagnostics", "Check all"),
    (" Diag", "diagnose-gpu", "GPU diagnostics", "CUDA/GPU"),
    (" Diag", "check-data", "Validate input data", "Data check"),
    (" Diag", "verify-config", "Verify configuration", "Config"),
    (" Diag", "verify-enhancements", "Check enhancements", "Features"),
    (" Diag", "test-dashboard", "Test Rich dashboard", "UI test"),
    # Benchmarking
    (" Bench", "benchmark-all", "Run all benchmarks", "All tests"),
    (" Bench", "benchmark-gpu", "GPU benchmark", "GPU test"),
    (" Bench", "benchmark-lns", "LNS benchmark", "LNS perf"),
    (" Bench", "benchmark-constraints", "Constraint benchmark", "Constraint"),
    # Configuration
    ("Config", "show-config", "Show all configuration", "Full config"),
    ("Config", "show-repair", "Show repair config", "Repair"),
    ("Config", "show-soft", "Show soft constraints", "Soft"),
    ("Config", "show-time", "Show time system", "Time"),
    ("Config", "list-experiments", "List all experiments", "Manifest"),
    # Development
    ("Dev", "tensorboard", "Start TensorBoard", "Visualization"),
    ("Dev", "git-squash", "Interactive squashing", "Git cleanup"),
    ("Dev", "clean-output", "Clean old outputs", "Disk cleanup"),
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
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
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
