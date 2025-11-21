#!/usr/bin/env python3
"""
Unified CLI Launcher for Schedule Engine

Convention:
- Main commands: 0-9 (nsga, train-rl, etc.)
- Helper commands: a-z (diagnose, clean, etc.)
- Profiles: --test, --med, --prod
- Configs: DRY hierarchy (test < med < prod)
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

console = Console()


def create_parser():
    """Create argument parser with all commands."""
    parser = argparse.ArgumentParser(
        description="Schedule Engine - Unified CLI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GA Experiments (Main Commands 0-4)
  uv run nsga --test          # Smoke test NSGA-II (30 gens, ~2 min)
  uv run nsga --med           # Medium NSGA-II (200 gens, ~30 min)
  uv run nsga --prod          # Production NSGA-II (2000 gens, ~3-5 hours)
  
  # RL Training (Main Command 5)
  uv run train-rl --test      # Smoke test RL (10K steps, ~5-10 min)
  uv run train-rl --med       # Medium RL (50K steps, ~30-45 min)
  uv run train-rl --prod      # Production RL (100K steps, ~1-2 hours)
  
  # Helpers (a-z)
  uv run diagnose             # System diagnostics
  uv run clean                # Clean output directory
  uv run list-experiments     # List all experiments

Profile Hierarchy (DRY):
  test: Quick smoke test (inherit base + test overrides)
  med:  Medium run (inherit test + med overrides)
  prod: Full production (inherit med + prod overrides)
        """,
    )

    # Profile selection (mutually exclusive)
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--test",
        action="store_const",
        const="test",
        dest="profile",
        help="Test profile (smoke test)",
    )
    profile_group.add_argument(
        "--med",
        action="store_const",
        const="med",
        dest="profile",
        help="Medium profile",
    )
    profile_group.add_argument(
        "--prod",
        action="store_const",
        const="prod",
        dest="profile",
        help="Production profile",
    )

    # Additional flags
    parser.add_argument(
        "--curriculum", action="store_true", help="Enable curriculum learning (RL only)"
    )
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--mode", type=str, help="Runtime mode override")
    parser.add_argument("--config", type=str, help="Custom config file")

    return parser


def main_nsga():
    """NSGA-II launcher with profile support."""
    parser = create_parser()
    args = parser.parse_args()

    profile = args.profile or "test"

    # Import and run main with environment
    from main import main

    sys.argv = ["main.py", "--env", profile]
    if args.name:
        sys.argv.extend(["--name", args.name])
    if args.mode:
        sys.argv.extend(["--mode", args.mode])
    if args.config:
        sys.argv.extend(["--config", args.config])

    sys.exit(main() or 0)


def main_train_rl():
    """RL training launcher with profile support."""
    parser = create_parser()
    args = parser.parse_args()

    profile = args.profile or "test"

    # Import RL training
    from src.rl.training.train_script import main as rl_main

    # Build argv for RL training
    sys.argv = ["train_script.py", "--profile", profile, "--agent", "ppo"]

    # Add curriculum flag if requested
    if args.curriculum:
        sys.argv.append("--curriculum")
    else:
        # Map profile to timesteps (if not using curriculum)
        timestep_map = {"test": 500, "med": 50000, "prod": 100000}
        sys.argv.extend(["--timesteps", str(timestep_map[profile])])

    sys.exit(rl_main() or 0)


# Helper commands (a-z)
def main_diagnose():
    """System diagnostics."""
    from scripts.diagnostics.diagnose_gpu import main as diagnose_gpu

    diagnose_gpu()


def main_clean():
    """Clean output directory."""
    import shutil
    from pathlib import Path

    output_dir = Path("output")
    if output_dir.exists():
        console.print(f"[yellow]Cleaning {output_dir}...[/yellow]")
        for item in output_dir.iterdir():
            if item.name != "experiment_manifest.json":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        console.print("[green]✓ Cleaned output directory[/green]")


def main_list():
    """List experiments."""
    from pathlib import Path
    import json
    from rich.table import Table

    manifest_path = Path("output/experiment_manifest.json")

    if not manifest_path.exists():
        console.print("[yellow]No experiments found[/yellow]")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    table = Table(title="Experiments")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Mode", style="yellow")
    table.add_column("Best Fitness", style="magenta")

    for exp in manifest.get("experiments", []):
        table.add_row(
            exp.get("timestamp", ""),
            exp.get("name", ""),
            exp.get("mode", ""),
            str(exp.get("best_fitness", "")),
        )

    console.print(table)


def main_interactive():
    """Interactive command launcher (TUI menu)."""
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.panel import Panel
    import subprocess

    commands = [
        # NSGA-II Experiments
        (
            "nsga",
            [
                ("1", "nsga --test", "Smoke test (30 gens, ~2 min)"),
                ("2", "nsga --med", "Medium run (200 gens, ~30 min)"),
                ("3", "nsga --prod", "Production (2000 gens, ~3-5 hrs)"),
            ],
        ),
        # RL Training
        (
            "rl",
            [
                ("4", "train-rl --test", "Smoke test (500 steps, ~2-3 min)"),
                ("5", "train-rl --med", "Medium run (50K steps, ~30-45 min)"),
                ("6", "train-rl --prod", "Production (100K steps, ~1-2 hrs)"),
                (
                    "c",
                    "train-rl --test --curriculum",
                    "Curriculum learning (test profile)",
                ),
            ],
        ),
        # Utilities
        (
            "misc",
            [
                ("7", "diagnose", "System diagnostics"),
                ("8", "clean", "Clean output directory"),
                ("9", "list-experiments", "List experiment history"),
            ],
        ),
    ]

    while True:
        console.clear()
        console.print(
            "\n[bold cyan]Schedule Engine - Interactive Launcher[/bold cyan]\n"
        )

        for category, cmds in commands:
            # Category header
            if category == "nsga":
                console.print("[bold magenta]NSGA-II Experiments[/bold magenta]")
            elif category == "rl":
                console.print("\n[bold magenta]RL Training[/bold magenta]")
            elif category == "misc":
                console.print("\n[bold magenta]Utilities[/bold magenta]")

            # Commands in category
            for num, cmd, desc in cmds:
                console.print(
                    f"  [yellow]{num}.[/yellow] [green]uv run {cmd:22}[/green]  [dim]{desc}[/dim]"
                )

        console.print("\n  [yellow]q.[/yellow] [dim]Exit[/dim]")
        console.print()

        choice = Prompt.ask("[cyan]Select[/cyan]", default="q").strip().lower()

        if choice == "q":
            console.print("[yellow]Goodbye![/yellow]")
            break

        # Flatten commands for lookup
        all_cmds = {}
        for _, cmds in commands:
            for num, cmd, _ in cmds:
                all_cmds[num] = cmd

        if choice in all_cmds:
            cmd = all_cmds[choice]
            console.print(f"\n[green]Running: uv run {cmd}[/green]\n")
            subprocess.run(["uv", "run"] + cmd.split(), check=False)
            input("\n[dim]Press Enter to continue...[/dim]")
        else:
            console.print(f"[red]Invalid choice: {choice}[/red]")
            input("\n[dim]Press Enter to continue...[/dim]")


if __name__ == "__main__":
    # Auto-detect command from script name
    script_name = Path(sys.argv[0]).stem

    if "nsga" in script_name:
        main_nsga()
    elif "train" in script_name and "rl" in script_name:
        main_train_rl()
    elif "diagnose" in script_name:
        main_diagnose()
    elif "clean" in script_name:
        main_clean()
    elif "list" in script_name:
        main_list()
    elif "launcher" in script_name or "interactive" in script_name:
        main_interactive()
    else:
        print(f"Unknown command: {script_name}")
        print("Use 'uv run launcher' to see all commands")
        sys.exit(1)
