#!/usr/bin/env python3
"""
Unified CLI Launcher for Schedule Engine

Convention:
- Main commands: 0-99+ (nsga, train-rl, curriculum, etc.)
- Helper commands: a-z (diagnose, clean, test-gpu, etc.)
- Profiles: --test, --med, --prod
- Configs: DRY hierarchy (test < med < prod)
"""

import sys
import os
import argparse
from pathlib import Path
from rich.console import Console

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

console = Console()


def create_parser():
    """Create minimal argument parser."""
    parser = argparse.ArgumentParser(
        description="Schedule Engine", add_help=False  # No --help drama
    )

    # Profile selection
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--test", action="store_const", const="test", dest="profile"
    )
    profile_group.add_argument(
        "--med", action="store_const", const="med", dest="profile"
    )
    profile_group.add_argument(
        "--prod", action="store_const", const="prod", dest="profile"
    )

    # Options
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--repair-after-every-generation", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--mode")
    parser.add_argument("--config")

    return parser


def main_train_rl():
    """RL training launcher with profile support."""
    parser = create_parser()
    args = parser.parse_args()

    profile = args.profile or "test"

    # Ensure downstream config loader uses the matching environment profile.
    # GA/RL share the same YAML hierarchy (configs/test.yaml, configs/prod.yaml).
    # Map RL profiles onto those environments so every worker process
    # inherits the correct settings instead of falling back to test.
    env_profile = "test" if profile == "test" else "prod"
    schedule_config_path = (
        "configs/test.yaml" if env_profile == "test" else "configs/prod.yaml"
    )

    os.environ["ENVIRONMENT"] = env_profile
    os.environ["SCHEDULE_CONFIG"] = schedule_config_path

    console.print(
        f"[dim]ENVIRONMENT set to[/dim] {env_profile} [dim]for RL profile[/dim] {profile}"
    )
    console.print(f"[dim]SCHEDULE_CONFIG ->[/dim] {schedule_config_path}")

    # Import RL training
    from src.rl.training.train_script import main as rl_main

    # Build argv for RL training
    sys.argv = ["train_script.py", "--profile", profile, "--agent", "ppo"]

    # Add curriculum flag if requested (curriculum works with any profile)
    if args.curriculum:
        sys.argv.append("--curriculum")
    else:
        # Map profile to timesteps (if not using curriculum)
        timestep_map = {"test": 500, "med": 50000, "prod": 100000}
        sys.argv.extend(["--timesteps", str(timestep_map[profile])])

    sys.exit(rl_main() or 0)


# ==================
# PROGRESSIVE MODE EXPERIMENTS (A→E: Increasing Complexity)
# ==================
# Mode A: Pure NSGA-II                           [repairs: NO,  memetic: NO,  heuristics: NO]
# Mode B: + Memetic local search                 [repairs: YES, memetic: YES, heuristics: NO]
# Mode C: + Round-robin (heuristics + repair)    [repairs: YES (round-robin), memetic: NO, heuristics: YES (fixed)]
# Mode D: + Adaptive heuristics                  [repairs: YES, memetic: NO,  heuristics: YES (adaptive)]
# Mode E: + RL-guided (deploys all techniques)   [repairs: YES, memetic: YES, heuristics: YES (RL-controlled)]


def main_baseline():
    """Mode A: Pure NSGA-II baseline (no repairs, no heuristics)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = args.profile or "test"

    from main import main

    sys.argv = ["main.py", "--config", "configs/baseline/a-pure-nsga.yaml", "--env", profile]
    if args.name:
        sys.argv.extend(["--experiment", args.name])

    console.print(
        f"[green]Mode A: Pure NSGA-II baseline ({profile} profile)[/green]"
    )
    console.print("[dim]  [repairs: NO, memetic: NO, heuristics: NO][/dim]")
    sys.exit(main() or 0)


def main_memetic():
    """Mode B: NSGA-II + memetic local search (no heuristics)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = args.profile or "test"

    from main import main

    sys.argv = ["main.py", "--config", "configs/nsga/b-nsga-memetic.yaml", "--env", profile]
    if args.name:
        sys.argv.extend(["--experiment", args.name])

    console.print(
        f"[green]Mode B: NSGA-II + memetic local search ({profile} profile)[/green]"
    )
    console.print("[dim]  [repairs: YES, memetic: YES, heuristics: NO][/dim]")
    sys.exit(main() or 0)


def main_roundrobin():
    """Mode C: Round-robin heuristics + repair (fixed rotation)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = args.profile or "test"

    from main import main

    sys.argv = [
        "main.py",
        "--config",
        "configs/hybrid/c-roundrobin.yaml",
        "--env",
        profile,
    ]
    if args.name:
        sys.argv.extend(["--experiment", args.name])

    console.print(
        f"[green]Mode C: Round-robin heuristics + repair ({profile} profile)[/green]"
    )
    console.print("[dim]  [repairs: YES (round-robin), memetic: NO, heuristics: YES (fixed rotation)][/dim]")
    sys.exit(main() or 0)


def main_adaptive():
    """Mode D: Adaptive heuristic selection."""
    parser = create_parser()
    args = parser.parse_args()

    profile = args.profile or "test"

    from main import main

    sys.argv = [
        "main.py",
        "--config",
        "configs/hybrid/d-adaptive.yaml",
        "--env",
        profile,
    ]
    if args.name:
        sys.argv.extend(["--experiment", args.name])

    console.print(
        f"[green]Mode D: Adaptive heuristic selection ({profile} profile)[/green]"
    )
    console.print("[dim]  [repairs: YES, memetic: NO, heuristics: YES (adaptive selection)][/dim]")
    sys.exit(main() or 0)


def main_rl():
    """Mode E: RL-guided hyper-heuristic (all techniques)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = args.profile or "test"

    from main import main

    sys.argv = ["main.py", "--config", "configs/rl/e-rl-guided.yaml", "--env", profile]
    if args.name:
        sys.argv.extend(["--experiment", args.name])

    console.print(
        f"[green]Mode E: RL-guided hyper-heuristic ({profile} profile)[/green]"
    )
    console.print("[dim]  [repairs: YES, memetic: YES, heuristics: YES (RL-controlled)][/dim]")
    sys.exit(main() or 0)





# Helper commands (a-z)
def main_diagnose():
    """System diagnostics."""
    from scripts.diagnostics.diagnose_gpu import main as diagnose_gpu

    diagnose_gpu()


def main_test_gpu():
    """Quick GPU/CUDA detection test."""
    import subprocess
    import sys

    console.print("[cyan]Testing GPU/CUDA availability...[/cyan]\n")
    result = subprocess.run([sys.executable, "test_gpu.py"], check=False)
    sys.exit(result.returncode)


def main_clean():
    """Clean consolidated output directory (experiments, logs, models, analysis)."""
    import shutil
    from pathlib import Path

    output_dir = Path("output")
    if output_dir.exists():
        console.print(
            f"[yellow]Cleaning consolidated structure: {output_dir}...[/yellow]"
        )
        for item in output_dir.iterdir():
            if item.name != "experiment_manifest.json":
                if item.is_dir():
                    shutil.rmtree(item)
                    console.print(f"  ️  Removed: {item.name}/")
                else:
                    item.unlink()
                    console.print(f"  ️  Removed: {item.name}")
        console.print("[green]✓ Cleaned consolidated output structure[/green]")
        console.print("[dim]  (experiments, logs, models, analysis all cleared)[/dim]")
    else:
        console.print("[yellow]No output directory to clean[/yellow]")


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


def main_archive():
    """Archive incomplete runs to clean manifest."""
    from src.workflows.experiment_manager import ExperimentManager
    
    manager = ExperimentManager()
    
    # Show stats before
    console.print("[cyan]Before archiving:[/cyan]")
    manager.print_manifest_stats()
    
    # Archive incomplete runs
    console.print()
    archived_count = manager.archive_incomplete_runs()
    
    # Show stats after
    if archived_count > 0:
        console.print()
        console.print("[cyan]After archiving:[/cyan]")
        manager.print_manifest_stats()


def main_stats():
    """Show manifest statistics."""
    from src.workflows.experiment_manager import ExperimentManager
    
    manager = ExperimentManager()
    manager.print_manifest_stats()


def main_interactive():
    """Interactive command launcher (TUI menu)."""
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.panel import Panel
    import subprocess

    commands = [
        # Mode A: Baseline
        (
            "mode-a",
            [
                ("a1", "baseline --test", "Mode A: Pure NSGA-II (~2 min, 30 gens)"),
                ("a2", "baseline --prod", "Mode A: Pure NSGA-II (~3-5 hrs, 2000 gens)"),
            ],
        ),
        # Mode B: Memetic
        (
            "mode-b",
            [
                ("b1", "memetic --test", "Mode B: + Memetic (~2 min, 30 gens)"),
                ("b2", "memetic --prod", "Mode B: + Memetic (~3-5 hrs, 2000 gens)"),
            ],
        ),
        # Mode C: Round-Robin
        (
            "mode-c",
            [
                ("c1", "roundrobin --test", "Mode C: + Round-robin (~3 min, 30 gens)"),
                ("c2", "roundrobin --prod", "Mode C: + Round-robin (~4-6 hrs, 2000 gens)"),
            ],
        ),
        # Mode D: Adaptive
        (
            "mode-d",
            [
                ("d1", "adaptive --test", "Mode D: + Adaptive (~3 min, 30 gens)"),
                ("d2", "adaptive --prod", "Mode D: + Adaptive (~4-6 hrs, 2000 gens)"),
            ],
        ),
        # Mode E: RL-Guided
        (
            "mode-e",
            [
                ("e1", "rl --test", "Mode E: + RL-guided (~3 min, 30 gens)"),
                ("e2", "rl --prod", "Mode E: + RL-guided (~4-6 hrs, 2000 gens)"),
                ("e3", "train-rl --test", "Train RL agent (10K steps, ~5 min)"),
                ("e4", "train-rl --prod", "Train RL agent (100K steps, ~1-2 hrs)"),
            ],
        ),
        # Utilities
        (
            "utilities",
            [
                ("u1", "analyze-results", "Generate comparison plots"),
                ("u2", "diagnose", "System diagnostics"),
                ("u3", "clean", "Clean output directory"),
                ("u4", "list-experiments", "List experiment history"),
            ],
        ),
    ]

    while True:
        console.clear()
        console.print(
            "\n[bold cyan]Schedule Engine - Progressive Experiments (A→E)[/bold cyan]\n"
        )

        for category, cmds in commands:
            # Category headers with descriptions
            if category == "mode-a":
                console.print("[bold magenta]MODE A: BASELINE[/bold magenta]")
                console.print("[dim]  Pure NSGA-II (no repairs, no heuristics)[/dim]")
            elif category == "mode-b":
                console.print("\n[bold magenta]MODE B: + MEMETIC[/bold magenta]")
                console.print("[dim]  NSGA-II + memetic local search[/dim]")
            elif category == "mode-c":
                console.print("\n[bold magenta]MODE C: + ROUND-ROBIN[/bold magenta]")
                console.print("[dim]  Fixed rotation (19 heuristics + 3 repairs)[/dim]")
            elif category == "mode-d":
                console.print("\n[bold magenta]MODE D: + ADAPTIVE[/bold magenta]")
                console.print("[dim]  Intelligent heuristic selection[/dim]")
            elif category == "mode-e":
                console.print("\n[bold magenta]MODE E: + RL-GUIDED[/bold magenta]")
                console.print("[dim]  RL learns optimal operator timing (24 heuristics)[/dim]")
            elif category == "utilities":
                console.print("\n[bold magenta]UTILITIES[/bold magenta]")

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
            try:
                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted - returning to menu[/yellow]")
        else:
            console.print(f"[red]Invalid choice: {choice}[/red]")
            try:
                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                console.print("\n[yellow]Returning to menu[/yellow]")


if __name__ == "__main__":
    # Auto-detect command from script name
    script_name = Path(sys.argv[0]).stem

    # Progressive experimental modes (A-E)
    if "baseline" in script_name:
        main_baseline()
    elif "memetic" in script_name:
        main_memetic()
    elif "roundrobin" in script_name:
        main_roundrobin()
    elif "adaptive" in script_name:
        main_adaptive()
    elif script_name == "rl":
        main_rl()
    # RL Training
    elif "train" in script_name and "rl" in script_name:
        main_train_rl()
    # Helper commands
    elif "test-gpu" in script_name or "test_gpu" in script_name:
        main_test_gpu()
    elif "diagnose" in script_name:
        main_diagnose()
    elif "clean" in script_name:
        main_clean()
    elif "list" in script_name:
        main_list()
    elif "launcher" in script_name or "interactive" in script_name or "run" in script_name:
        main_interactive()
    else:
        print(f"Unknown command: {script_name}")
        print("Use 'uv run launcher' to see all commands")
        sys.exit(1)
