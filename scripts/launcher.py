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
    """Create argument parser with all commands."""
    parser = argparse.ArgumentParser(
        description="Schedule Engine - Unified CLI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Group A: Baseline Methods
  uv run baseline --test      # A1: Pure NSGA-II baseline (~2 min)
  uv run baseline --prod      # A1: Pure NSGA-II (2000 gens, ~3-5 hours)
  uv run repairs --test       # A2: NSGA + Repairs (~2 min) 
  uv run repairs --prod       # A2: NSGA + Repairs (full run)
  
  # Group B: GA Enhancement Methods  
  uv run heuristics --test    # B2: NSGA + 19 Heuristics (~3 min)
  uv run heuristics --prod    # B2: NSGA + Heuristics (full run)
  uv run full --test          # B3: Full GA with local search (~3 min)
  uv run full --prod          # B3: Full GA (full run)
  
  # Group C: Hyper-Heuristic Methods
  uv run roundrobin --test    # C1: Round-Robin selection (~3 min)
  uv run roundrobin --prod    # C1: Round-Robin (full run)
  uv run rl --test            # C2: RL-Guided selection (~3 min)
  uv run rl --prod            # C2: RL-Guided (full run)
  
  # RL Training
  uv run train-rl --test      # RL Training (500 steps, ~2-3 min)
  uv run train-rl --med       # RL Training (50K steps, ~30-45 min)  
  uv run train-rl --prod      # RL Training (100K steps, ~1-2 hours)
  
  # Analysis & Utilities
  uv run analyze-results      # Generate comparison analysis & plots
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
    parser.add_argument(
        "--repair-after-every-generation", 
        action="store_true", 
        help="Enable repair after every crossover/mutation (for comparison)"
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

    # Select appropriate config based on repair flag
    config_override = None
    if profile == "prod":
        if args.repair_after_every_generation:
            config_override = "configs/baseline/nsga-with-repairs-prod.yaml"
            console.print("[yellow]Using NSGA-II with repairs after every generation[/yellow]")
        else:
            config_override = "configs/baseline/pure-nsga-prod.yaml"
            console.print("[green]Using Pure NSGA-II baseline (NO repairs, NO heuristics)[/green]")
    
    # Import and run main with environment
    from main import main

    sys.argv = ["main.py", "--env", profile]
    if config_override:
        sys.argv.extend(["--config", config_override])
    elif args.config:
        sys.argv.extend(["--config", args.config])
    if args.name:
        sys.argv.extend(["--name", args.name])
    if args.mode:
        sys.argv.extend(["--mode", args.mode])

    sys.exit(main() or 0)


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


# Experimental Method Commands (baseline, repairs, heuristics, full, roundrobin, rl)
def main_baseline():
    """A1: Pure NSGA-II baseline (no repairs, no heuristics)."""
    parser = create_parser()
    args = parser.parse_args()
    
    profile = args.profile or "test"
    
    from main import main
    
    sys.argv = ["main.py", "--env", profile, "--mode", "baseline"]
    if args.name:
        sys.argv.extend(["--name", args.name])
    
    console.print(f"[green]Running A1: Pure NSGA-II baseline ({profile} profile)[/green]")
    sys.exit(main() or 0)


def main_repairs():
    """A2: NSGA-II + repairs after every generation."""
    parser = create_parser()
    args = parser.parse_args()
    
    profile = args.profile or "test"
    
    from main import main
    
    sys.argv = ["main.py", "--env", profile, "--mode", "nsga_repairs"]
    if args.name:
        sys.argv.extend(["--name", args.name])
    
    console.print(f"[green]Running A2: NSGA-II + repairs ({profile} profile)[/green]")
    sys.exit(main() or 0)


def main_heuristics():
    """B2: NSGA-II + repairs + 19 heuristics."""
    parser = create_parser()
    args = parser.parse_args()
    
    profile = args.profile or "test"
    
    from main import main
    
    sys.argv = ["main.py", "--env", profile, "--mode", "nsga_heuristics"]
    if args.name:
        sys.argv.extend(["--name", args.name])
    
    console.print(f"[green]Running B2: NSGA-II + heuristics ({profile} profile)[/green]")
    sys.exit(main() or 0)


def main_full():
    """B3: Full GA (repairs + heuristics + local search)."""
    parser = create_parser()
    args = parser.parse_args()
    
    profile = args.profile or "test"
    
    from main import main
    
    sys.argv = ["main.py", "--env", profile, "--mode", "nsga_full"]
    if args.name:
        sys.argv.extend(["--name", args.name])
    
    console.print(f"[green]Running B3: Full GA ({profile} profile)[/green]")
    sys.exit(main() or 0)


def main_roundrobin():
    """C1: Round-robin hyper-heuristic."""
    parser = create_parser()
    args = parser.parse_args()
    
    profile = args.profile or "test"
    
    from main import main
    
    sys.argv = ["main.py", "--env", profile, "--mode", "round_robin"]
    if args.name:
        sys.argv.extend(["--name", args.name])
    
    console.print(f"[green]Running C1: Round-robin hyper-heuristic ({profile} profile)[/green]")
    sys.exit(main() or 0)


def main_rl_guided():
    """C2: RL-guided hyper-heuristic."""
    parser = create_parser()
    args = parser.parse_args()
    
    profile = args.profile or "test"
    
    from main import main
    
    sys.argv = ["main.py", "--env", profile, "--mode", "rl_guided"]
    if args.name:
        sys.argv.extend(["--name", args.name])
    
    console.print(f"[green]Running C2: RL-guided hyper-heuristic ({profile} profile)[/green]")
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
        console.print(f"[yellow]Cleaning consolidated structure: {output_dir}...[/yellow]")
        for item in output_dir.iterdir():
            if item.name != "experiment_manifest.json":
                if item.is_dir():
                    shutil.rmtree(item)
                    console.print(f"  🗑️  Removed: {item.name}/")
                else:
                    item.unlink()
                    console.print(f"  🗑️  Removed: {item.name}")
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


def main_interactive():
    """Interactive command launcher (TUI menu)."""
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.panel import Panel
    import subprocess

    commands = [
        # Group A: Baseline Methods
        (
            "baseline",
            [
                ("1", "baseline --test", "A1: Pure NSGA-II (~2 min)"),
                ("2", "baseline --prod", "A1: Pure NSGA-II (full run)"),
                ("3", "repairs --test", "A2: NSGA + Repairs (~2 min)"),
                ("4", "repairs --prod", "A2: NSGA + Repairs (full run)"),
            ],
        ),
        # Group B: GA Enhancement Methods
        (
            "ga-enhancement",
            [
                ("5", "heuristics --test", "B2: NSGA + Heuristics (~3 min)"),
                ("6", "heuristics --prod", "B2: NSGA + Heuristics (full run)"),
                ("7", "full --test", "B3: Full GA (~3 min)"),
                ("8", "full --prod", "B3: Full GA (full run)"),
            ],
        ),
        # Group C: Hyper-Heuristic Methods
        (
            "hyper-heuristic",
            [
                ("9", "roundrobin --test", "C1: Round-Robin (~3 min)"),
                ("a", "roundrobin --prod", "C1: Round-Robin (full run)"),
                ("b", "rl --test", "C2: RL-Guided (~3 min)"),
                ("c", "rl --prod", "C2: RL-Guided (full run)"),
            ],
        ),
        # RL Training
        (
            "rl-training",
            [
                ("d", "train-rl --test", "RL Training (500 steps, ~2-3 min)"),
                ("e", "train-rl --med", "RL Training (50K steps, ~30-45 min)"),
                ("f", "train-rl --prod", "RL Training (100K steps, ~1-2 hrs)"),
            ],
        ),
        # Utilities
        (
            "utilities",
            [
                ("g", "analyze-results", "Generate analysis & plots"),
                ("h", "diagnose", "System diagnostics"),
                ("i", "clean", "Clean output directory"),
                ("j", "list-experiments", "List experiment history"),
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
            if category == "baseline":
                console.print("[bold magenta]Group A: Baseline Methods[/bold magenta]")
            elif category == "ga-enhancement":
                console.print("\n[bold magenta]Group B: GA Enhancement Methods[/bold magenta]")
            elif category == "hyper-heuristic":
                console.print("\n[bold magenta]Group C: Hyper-Heuristic Methods[/bold magenta]")
            elif category == "rl-training":
                console.print("\n[bold magenta]RL Training[/bold magenta]")
            elif category == "utilities":
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

    # Main GA commands
    if "nsga" in script_name:
        main_nsga()
    elif "train" in script_name and "rl" in script_name:
        main_train_rl()
    # Experimental method commands
    elif "baseline" in script_name:
        main_baseline()
    elif "repairs" in script_name:
        main_repairs()
    elif "heuristics" in script_name:
        main_heuristics()
    elif "full" in script_name:
        main_full()
    elif "roundrobin" in script_name:
        main_roundrobin()
    elif script_name == "rl":
        main_rl_guided()
    # Helper commands
    elif "test-gpu" in script_name or "test_gpu" in script_name:
        main_test_gpu()
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
