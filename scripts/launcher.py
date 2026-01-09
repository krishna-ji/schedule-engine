#!/usr/bin/env python3
"""
Unified CLI Launcher for Schedule Engine

Convention:
- Main commands: 0-99+ (nsga, train-rl, curriculum, etc.)
- Helper commands: a-z (diagnose, clean, test-gpu, etc.)
     - Profiles: --test, --prod
     - Configs: DRY hierarchy (test < prod)
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from configs.profiles import Profile  # noqa: E402

console = Console()

PROFILE_DESCRIPTIONS: dict[Profile, str] = {
    Profile.TEST: "Smoke test (~30 gens, 10 pop, 2-5 min)",
    Profile.PROD: "Production (~2000 gens, 400 pop, multi-hour)",
}

PROFILE_CHOICES = tuple(profile.value for profile in Profile)
DEFAULT_PROFILE = Profile.TEST


def _resolve_profile(value: str | None) -> Profile:
    """Validate profile value and fallback to default when missing."""

    try:
        return Profile.from_string(value or DEFAULT_PROFILE.value)
    except ValueError as exc:
        console.print(f"[bold red][!err] {exc}[/bold red]")
        sys.exit(1)


def _profile_banner(profile: Profile) -> str:
    """Format profile description for console output."""

    description = PROFILE_DESCRIPTIONS.get(profile, "")
    return f"{profile.value} ({description})"


def create_parser() -> argparse.ArgumentParser:
    """Create shared CLI parser with unified profile/config flags."""

    parser = argparse.ArgumentParser(
        description="Schedule Engine",
        add_help=False,  # No --help drama
    )

    # Profile selection (explicit flag + shorthand toggles)
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        help="Profile selector: test (smoke) or prod (full)",
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--test", action="store_const", const="test", dest="profile"
    )
    profile_group.add_argument(
        "--prod", action="store_const", const="prod", dest="profile"
    )

    # Options shared across commands
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--repair-after-every-generation", action="store_true")
    parser.add_argument("--name")
    parser.add_argument(
        "--mode",
        help=(
            "Experiment selector (a=baseline, b=memetic, c=roundrobin, d=adaptive, e=rl). "
            "Also accepts legacy names (nsga-full, baseline, adaptive, rl, etc.)"
        ),
    )

    return parser


def main_train_rl():
    """Launch RL training script with unified profile handling."""
    parser = create_parser()
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)
    profile_value = profile.value

    # Ensure downstream config loader uses the matching environment profile.
    os.environ["ENVIRONMENT"] = profile_value

    console.print(
        "[dim]ENVIRONMENT set to[/dim] "
        f"{profile_value} [dim]for RL profile[/dim] {_profile_banner(profile)}"
    )
    console.print(f"[green]Mode E: RL training ({_profile_banner(profile)})[/green]")
    console.print(
        "[dim]  Train PPO/DQN agents with optional curriculum (add --curriculum).[/dim]"
    )

    # Import RL training
    from src.rl.training.train_script import main as rl_main

    # Build argv for RL training
    sys.argv = ["train_script.py", "--profile", profile_value, "--agent", "ppo"]

    # Add curriculum flag if requested (curriculum works with any profile)
    if args.curriculum:
        sys.argv.append("--curriculum")
    else:
        # Map profile to timesteps (if not using curriculum)
        timestep_map = {"test": 10_000, "prod": 100_000}
        sys.argv.extend(["--timesteps", str(timestep_map[profile_value])])

    exit_code = rl_main() or 0

    if exit_code == 0:
        console.print("[green]RL training finished successfully.[/green]")
        console.print(
            f"[dim]Promote the desired checkpoint (see scripts/training/promote_model_to_prod.py) "
            f"and launch the GA via Mode F: 'uv run rl --{profile.value}'.[/dim]"
        )

    sys.exit(exit_code)


# ==================
# PROGRESSIVE MODE EXPERIMENTS (A→E: Increasing Complexity)
# ==================
# Mode A: Pure NSGA-II                           [repairs: NO,  memetic: NO,  heuristics: NO]
# Mode B: + Memetic local search                 [repairs: YES, memetic: YES, heuristics: NO]
# Mode C: + Round-robin (heuristics + repair)    [repairs: YES (round-robin), memetic: NO, heuristics: YES (fixed)]
# Mode D: + Adaptive heuristics                  [repairs: YES, memetic: NO,  heuristics: YES (adaptive)]
# Mode E: RL training (curriculum-ready)         [train PPO/DQN agents, optional curriculum stages]
# Mode F: RL-guided inference (deploy agents)    [repairs: YES, memetic: YES, heuristics: YES (RL-controlled)]


def main_baseline():
    """Run Mode A (pure NSGA-II) experiment with profile-aware config."""
    parser = create_parser()
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)

    from main import main

    sys.argv = [
        "main.py",
        "--experiment",
        "a",
        "--profile",
        profile.value,
    ]
    if args.name:
        sys.argv.extend(["--name", args.name])

    console.print(
        f"[green]Mode A: Pure NSGA-II baseline ({_profile_banner(profile)})[/green]"
    )
    console.print("[dim]  [repairs: NO, memetic: NO, heuristics: NO][/dim]")
    sys.exit(main() or 0)


def main_memetic():
    """Run Mode B experiment (NSGA-II + memetic local search)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)

    from main import main

    sys.argv = [
        "main.py",
        "--experiment",
        "b",
        "--profile",
        profile.value,
    ]
    if args.name:
        sys.argv.extend(["--name", args.name])

    console.print(
        f"[green]Mode B: NSGA-II + memetic local search ({_profile_banner(profile)})[/green]"
    )
    console.print("[dim]  [repairs: YES, memetic: YES, heuristics: NO][/dim]")
    sys.exit(main() or 0)


def main_roundrobin():
    """Run Mode C experiment (round-robin heuristics + repair)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)

    from main import main

    sys.argv = [
        "main.py",
        "--experiment",
        "c",
        "--profile",
        profile.value,
    ]
    if args.name:
        sys.argv.extend(["--name", args.name])

    console.print(
        f"[green]Mode C: Round-robin heuristics + repair ({_profile_banner(profile)})[/green]"
    )
    console.print(
        "[dim]  [repairs: YES (round-robin), memetic: NO, heuristics: YES (fixed rotation)][/dim]"
    )
    sys.exit(main() or 0)


def main_adaptive():
    """Run Mode D experiment (adaptive heuristic selection)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)

    from main import main

    sys.argv = [
        "main.py",
        "--experiment",
        "d",
        "--profile",
        profile.value,
    ]
    if args.name:
        sys.argv.extend(["--name", args.name])

    console.print(
        f"[green]Mode D: Adaptive heuristic selection ({_profile_banner(profile)})[/green]"
    )
    console.print(
        "[dim]  [repairs: YES, memetic: NO, heuristics: YES (adaptive selection)][/dim]"
    )
    sys.exit(main() or 0)


def main_rl():
    """Run Mode E experiment (RL-guided inference run)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)

    from main import main

    sys.argv = [
        "main.py",
        "--experiment",
        "e",
        "--profile",
        profile.value,
    ]
    if args.name:
        sys.argv.extend(["--name", args.name])

    console.print(
        f"[green]Mode E: RL-guided inference ({_profile_banner(profile)})[/green]"
    )
    console.print(
        "[dim]  Deploy trained agents for heuristic control (repairs+memetic+RL).[/dim]"
    )
    console.print(
        "[dim]  Update rl.agent.model_path in configs/experiment_e_rl_guided.py before running to point at your promoted model.[/dim]"
    )
    sys.exit(main() or 0)


def main_heuristic_testing():
    """Run Mode F experiment (individual heuristic testing)."""
    parser = create_parser()
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)

    # Load config to detect enabled heuristic
    from configs.experiment_f_heuristic_testing import get_config
    from configs.profiles import Profile as ConfigProfile

    config_profile = (
        ConfigProfile.TEST if profile.value == "test" else ConfigProfile.PROD
    )
    test_config = get_config(config_profile)

    # Extract enabled heuristic name (if exactly one is enabled)
    heuristic_name = test_config.get_enabled_heuristic_name()

    # Auto-generate name with heuristic suffix if not provided
    final_name = args.name
    if heuristic_name and not final_name:
        final_name = f"test-{heuristic_name}"
        console.print(f"[dim]Auto-detected heuristic: {heuristic_name}[/dim]")

    from main import main

    sys.argv = [
        "main.py",
        "--experiment",
        "f",
        "--profile",
        profile.value,
    ]
    if final_name:
        sys.argv.extend(["--name", final_name])

    console.print(
        f"[green]Mode F: Individual Heuristic Testing ({_profile_banner(profile)})[/green]"
    )
    if heuristic_name:
        console.print(f"[dim]  Testing: {heuristic_name}[/dim]")

    sys.exit(main() or 0)


def main_nsga():
    """Unified NSGA-II launcher with experiment selection."""

    parser = create_parser()
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)

    from main import main

    sys.argv = ["main.py"]

    # Default to experiment D (adaptive) if no mode specified
    experiment = args.mode or "d"

    # Map legacy mode names to experiments
    mode_map = {
        "nsga-full": "d",
        "mode-a": "a",
        "baseline": "a",
        "mode-b": "b",
        "memetic": "b",
        "mode-c": "c",
        "roundrobin": "c",
        "mode-d": "d",
        "adaptive": "d",
        "mode-e": "e",
        "rl": "e",
        "rl-guided": "e",
    }

    experiment = mode_map.get(experiment.lower(), experiment)

    sys.argv.extend(["--experiment", experiment])
    sys.argv.extend(["--profile", profile.value])

    if args.name:
        sys.argv.extend(["--name", args.name])

    experiment_names = {
        "a": "Experiment A: Pure NSGA-II",
        "b": "Experiment B: Memetic",
        "c": "Experiment C: Round-robin",
        "d": "Experiment D: Adaptive",
        "e": "Experiment E: RL-guided",
    }

    descriptor = experiment_names.get(experiment, f"experiment={experiment}")
    console.print(
        f"[green]NSGA-II launcher ({_profile_banner(profile)}) — {descriptor}[/green]"
    )

    sys.exit(main() or 0)


# Helper commands (a-z)
def main_diagnose():
    """System diagnostics."""
    from scripts.diagnostics.diagnose_gpu import main as diagnose_gpu

    diagnose_gpu()


def main_test_gpu():
    """Run lightweight CUDA sanity checks without full diagnostics."""

    from scripts.diagnostics import diagnose_gpu

    console.print("[cyan]Running quick GPU sanity check...[/cyan]\n")
    cuda_ok = diagnose_gpu.check_cuda_availability()
    ops_ok = diagnose_gpu.test_gpu_operations() if cuda_ok else False

    sys.exit(0 if cuda_ok and ops_ok else 1)


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
    import json
    from pathlib import Path

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
    import subprocess

    from rich.prompt import Prompt

    commands = [
        # Direct NSGA launcher (experiment-aware entry point)
        (
            "nsga-launcher",
            [
                (
                    "n1",
                    "nsga --test",
                    "NSGA launcher (default: adaptive) (~3 min, 30 gens)",
                ),
                (
                    "n2",
                    "nsga --prod",
                    "NSGA launcher (default: adaptive) (~4-6 hrs, 2000 gens)",
                ),
            ],
        ),
        # Experiment A: Baseline
        (
            "experiment-a",
            [
                (
                    "a1",
                    "baseline --test",
                    "Experiment A: Pure NSGA-II (~2 min, 30 gens)",
                ),
                (
                    "a2",
                    "baseline --prod",
                    "Experiment A: Pure NSGA-II (~3-5 hrs, 2000 gens)",
                ),
            ],
        ),
        # Experiment B: Memetic
        (
            "experiment-b",
            [
                ("b1", "memetic --test", "Experiment B: + Memetic (~2 min, 30 gens)"),
                (
                    "b2",
                    "memetic --prod",
                    "Experiment B: + Memetic (~3-5 hrs, 2000 gens)",
                ),
            ],
        ),
        # Experiment C: Round-Robin
        (
            "experiment-c",
            [
                (
                    "c1",
                    "roundrobin --test",
                    "Experiment C: + Round-robin (~3 min, 30 gens)",
                ),
                (
                    "c2",
                    "roundrobin --prod",
                    "Experiment C: + Round-robin (~4-6 hrs, 2000 gens)",
                ),
            ],
        ),
        # Experiment D: Adaptive
        (
            "experiment-d",
            [
                ("d1", "adaptive --test", "Experiment D: + Adaptive (~3 min, 30 gens)"),
                (
                    "d2",
                    "adaptive --prod",
                    "Experiment D: + Adaptive (~4-6 hrs, 2000 gens)",
                ),
            ],
        ),
        # Experiment E: RL Training (with curriculum options)
        (
            "rl-training",
            [
                ("e1", "train-rl --test", "RL training (10K steps, ~5-10 min)"),
                ("e2", "train-rl --prod", "RL training (100K steps, ~1-2 hrs)"),
                (
                    "e3",
                    "train-rl --test --curriculum",
                    "RL curriculum smoke (multi-stage, ~15 min)",
                ),
                (
                    "e4",
                    "train-rl --prod --curriculum",
                    "RL curriculum full run (~2-3 hrs)",
                ),
            ],
        ),
        # Experiment E: RL-guided inference
        (
            "experiment-e",
            [
                ("f1", "rl --test", "Experiment E: RL-guided GA (~3 min, 30 gens)"),
                ("f2", "rl --prod", "Experiment E: RL-guided GA (~4-6 hrs, 2000 gens)"),
            ],
        ),
        # RL Inference with Latest Model
        (
            "rl-inference",
            [
                ("i1", "rl-inference --test", "Run latest model (smoke test)"),
                ("i2", "rl-inference --prod", "Run latest model (production)"),
                ("i3", "rl-inference --list-only", "List available models"),
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
        # Isolated Heuristic Testing
        (
            "heuristic-testing",
            [
                # Construction Heuristics (3)
                (
                    "h1",
                    "heuristic-testing --test --name test-largest-degree-first",
                    "Construction: largest_degree_first",
                ),
                (
                    "h2",
                    "heuristic-testing --test --name test-most-constrained-first",
                    "Construction: most_constrained_first",
                ),
                (
                    "h3",
                    "heuristic-testing --test --name test-earliest-deadline-first",
                    "Construction: earliest_deadline_first",
                ),
                # Perturbation Heuristics (5)
                (
                    "h4",
                    "heuristic-testing --test --name test-random-swap",
                    "Perturbation: random_swap",
                ),
                (
                    "h5",
                    "heuristic-testing --test --name test-temporal-shift",
                    "Perturbation: temporal_shift",
                ),
                (
                    "h6",
                    "heuristic-testing --test --name test-room-shuffle",
                    "Perturbation: room_shuffle",
                ),
                (
                    "h7",
                    "heuristic-testing --test --name test-instructor-reassign",
                    "Perturbation: instructor_reassign",
                ),
                (
                    "h8",
                    "heuristic-testing --test --name test-multi-perturbation",
                    "Perturbation: multi_perturbation",
                ),
                # Improvement Heuristics (3)
                (
                    "h9",
                    "heuristic-testing --test --name test-kempe-chain",
                    "Improvement: kempe_chain",
                ),
                (
                    "h10",
                    "heuristic-testing --test --name test-ejection-chain",
                    "Improvement: ejection_chain",
                ),
                (
                    "h11",
                    "heuristic-testing --test --name test-variable-depth-search",
                    "Improvement: variable_depth_search",
                ),
                # Diversity Heuristics (4)
                (
                    "h12",
                    "heuristic-testing --test --name test-distance-preserving-crossover",
                    "Diversity: distance_preserving_crossover",
                ),
                (
                    "h13",
                    "heuristic-testing --test --name test-crowding-mutation",
                    "Diversity: crowding_mutation",
                ),
                (
                    "h14",
                    "heuristic-testing --test --name test-niching-selection",
                    "Diversity: niching_selection",
                ),
                (
                    "h15",
                    "heuristic-testing --test --name test-adaptive-diversity-maintenance",
                    "Diversity: adaptive_diversity_maintenance",
                ),
                # Meta Heuristics (4)
                (
                    "h16",
                    "heuristic-testing --test --name test-variable-neighborhood-descent",
                    "Meta: variable_neighborhood_descent",
                ),
                (
                    "h17",
                    "heuristic-testing --test --name test-iterated-local-search",
                    "Meta: iterated_local_search",
                ),
                (
                    "h18",
                    "heuristic-testing --test --name test-adaptive-large-neighborhood",
                    "Meta: adaptive_large_neighborhood",
                ),
                (
                    "h19",
                    "heuristic-testing --test --name test-guided-local-search",
                    "Meta: guided_local_search",
                ),
                # Repair Heuristics (6)
                (
                    "h20",
                    "heuristic-testing --test --name test-exhaustive-repair",
                    "Repair: exhaustive_repair",
                ),
                (
                    "h21",
                    "heuristic-testing --test --name test-greedy-repair",
                    "Repair: greedy_repair",
                ),
                (
                    "h22",
                    "heuristic-testing --test --name test-igls-repair",
                    "Repair: igls_repair",
                ),
                (
                    "h23",
                    "heuristic-testing --test --name test-lns-repair",
                    "Repair: lns_repair",
                ),
                (
                    "h24",
                    "heuristic-testing --test --name test-memetic-repair",
                    "Repair: memetic_repair",
                ),
                (
                    "h25",
                    "heuristic-testing --test --name test-selective-repair",
                    "Repair: selective_repair",
                ),
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
            if category == "nsga-launcher":
                console.print("[bold magenta]NSGA LAUNCHER[/bold magenta]")
                console.print(
                    "[dim]  Direct NSGA entry — pass any experiment via --mode[/dim]"
                )
            elif category == "experiment-a":
                console.print("[bold magenta]EXPERIMENT A: BASELINE[/bold magenta]")
                console.print("[dim]  Pure NSGA-II (no repairs, no heuristics)[/dim]")
            elif category == "experiment-b":
                console.print("\n[bold magenta]EXPERIMENT B: + MEMETIC[/bold magenta]")
                console.print("[dim]  NSGA-II + memetic local search[/dim]")
            elif category == "experiment-c":
                console.print(
                    "\n[bold magenta]EXPERIMENT C: + ROUND-ROBIN[/bold magenta]"
                )
                console.print("[dim]  Fixed rotation (19 heuristics + 3 repairs)[/dim]")
            elif category == "experiment-d":
                console.print("\n[bold magenta]EXPERIMENT D: + ADAPTIVE[/bold magenta]")
                console.print("[dim]  Intelligent heuristic selection[/dim]")
            elif category == "rl-training":
                console.print("\n[bold magenta]RL TRAINING[/bold magenta]")
                console.print(
                    "[dim]  Train PPO/DQN agents; add --curriculum for staged runs[/dim]"
                )
            elif category == "experiment-e":
                console.print("\n[bold magenta]EXPERIMENT E: RL-GUIDED[/bold magenta]")
                console.print(
                    "[dim]  Deploy trained agents for heuristic control[/dim]"
                )
            elif category == "rl-inference":
                console.print("\n[bold magenta]RL INFERENCE[/bold magenta]")
                console.print("[dim]  Run RL-guided GA with latest trained model[/dim]")
            elif category == "utilities":
                console.print("\n[bold magenta]UTILITIES[/bold magenta]")
            elif category == "heuristic-testing":
                console.print(
                    "\n[bold magenta]ISOLATED HEURISTIC TESTING[/bold magenta]"
                )
                console.print(
                    "[dim]  Test individual heuristics in isolation (edit configs/experiments/heuristic_testing.py first)[/dim]"
                )

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

    # Progressive experimental modes (A-F)
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
    elif (
        "launcher" in script_name
        or "interactive" in script_name
        or "run" in script_name
    ):
        main_interactive()
    else:
        print(f"Unknown command: {script_name}")
        print("Use 'uv run launcher' to see all commands")
        sys.exit(1)
