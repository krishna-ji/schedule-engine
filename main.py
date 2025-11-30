"""
Schedule Engine Entry Point

Runs standard GA-based course scheduling workflow with experiment configs.
"""

from __future__ import annotations

import argparse
import sys

# Import all experiments
from configs import experiment_a, experiment_b, experiment_c, experiment_d, experiment_e
from src.config.loader import load_config
from src.config.presets.profiles import Profile
from src.utils.console_service import get_console
from src.utils.experiment import sanitize_experiment_name
from src.utils.structured_logger import setup_logging
from src.workflows import run_standard_workflow
from src.workflows.experiment_manager import ExperimentManager

console = get_console()

# Experiment registry
EXPERIMENTS = {
    "a": ("Experiment A: Pure NSGA-II", experiment_a),
    "b": ("Experiment B: Memetic NSGA-II", experiment_b),
    "c": ("Experiment C: Round-Robin Heuristics", experiment_c),
    "d": ("Experiment D: Adaptive Selection", experiment_d),
    "e": ("Experiment E: RL-Guided", experiment_e),
    "baseline": ("Experiment A: Pure NSGA-II", experiment_a),
    "memetic": ("Experiment B: Memetic NSGA-II", experiment_b),
    "roundrobin": ("Experiment C: Round-Robin Heuristics", experiment_c),
    "adaptive": ("Experiment D: Adaptive Selection", experiment_d),
    "rl": ("Experiment E: RL-Guided", experiment_e),
}


def list_experiments() -> str:
    """Generate formatted list of available experiments."""
    lines = ["Available Experiments:", ""]

    seen = set()
    for key in ["a", "b", "c", "d", "e"]:
        if key in seen:
            continue
        seen.add(key)
        name, blueprint = EXPERIMENTS[key]
        lines.append(f"  {key}")
        lines.append(f"    {name}")
        lines.append(f"    {blueprint.description}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """
    Execute standard scheduling workflow.

    Pipeline:
        1. Build configuration via experiment blueprint + profile
        2. Load input data from data/
        3. Validate input for consistency
        4. Check feasibility (optional)
        5. Run NSGA-II genetic algorithm (with optional parallelization)
        6. Export best schedule to output/
        7. Generate evolution plots and reports

    Results:
        Saved to output/<experiment-name>/evaluation_<timestamp>/
    """

    setup_logging(
        log_file=None,
        console_level="DEBUG",
        file_level="DEBUG",
        show_time=True,
        show_path=False,
    )

    parser = argparse.ArgumentParser(
        description="University Course Scheduling Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=list_experiments(),
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=False,
        help="Experiment ID (a, b, c, d, e) or alias (baseline, memetic, roundrobin, adaptive, rl)",
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        type=str,
        choices=["test", "prod", "debug"],
        help="Profile selector: test (smoke) or prod (full)",
    )
    profile_group.add_argument(
        "--test",
        action="store_const",
        const="test",
        dest="profile",
        help="Shortcut for --profile test",
    )
    profile_group.add_argument(
        "--prod",
        action="store_const",
        const="prod",
        dest="profile",
        help="Shortcut for --profile prod",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Custom experiment name (overrides auto-generated name)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments and exit",
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        console.print(list_experiments())
        return 0

    # Validate experiment
    if not args.experiment:
        console.print("[red]Error:[/red] --experiment/-e is required")
        console.print("\nUse --list to see available experiments")
        return 1

    experiment_key = args.experiment.lower()
    if experiment_key not in EXPERIMENTS:
        console.print(f"[red]Error:[/red] Unknown experiment: {args.experiment}")
        console.print("\nAvailable experiments: a, b, c, d, e")
        console.print("Use --list for details")
        return 1

    experiment_name, blueprint = EXPERIMENTS[experiment_key]

    # Determine profile
    profile_str = args.profile or "test"
    profile = Profile(profile_str)

    # Build config
    console.print(f"[cyan]Experiment:[/cyan] {experiment_name}")
    console.print(f"[cyan]Profile:[/cyan] {profile.value.upper()}")
    console.print(f"[cyan]Blueprint:[/cyan] {blueprint.name}")
    console.print()

    config = load_config(blueprint, profile)

    # Initialize global config for modules that use get_config()
    from src.config import init_config

    init_config(config_obj=config)

    # Experiment manager
    manager = ExperimentManager()

    # Use custom name or generate from experiment key
    if args.name:
        exp_name = sanitize_experiment_name(args.name)
    else:
        exp_name = f"experiment_{experiment_key}_{profile.value}"

    # Create output directory
    output_dir = manager.create_output_dir(
        runtime_mode=experiment_key,
        experiment_name=exp_name,
    )
    console.print(f"[cyan]Output Directory:[/cyan] {output_dir}")
    console.print()

    # Register experiment
    experiment_run = manager.register_run(
        runtime_mode=experiment_key,
        config_reference=f"{blueprint.name}:{profile.value}",
        output_path=output_dir,
        experiment_name=exp_name,
    )

    # Run workflow
    try:
        result = run_standard_workflow(
            pop_size=config.ga.pop_size,
            generations=config.ga.ngen,
            config=config,
            output_dir=str(output_dir),
        )

        if result["best_individual"] is None:
            console.print("[yellow]Warning:[/yellow] No valid solution found")
            return 1

        # Update run with results
        manager.update_run_results(
            run=experiment_run,
            best_hard_violations=float(result["best_individual"].fitness.values[0]),
            best_soft_penalty=float(result["best_individual"].fitness.values[1]),
            generations=config.ga.ngen,
            population_size=config.ga.pop_size,
        )

        console.print()
        console.print("[green]✓ Scheduling complete![/green]")
        console.print(f"[cyan]Results saved to:[/cyan] {output_dir}")
        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
