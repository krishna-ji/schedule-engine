"""
Schedule Engine Entry Point

Runs standard GA-based course scheduling workflow with runtime mode support.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from src.workflows import run_standard_workflow
from src.workflows.experiment_manager import ExperimentManager
from src.utils.experiment import sanitize_experiment_name
from src.config import init_config
from src.config.runtime_mode import RuntimeMode
from src.config.loader import load_config
from src.utils.console_service import get_console
from src.utils.structured_logger import setup_logging

console = get_console()


def main():
    """
    Execute standard scheduling workflow.

    Pipeline:
        1. Load configuration from YAML (configs/test|dev|prod.yaml)
        2. Load input data from data/
        3. Validate input for consistency
        4. Check feasibility (optional)
        5. Run NSGA-II genetic algorithm (with optional parallelization)
        6. Export best schedule to output/
        7. Generate evolution plots and reports

    Configuration:
        Loaded from YAML files in configs/ directory.
        Use --env {test|dev|prod} to select configuration.

    Results:
        Saved to output/evaluation_<timestamp>/

    Parallelization:
        Controlled by parallel.use_multiprocessing in YAML config.
        Provides 3-6x speedup on multi-core systems.
    """
    # Initialize structured logging system (console + file)
    log_dir = Path("logs") / "nsga"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nsga_{timestamp}.log"

    setup_logging(
        log_file=log_file,
        console_level="DEBUG",
        file_level="DEBUG",
        show_time=True,
        show_path=False,
    )

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="University Course Scheduling Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RuntimeMode.list_modes(),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in RuntimeMode]
        + ["baseline", "repairs", "heuristics", "full", "rl", "roundrobin"],
        help="Runtime mode: baseline, nsga-repairs, nsga-heuristics, nsga-full, rl-guided, roundrobin",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (overrides --mode)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["test", "prod"],
        help="Environment: test (smoke test), prod (best quality)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional experiment name to tag the output folder",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List all available runtime modes and exit",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show runtime mode comparison table and exit",
    )
    args = parser.parse_args()

    # Handle --list-modes
    if args.list_modes:
        console.print(RuntimeMode.list_modes())
        return 0

    # Handle --compare
    if args.compare:
        manager = ExperimentManager()
        table = manager.compare_modes()
        console.print(table)
        return 0

    # Set environment variable if --env provided
    if args.env:
        import os

        os.environ["ENVIRONMENT"] = args.env

    # Parse runtime mode
    runtime_mode = None
    if args.mode:
        try:
            runtime_mode = RuntimeMode.from_string(args.mode)
            console.print(f"[cyan]Runtime Mode:[/cyan] {runtime_mode.display_name}")
            console.print(f"[dim]{runtime_mode.description}[/dim]")
            console.print()
        except ValueError as e:
            console.print(f"[bold red][!err] {e}[/bold red]")
            return 1

    # Experiment name: interactive prompt fallback
    exp_name = args.experiment
    # Use automatic experiment naming - no interactive prompt needed
    if exp_name is None:
        exp_name = "auto_generated"
    
    exp_name = sanitize_experiment_name(exp_name)

    # Initialize experiment manager
    manager = ExperimentManager()

    # Load configuration (with runtime mode support)
    try:
        if args.config:
            # Explicit config overrides runtime mode
            config = init_config(args.config)
        elif runtime_mode:
            # Load config for runtime mode
            config = load_config(runtime_mode=runtime_mode)
        else:
            # Default config
            config = init_config(None)

        console.print()
        console.print(config.summary())
        console.print()
    except Exception as e:
        console.print(f"[bold red][!err] failed to load config:[/bold red] {e}")
        return 1

    # Create output directory with runtime mode structure
    if runtime_mode:
        output_dir = manager.create_output_dir(runtime_mode, exp_name)
        console.print(f"[cyan]Output Directory:[/cyan] {output_dir}")
        console.print()
    else:
        output_dir = (
            None
            if exp_name is None
            else f"output/evaluation_{{}}_{exp_name}".format(
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        )

    # Register experiment run
    experiment_run = None
    if runtime_mode:
        experiment_run = manager.register_run(
            runtime_mode=runtime_mode,
            config_path=runtime_mode.config_path,
            output_path=output_dir,
            experiment_name=exp_name,
            seed=69,
        )

    # Track start time
    start_time = time.time()

    # Run workflow with config
    result = run_standard_workflow(
        pop_size=config.ga.pop_size,
        generations=config.ga.ngen,
        crossover_prob=config.ga.cxpb,
        mutation_prob=config.ga.mutpb,
        validate=True,  # Enable input validation
        config=config,  # Pass config object to workflow
        output_dir=str(output_dir) if output_dir else None,
    )

    # Track end time
    duration_seconds = time.time() - start_time

    # Print final summary with beautiful rich formatting
    console.print()
    console.print("[bold cyan]results[/bold cyan]")
    console.print()

    hard_viol = result["best_individual"].fitness.values[0]
    soft_pen = result["best_individual"].fitness.values[1]

    if hard_viol == 0:
        console.print("[green][!ok] perfect schedule (no hard violations)[/green]")
    else:
        console.print(f"[yellow][!warn] hard violations:[/yellow] {hard_viol:.0f}")

    console.print(f"  [dim]soft penalty:[/dim] {soft_pen:.2f}")
    console.print(f"  [dim]sessions:[/dim] {len(result['decoded_schedule'])}")
    console.print(f"  [dim]output:[/dim] {result['output_path']}")
    console.print(f"  [dim]runtime:[/dim] {duration_seconds:.1f}s")
    console.print()

    # Update experiment run with results
    if experiment_run and runtime_mode:
        manager.update_run_results(
            run=experiment_run,
            duration_seconds=duration_seconds,
            generations=config.ga.ngen,
            population_size=config.ga.pop_size,
            best_hard_violations=abs(hard_viol),
            best_soft_penalty=soft_pen,
        )
        console.print(
            f"[dim]Experiment logged: {experiment_run.run_id} ({runtime_mode.display_name})[/dim]"
        )
        console.print()


# Removed timed_input function - no longer needed with automatic experiment naming


def _create_env_entry_point(env: str):
    """Factory for environment entry points."""

    def entry_point():
        import os
        import sys

        os.environ["ENVIRONMENT"] = env
        sys.exit(main() or 0)

    return entry_point


def _create_mode_entry_point(mode: str, env: str = "prod"):
    """Factory for runtime mode entry points."""

    def entry_point():
        import sys

        # Preserve user arguments (e.g. --experiment name)
        # Filter out 'uv run' artifacts if present, though usually sys.argv[1:] is enough
        user_args = sys.argv[1:]

        # Construct new argv: script + forced args + user args
        sys.argv = ["main.py", "--mode", mode, "--env", env] + user_args
        sys.exit(main() or 0)

    return entry_point


# Environment entry points
main_prod = _create_env_entry_point("prod")
main_prod.__doc__ = "Entry point for production runs (uv run prod)"

main_test = _create_env_entry_point("test")
main_test.__doc__ = "Entry point for test runs (uv run test)"


# Runtime mode entry points (Modes 1-10)
_MODE_MAPPING = {
    "baseline": (
        RuntimeMode.BASELINE.value,
        "Mode 1: Pure NSGA-II (uv run baseline)",
    ),
    "repairs": (
        RuntimeMode.NSGA_REPAIRS.value,
        "Mode 2: NSGA-II + repairs (uv run repairs)",
    ),
    "heuristics": (
        RuntimeMode.NSGA_HEURISTICS.value,
        "Mode 3: NSGA-II + repairs + heuristics (uv run heuristics)",
    ),
    "full": (
        RuntimeMode.NSGA_FULL.value,
        "Mode 4: Full GA (uv run full)",
    ),
    "rl": (
        RuntimeMode.RL_GUIDED.value,
        "Mode 5: RL-guided heuristics (uv run rl)",
    ),
    "roundrobin": (
        RuntimeMode.ROUND_ROBIN.value,
        "Mode 6: Round-robin (uv run roundrobin)",
    ),
    "specialists": (
        RuntimeMode.RL_SPECIALISTS.value,
        "Mode 7: RL specialists (uv run specialists)",
    ),
    "archive": (
        RuntimeMode.ARCHIVE_DIVERSITY.value,
        "Mode 8: Archive diversity (uv run archive)",
    ),
    "hierarchical": (
        RuntimeMode.RL_HIERARCHICAL.value,
        "Mode 9: Hierarchical RL (uv run hierarchical)",
    ),
    "multiagent": (
        RuntimeMode.RL_MULTIAGENT.value,
        "Mode 10: Multi-agent RL (uv run multiagent)",
    ),
}

# Generate entry points dynamically
for func_name, (mode_value, doc) in _MODE_MAPPING.items():
    # Default (Prod)
    entry_point = _create_mode_entry_point(mode_value, env="prod")
    entry_point.__doc__ = doc
    globals()[f"main_{func_name}"] = entry_point

    # Explicit Prod
    entry_point_prod = _create_mode_entry_point(mode_value, env="prod")
    entry_point_prod.__doc__ = f"{doc} (Prod)"
    globals()[f"main_{func_name}_prod"] = entry_point_prod

    # Explicit Test
    entry_point_test = _create_mode_entry_point(mode_value, env="test")
    entry_point_test.__doc__ = f"{doc} (Test)"
    globals()[f"main_{func_name}_test"] = entry_point_test


if __name__ == "__main__":
    main()
