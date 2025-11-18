"""
Schedule Engine Entry Point

Runs standard GA-based course scheduling workflow with runtime mode support.
"""

import argparse
import time
from datetime import datetime
from rich.console import Console
from src.workflows import run_standard_workflow
from src.workflows.experiment_manager import ExperimentManager
from src.utils.experiment import sanitize_experiment_name
from src.config import init_config
from src.config.runtime_mode import RuntimeMode
from src.config.loader import load_config

console = Console()


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
    if exp_name is None:
        # Only prompt when running interactively (TTY). In CI / uv runs we'll skip prompt.
        import sys

        if sys.stdin and sys.stdin.isatty():
            try:
                raw = input("Experiment name (optional): ").strip()
                exp_name = raw if raw else None
            except Exception:
                exp_name = None
        else:
            exp_name = None

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


def main_prod():
    """Entry point for production runs (uv run prod)"""
    import os
    import sys

    os.environ["ENVIRONMENT"] = "prod"
    sys.exit(main() or 0)


def main_test():
    """Entry point for test runs (uv run test)"""
    import os
    import sys

    os.environ["ENVIRONMENT"] = "test"
    sys.exit(main() or 0)


# Runtime mode entry points for UV shortcuts
def main_baseline():
    """Entry point for baseline mode (uv run baseline)"""
    import sys

    sys.argv = ["main.py", "--mode", "baseline", "--env", "prod"]
    sys.exit(main() or 0)


def main_repairs():
    """Entry point for repairs mode (uv run repairs)"""
    import sys

    sys.argv = ["main.py", "--mode", "nsga-repairs", "--env", "prod"]
    sys.exit(main() or 0)


def main_heuristics():
    """Entry point for heuristics mode (uv run heuristics)"""
    import sys

    sys.argv = ["main.py", "--mode", "nsga-heuristics", "--env", "prod"]
    sys.exit(main() or 0)


def main_full():
    """Entry point for full mode (uv run full)"""
    import sys

    sys.argv = ["main.py", "--mode", "nsga-full", "--env", "prod"]
    sys.exit(main() or 0)


def main_rl():
    """Entry point for RL-guided mode (uv run rl)"""
    import sys

    sys.argv = ["main.py", "--mode", "rl-guided", "--env", "prod"]
    sys.exit(main() or 0)


def main_roundrobin():
    """Entry point for round-robin mode (uv run roundrobin)"""
    import sys

    sys.argv = ["main.py", "--mode", "roundrobin", "--env", "prod"]
    sys.exit(main() or 0)


if __name__ == "__main__":
    main()
