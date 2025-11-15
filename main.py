"""
Schedule Engine Entry Point

Runs standard GA-based course scheduling workflow.
"""

import argparse
from datetime import datetime
from rich.console import Console
from src.workflows import run_standard_workflow
from src.utils.experiment import sanitize_experiment_name
from src.config import init_config

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
    parser = argparse.ArgumentParser(description="University Course Scheduling Engine")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (e.g., configs/custom.yaml)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["test", "notprod", "prod"],
        help="Environment: test (smoke test), notprod (medium), prod (best quality)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional experiment name to tag the output folder",
    )
    args = parser.parse_args()

    # Set environment variable if --env provided
    if args.env:
        import os

        os.environ["ENVIRONMENT"] = args.env

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

    # Load configuration
    try:
        config = init_config(args.config)
        console.print()
        console.print(config.summary())
        console.print()
    except Exception as e:
        console.print(f"[bold red][!err] failed to load config:[/bold red] {e}")
        return 1

    # Run workflow with config
    result = run_standard_workflow(
        pop_size=config.ga.pop_size,
        generations=config.ga.ngen,
        crossover_prob=config.ga.cxpb,
        mutation_prob=config.ga.mutpb,
        validate=True,  # Enable input validation
        config=config,  # Pass config object to workflow
        output_dir=(
            None
            if exp_name is None
            else f"output/evaluation_{{}}_{exp_name}".format(
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        ),
    )

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
    console.print()


def main_prod():
    """Entry point for production runs (uv run prod)"""
    import os

    os.environ["ENVIRONMENT"] = "prod"
    main()


def main_notprod():
    """Entry point for notprod runs (uv run notprod)"""
    import os

    os.environ["ENVIRONMENT"] = "notprod"
    main()


def main_test():
    """Entry point for test runs (uv run test)"""
    import os

    os.environ["ENVIRONMENT"] = "test"
    main()


if __name__ == "__main__":
    main()
