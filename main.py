"""
Schedule Engine Entry Point

Runs standard GA-based course scheduling workflow.
"""

import argparse
from rich.console import Console
from rich.panel import Panel
from src.workflows import run_standard_workflow
from config import init_config

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
        choices=["test", "dev", "prod"],
        help="Environment: test (fast), dev (medium), prod (full quality)",
    )
    args = parser.parse_args()

    # Determine config path
    config_path = args.config
    if args.env:
        config_path = f"configs/{args.env}.yaml"

    # Load configuration
    try:
        config = init_config(config_path)
        console.print()
        console.print(
            Panel(config.summary(), title="Configuration", border_style="cyan")
        )
        console.print()
    except Exception as e:
        console.print(f"[bold red][!ERR] Failed to load config:[/bold red] {e}")
        return 1

    # Run workflow with config
    result = run_standard_workflow(
        pop_size=config.ga.pop_size,
        generations=config.ga.ngen,
        crossover_prob=config.ga.cxpb,
        mutation_prob=config.ga.mutpb,
        validate=True,  # Enable input validation
        config=config,  # Pass config object to workflow
    )

    # Print final summary with beautiful rich formatting
    console.print()
    console.rule("[bold green]FINAL RESULTS[/bold green]", style="green")
    console.print()

    hard_viol = result["best_individual"].fitness.values[0]
    soft_pen = result["best_individual"].fitness.values[1]

    if hard_viol == 0:
        console.print(
            "[OK] [bold green]Perfect schedule found (no hard constraint violations)![/bold green]"
        )
    else:
        console.print(
            f"[!] [yellow]Hard constraint violations: {hard_viol:.0f}[/yellow]"
        )

    console.print(f"[cyan]Soft constraint penalty: {soft_pen:.2f}[/cyan]")
    console.print(f"[cyan]Schedule sessions: {len(result['decoded_schedule'])}[/cyan]")
    console.print(f"[cyan]Output location: {result['output_path']}[/cyan]")
    console.print()
    console.rule(style="green")


if __name__ == "__main__":
    main()
