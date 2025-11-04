"""
Schedule Engine Entry Point

Supports multiple solving modes:
    - hybrid: CP-SAT feasibility + NSGA-II optimization (recommended)
    - cpsat: Pure CP-SAT constraint programming
    - ga: Legacy NSGA-II genetic algorithm (deprecated)
"""

import argparse
import time
from typing import Dict, Optional
from rich.console import Console
from rich.panel import Panel
from src.workflows import run_standard_workflow
from src.config import init_config

console = Console()


def main():
    """
    Execute scheduling workflow.

    Modes:
        --mode hybrid: CP-SAT + NSGA-II (default, recommended)
        --mode cpsat: Pure CP-SAT for fast feasible solution
        --mode ga: Legacy GA-only (deprecated)

    Pipeline:
        1. Load configuration from YAML
        2. Load input data from data/
        3. Validate input for consistency
        4. Run solver based on mode
        5. Export best schedule to output/

    Configuration:
        Use --env {test|dev|prod} to select configuration.

    Results:
        Saved to output/evaluation_<timestamp>/
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "cpsat", "ga"],
        default="hybrid",
        help="Solver mode: hybrid (CP-SAT+NSGA-II), cpsat (pure CP-SAT), ga (legacy)",
    )
    args = parser.parse_args()

    # Determine config strategy
    config_path = args.config
    if args.env:
        # Set environment variable for common.yaml + env.yaml merge
        import os

        os.environ["ENVIRONMENT"] = args.env
        config_path = None  # Trigger merge strategy in loader

    # Load configuration
    try:
        config = init_config(config_path)
        console.print()
        console.print(config.summary())
        console.print()
    except Exception as e:
        console.print(f"[bold red][!err] failed to load config:[/bold red] {e}")
        return 1

    # Route to appropriate workflow based on mode
    if args.mode == "hybrid":
        result = run_hybrid_mode(config)
    elif args.mode == "cpsat":
        result = run_cpsat_mode(config)
    else:  # ga mode
        result = run_ga_mode(config)

    # Print final summary
    if result:
        print_results(result, args.mode)

    return 0


def run_hybrid_mode(config) -> Dict:
    """Run hybrid CP-SAT + NSGA-II workflow."""
    from src.workflows.hybrid_workflow_v2 import run_hybrid_workflow
    from src.workflows.standard_run import load_input_data
    from src.validation.input_validator import validate_input

    console.print("\n[bold cyan]═══ Hybrid Mode: CP-SAT + NSGA-II ═══[/bold cyan]\n")

    # Load data
    qts, context = load_input_data(config.io.data_dir)

    # Validate
    if not validate_input(context):
        console.print("[bold red]✗ Input validation failed[/bold red]")
        return None

    # Run hybrid workflow
    result = run_hybrid_workflow(
        context=context,
        qts=qts,
        num_cp_solutions=config.ortools.num_solutions,
        ga_population_size=config.ga.pop_size,
        ga_generations=config.ga.ngen,
        cp_time_limit=config.ortools.time_limit,
    )

    return result


def run_cpsat_mode(config) -> Dict:
    """Run pure CP-SAT workflow for single feasible solution."""
    from src.ortools.cp_scheduler import CPScheduler, setup_cp_logger
    from src.workflows.standard_run import load_input_data
    from src.validation.input_validator import validate_input

    console.print(
        "\n[bold cyan]═══ CP-SAT Mode: Pure Constraint Programming ═══[/bold cyan]\n"
    )

    # Setup logger
    logger = setup_cp_logger()
    logger.info("CP-SAT MODE - Started")
    logger.info(f"Configuration: {config.ortools.time_limit}s time limit")
    logger.info("-" * 80)

    # Load data
    logger.info("DATA LOADING - Started")
    data_start = time.time()
    qts, context = load_input_data(config.io.data_dir)
    data_time = time.time() - data_start
    logger.info(f"DATA LOADING - Completed in {data_time:.2f}s")
    logger.info(f"  Courses: {len(context.courses)}")
    logger.info(f"  Groups: {len(context.groups)}")
    logger.info(f"  Instructors: {len(context.instructors)}")
    logger.info(f"  Rooms: {len(context.rooms)}")
    logger.info("-" * 80)

    # Validate
    logger.info("INPUT VALIDATION - Started")
    valid_start = time.time()
    validation_result = validate_input(context)
    valid_time = time.time() - valid_start
    logger.info(f"INPUT VALIDATION - Completed in {valid_time:.2f}s")

    if not validation_result:
        console.print("[bold red]✗ Input validation failed[/bold red]")
        logger.error("Input validation failed")
        return None

    logger.info("  ✓ Input validation passed")
    logger.info("-" * 80)

    # Run CP-SAT
    scheduler = CPScheduler(
        context=context,
        qts=qts,
        time_limit_seconds=config.ortools.time_limit,
    )

    try:
        best_sessions = scheduler.generate_single_solution(logger=logger)
        logger.info("=" * 80)
        logger.info("CP-SAT MODE - COMPLETED SUCCESSFULLY")
        return {"success": True, "best_sessions": best_sessions, "mode": "cpsat"}
    except ValueError as e:
        console.print(f"[bold red]✗ CP-SAT failed:[/bold red] {e}")
        logger.error(f"CP-SAT failed: {e}")
        logger.info("=" * 80)
        return None


def run_ga_mode(config) -> Dict:
    """Run legacy GA-only workflow (deprecated)."""
    console.print("\n[bold yellow]⚠ Warning: GA-only mode is deprecated[/bold yellow]")
    console.print("[yellow]Consider using --mode hybrid for better results[/yellow]\n")

    result = run_standard_workflow(
        pop_size=config.ga.pop_size,
        generations=config.ga.ngen,
        crossover_prob=config.ga.cxpb,
        mutation_prob=config.ga.mutpb,
        validate=True,
        config=config,
    )

    return result


def print_results(result: Optional[Dict], mode: str):
    """Print final results summary."""
    if not result or not result.get("success", True):
        console.print("\n[bold red]✗ Workflow failed[/bold red]")
        return

    console.print()
    console.print("[bold cyan]═══ Results ═══[/bold cyan]")
    console.print()

    if mode == "hybrid":
        console.print(f"[dim]Mode:[/dim] Hybrid (CP-SAT + NSGA-II)")
        console.print(
            f"[dim]Feasible solutions generated:[/dim] {result.get('num_feasible_solutions', 'N/A')}"
        )
        console.print(
            f"[dim]Pareto front size:[/dim] {len(result.get('pareto_front', []))}"
        )
        console.print(f"[dim]CP-SAT time:[/dim] {result.get('cp_time', 0):.2f}s")
        console.print(f"[dim]GA time:[/dim] {result.get('ga_time', 0):.2f}s")

        best_ind = result.get("best_individual")
        if best_ind:
            strict, loose = best_ind.fitness.values
            console.print(
                f"[green]✓ Best solution:[/green] Strict={strict:.2f}, Loose={loose:.2f}"
            )

    elif mode == "cpsat":
        console.print(f"[dim]Mode:[/dim] Pure CP-SAT")
        sessions = result.get("best_sessions", [])
        console.print(
            f"[green]✓ Feasible solution with {len(sessions)} sessions[/green]"
        )

    else:  # ga mode
        hard_viol = result["best_individual"].fitness.values[0]
        soft_pen = result["best_individual"].fitness.values[1]

        if hard_viol == 0:
            console.print("[green]✓ Perfect schedule (no hard violations)[/green]")
        else:
            console.print(f"[yellow]⚠ Hard violations:[/yellow] {hard_viol:.0f}")

        console.print(f"  [dim]Soft penalty:[/dim] {soft_pen:.2f}")
        console.print(f"  [dim]Sessions:[/dim] {len(result['decoded_schedule'])}")

    if "output_path" in result:
        console.print(f"  [dim]Output:[/dim] {result['output_path']}")

    console.print()


if __name__ == "__main__":
    main()
