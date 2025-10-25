"""
Schedule Engine Entry Point

Runs standard GA-based course scheduling workflow.
"""

from rich.console import Console
from rich.panel import Panel
from src.workflows import run_standard_workflow
from config.ga_params import (
    POP_SIZE,
    NGEN,
    CXPB,
    MUTPB,
    USE_MULTIPROCESSING,
    NUM_WORKERS,
)

console = Console()


def main():
    """
    Execute standard scheduling workflow.

    Pipeline:
        1. Load input data from data/
        2. Validate input for consistency
        3. Run NSGA-II genetic algorithm (with optional parallelization)
        4. Export best schedule to output/
        5. Generate evolution plots and reports

    Configuration is loaded from config/ga_params.py and config/constraints.py.
    Results are saved to output/evaluation_<timestamp>/.

    Parallelization:
        - If USE_MULTIPROCESSING=True, uses multiprocessing.Pool for parallel fitness evaluation
        - Provides 3-6x speedup on multi-core systems
        - Set USE_MULTIPROCESSING=False for debugging or single-threaded execution
    """
    # Pool will be created inside workflow after data is loaded
    # (needed to pass data to worker initializer)
    result = run_standard_workflow(
        pop_size=POP_SIZE,
        generations=NGEN,
        crossover_prob=CXPB,
        mutation_prob=MUTPB,
        validate=True,  # Enable input validation
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
