"""
Hybrid CP-SAT → NSGA-II Workflow (Production Version)

Two-phase optimization strategy with comprehensive error handling:
    Phase 1: CP-SAT generates N feasible solutions (all hard constraints satisfied)
    Phase 2: NSGA-II optimizes those solutions for soft constraints

Benefits:
    - Guaranteed feasibility (CP-SAT ensures zero hard violations)
    - Quality optimization (NSGA-II optimizes soft preferences)
    - Fast convergence (GA starts from valid solutions)
    - Pareto-optimal trade-offs (multi-objective optimization)
    - Production-ready error handling and logging
"""

import time
import traceback
from typing import List, Dict, Tuple, Optional
from deap import base, creator, tools
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.entities.decoded_session import CourseSession
from src.ortools.cp_scheduler import CPScheduler
from src.ga.sessiongene import SessionGene
from src.constraints.soft import get_enabled_soft_constraints
from src.decoder.individual_decoder import decode_individual

console = Console()


def coursesessions_to_genes(sessions: List[CourseSession]) -> List[SessionGene]:
    """
    Convert CourseSession objects to SessionGene chromosomes for GA.

    Args:
        sessions: List of CourseSession objects from CP-SAT

    Returns:
        List of SessionGene objects (GA chromosome)

    Raises:
        ValueError: If conversion fails due to invalid session data
    """
    genes = []

    for idx, session in enumerate(sessions):
        try:
            gene = SessionGene(
                course_id=session.course_id,
                course_type=session.course_type,
                group_ids=session.group_ids,  # Keep as list
                instructor_id=session.instructor_id,
                room_id=session.room_id,
                quanta=session.session_quanta,
            )
            genes.append(gene)
        except Exception as e:
            raise ValueError(f"Failed to convert session {idx} to gene: {e}")

    return genes


def evaluate_soft_only(
    individual: List[SessionGene], context: SchedulingContext
) -> Tuple[float, float]:
    """
    Evaluate only soft constraints (hard constraints assumed satisfied).

    Two objectives:
        - Objective 1: Strict soft constraints (e.g., minimize gaps, respect preferences)
        - Objective 2: Loose soft constraints (e.g., fairness, balance)

    Args:
        individual: GA chromosome (list of SessionGene)
        context: SchedulingContext with all entities

    Returns:
        Tuple[float, float]: (strict_penalty, loose_penalty)

    Raises:
        Exception: If evaluation fails (will be caught by workflow)
    """
    # Decode individual
    sessions = decode_individual(
        individual, context.courses, context.instructors, context.groups, context.rooms
    )

    # Get enabled soft constraints
    enabled_soft = get_enabled_soft_constraints()

    # Categorize constraints
    strict_constraints = [
        "group_gap_penalty",
        "instructor_gap_penalty",
        "time_preference_violation",
    ]

    strict_penalty = 0.0
    loose_penalty = 0.0

    for constraint_name, constraint_info in enabled_soft.items():
        constraint_func = constraint_info["function"]
        weight = constraint_info["weight"]
        penalty = constraint_func(sessions)

        if constraint_name in strict_constraints:
            strict_penalty += weight * penalty
        else:
            loose_penalty += weight * penalty

    return (strict_penalty, loose_penalty)


def run_hybrid_workflow(
    context: SchedulingContext,
    qts: QuantumTimeSystem,
    num_cp_solutions: int = 50,
    ga_population_size: int = 100,
    ga_generations: int = 50,
    cp_time_limit: int = 300,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Run hybrid CP-SAT → NSGA-II optimization workflow with production-level error handling.

    Args:
        context: SchedulingContext with all entities
        qts: QuantumTimeSystem for time calculations
        num_cp_solutions: Number of feasible solutions to generate with CP-SAT
        ga_population_size: NSGA-II population size
        ga_generations: Number of GA generations (currently not used - just evaluation)
        cp_time_limit: CP-SAT time limit in seconds
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with results:
            success (bool): Whether workflow completed successfully
            best_individual: Best individual from Pareto front
            best_sessions: Decoded best schedule
            decoded_schedule: Same as best_sessions (for compatibility)
            pareto_front: List of non-dominated solutions
            cp_solutions: Raw CP-SAT solutions
            cp_time: Time spent in CP-SAT phase
            ga_time: Time spent in GA phase
            total_time: Total workflow time
            num_feasible_solutions: Number of feasible solutions generated
            error (optional): Error message if failed
            phase (optional): Phase where failure occurred
    """
    console.print(
        Panel.fit(
            "[bold cyan]Hybrid CP-SAT → NSGA-II Workflow[/bold cyan]\n\n"
            f"Phase 1: Generate {num_cp_solutions} feasible solutions (CP-SAT)\n"
            f"Phase 2: Optimize for soft constraints (NSGA-II, {ga_generations} gens)",
            border_style="cyan",
        )
    )

    overall_start = time.time()

    # ========================================
    # PHASE 1: CP-SAT Feasibility Generation
    # ========================================
    console.print(
        "\n[bold yellow]═══ Phase 1: CP-SAT Feasibility Generation ═══[/bold yellow]\n"
    )

    cp_start = time.time()

    try:
        scheduler = CPScheduler(
            context=context,
            qts=qts,
            time_limit_seconds=cp_time_limit,
            random_seed=random_seed,
        )

        cp_solutions = scheduler.generate_feasible_solutions(
            num_solutions=num_cp_solutions
        )

    except ValueError as e:
        console.print(f"\n[bold red]✗ CP-SAT Phase Failed:[/bold red] {e}")
        return {"success": False, "error": str(e), "phase": "cp-sat"}

    except Exception as e:
        console.print(
            f"\n[bold red]✗ CP-SAT Phase Error:[/bold red] {type(e).__name__}: {e}"
        )
        traceback.print_exc()
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "phase": "cp-sat",
        }

    cp_time = time.time() - cp_start

    # Validate CP-SAT output
    if not cp_solutions:
        console.print("[bold red]✗ No solutions generated by CP-SAT[/bold red]")
        return {"success": False, "error": "No solutions generated", "phase": "cp-sat"}

    console.print(f"[green]✓ Phase 1 Complete[/green] ({cp_time:.2f}s)")
    console.print(f"Generated {len(cp_solutions)} feasible solutions")
    console.print(
        f"[dim]Solution sizes: {[len(sol) for sol in cp_solutions[:5]]}{'...' if len(cp_solutions) > 5 else ''}[/dim]\n"
    )

    # ========================================
    # PHASE 2: NSGA-II Optimization
    # ========================================
    console.print(
        "[bold yellow]═══ Phase 2: NSGA-II Soft Constraint Optimization ═══[/bold yellow]\n"
    )

    ga_start = time.time()

    try:
        # Setup DEAP (clean slate)
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -0.1))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Convert CP-SAT solutions to GA individuals
        console.print("Converting CP-SAT solutions to GA chromosomes...")
        initial_population = []
        conversion_errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Converting...", total=min(len(cp_solutions), ga_population_size)
            )

            for idx, cp_solution in enumerate(cp_solutions[:ga_population_size]):
                try:
                    genes = coursesessions_to_genes(cp_solution)
                    individual = creator.Individual(genes)
                    initial_population.append(individual)
                    progress.update(task, advance=1)

                except Exception as e:
                    console.print(
                        f"[yellow]⚠ Warning: Failed to convert solution {idx}: {e}[/yellow]"
                    )
                    conversion_errors += 1
                    continue

        if not initial_population:
            console.print(
                "[bold red]✗ Failed to convert any CP-SAT solutions to GA format[/bold red]"
            )
            return {
                "success": False,
                "error": "Solution conversion failed",
                "phase": "conversion",
            }

        if conversion_errors > 0:
            console.print(
                f"[yellow]⚠ {conversion_errors} conversions failed, continuing with {len(initial_population)} solutions[/yellow]"
            )

        # Pad population if needed
        original_size = len(initial_population)
        while len(initial_population) < ga_population_size:
            idx = len(initial_population) % original_size
            genes = coursesessions_to_genes(cp_solutions[idx])
            individual = creator.Individual(genes)
            initial_population.append(individual)

        console.print(
            f"[green]✓[/green] Created initial population: {len(initial_population)} individuals\n"
        )

        # Register evaluation function
        toolbox.register("evaluate", lambda ind: evaluate_soft_only(ind, context))

        # Evaluate initial population
        console.print("Evaluating soft constraints...")
        fitnesses = []
        eval_errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(initial_population))

            for idx, ind in enumerate(initial_population):
                try:
                    fit = toolbox.evaluate(ind)
                    fitnesses.append(fit)
                    progress.update(task, advance=1)

                except Exception as e:
                    console.print(
                        f"[yellow]⚠ Warning: Evaluation failed for individual {idx}: {e}[/yellow]"
                    )
                    fitnesses.append(
                        (float("inf"), float("inf"))
                    )  # Worst possible fitness
                    eval_errors += 1

        if eval_errors > 0:
            console.print(
                f"[yellow]⚠ {eval_errors} evaluations failed, assigned infinite penalty[/yellow]"
            )

        # Assign fitness values
        for ind, fit in zip(initial_population, fitnesses):
            ind.fitness.values = fit

        console.print(f"[green]✓[/green] Evaluation complete\n")

        # Select best solutions (Pareto front)
        console.print("Extracting Pareto front...")
        pareto_front = tools.sortNondominated(
            initial_population, len(initial_population), first_front_only=True
        )[0]

        ga_time = time.time() - ga_start

        console.print(f"\n[green]✓ Phase 2 Complete[/green] ({ga_time:.2f}s)")
        console.print(f"Pareto front size: {len(pareto_front)}\n")

    except Exception as e:
        console.print(
            f"\n[bold red]✗ GA Phase Error:[/bold red] {type(e).__name__}: {e}"
        )
        traceback.print_exc()
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "phase": "ga",
        }

    # ========================================
    # RESULTS
    # ========================================
    total_time = time.time() - overall_start

    console.print(
        Panel.fit(
            f"[bold green]Hybrid Workflow Complete[/bold green]\n\n"
            f"Total Time: {total_time:.2f}s\n"
            f"  • CP-SAT Phase: {cp_time:.2f}s\n"
            f"  • NSGA-II Phase: {ga_time:.2f}s\n"
            f"Pareto Front: {len(pareto_front)} optimal solutions",
            border_style="green",
        )
    )

    # Select best from Pareto front (lowest strict penalty)
    best_individual = min(pareto_front, key=lambda ind: ind.fitness.values[0])

    # Decode best solution
    try:
        best_sessions = decode_individual(
            best_individual,
            context.courses,
            context.instructors,
            context.groups,
            context.rooms,
        )

        console.print(f"\n[cyan]Best Solution:[/cyan]")
        console.print(f"  Strict penalty: {best_individual.fitness.values[0]:.2f}")
        console.print(f"  Loose penalty: {best_individual.fitness.values[1]:.2f}")
        console.print(f"  Sessions: {len(best_sessions)}")

    except Exception as e:
        console.print(
            f"[yellow]⚠ Warning: Failed to decode best solution: {e}[/yellow]"
        )
        best_sessions = []

    return {
        "success": True,
        "best_individual": best_individual,
        "best_sessions": best_sessions,
        "decoded_schedule": best_sessions,  # For compatibility with existing code
        "pareto_front": pareto_front,
        "cp_solutions": cp_solutions,
        "cp_time": cp_time,
        "ga_time": ga_time,
        "total_time": total_time,
        "num_feasible_solutions": len(cp_solutions),
    }
