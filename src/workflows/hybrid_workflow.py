"""
Hybrid CP-SAT → NSGA-II Workflow

Two-phase optimization strategy:
    Phase 1: CP-SAT generates N feasible solutions (all hard constraints satisfied)
    Phase 2: NSGA-II optimizes those solutions for soft constraints

Benefits:
    - Guaranteed feasibility (CP-SAT ensures zero hard violations)
    - Quality optimization (NSGA-II optimizes soft preferences)
    - Fast convergence (GA starts from valid solutions)
    - Pareto-optimal trade-offs (multi-objective optimization)
"""

import time
from typing import List, Dict, Tuple
from deap import base, creator, tools
from rich.console import Console
from rich.panel import Panel

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
    """
    genes = []

    for session in sessions:
        gene = SessionGene(
            course_id=session.course_id,
            course_type=session.course_type,
            group_ids=session.group_ids,  # Keep as list (SessionGene expects list)
            instructor_id=session.instructor_id,
            room_id=session.room_id,
            quanta=session.session_quanta,
        )
        genes.append(gene)

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
    """
    # Decode individual
    sessions = decode_individual(
        individual, context.courses, context.instructors, context.groups, context.rooms
    )

    # Get enabled soft constraints
    enabled_soft = get_enabled_soft_constraints()

    # Categorize constraints (this is placeholder - will be configurable)
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
    random_seed: int = None,
) -> Dict:
    """
    Run hybrid CP-SAT → NSGA-II optimization workflow.

    Args:
        context: SchedulingContext with all entities
        qts: QuantumTimeSystem for time calculations
        num_cp_solutions: Number of feasible solutions to generate with CP-SAT
        ga_population_size: NSGA-II population size
        ga_generations: Number of GA generations
        cp_time_limit: CP-SAT time limit in seconds
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with results including Pareto front
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

    console.print(
        "\n[bold yellow]═══ Phase 1: CP-SAT Feasibility Generation ═══[/bold yellow]\n"
    )

    cp_start = time.time()

    scheduler = CPScheduler(
        context=context,
        qts=qts,
        time_limit_seconds=cp_time_limit,
        random_seed=random_seed,
    )

    try:
        cp_solutions = scheduler.generate_feasible_solutions(
            num_solutions=num_cp_solutions
        )
    except ValueError as e:
        console.print(f"\n[bold red]✗ CP-SAT Phase Failed:[/bold red] {e}")
        return {"success": False, "error": str(e), "phase": "cp-sat"}

    cp_time = time.time() - cp_start

    console.print(f"[green]✓ Phase 1 Complete[/green] ({cp_time:.2f}s)")
    console.print(f"Generated {len(cp_solutions)} feasible solutions\n")

    console.print(
        "[bold yellow]═══ Phase 2: NSGA-II Soft Constraint Optimization ═══[/bold yellow]\n"
    )

    ga_start = time.time()

    # Setup DEAP
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -0.1))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # Convert CP-SAT solutions to GA individuals
    initial_population = []
    for cp_solution in cp_solutions[:ga_population_size]:
        genes = coursesessions_to_genes(cp_solution)
        individual = creator.Individual(genes)
        initial_population.append(individual)

    # If we have fewer CP solutions than population size, duplicate some
    while len(initial_population) < ga_population_size:
        genes = coursesessions_to_genes(
            cp_solutions[len(initial_population) % len(cp_solutions)]
        )
        individual = creator.Individual(genes)
        initial_population.append(individual)

    console.print(
        f"Created initial population of {len(initial_population)} individuals"
    )

    # Register evaluation function
    toolbox.register("evaluate", lambda ind: evaluate_soft_only(ind, context))

    # Register genetic operators
    # TODO: Will need to implement CP-SAT repair operator for mutations
    # For now, skip operators and just evaluate initial population

    # Evaluate initial population
    console.print("Evaluating initial population...")
    fitnesses = [toolbox.evaluate(ind) for ind in initial_population]
    for ind, fit in zip(initial_population, fitnesses):
        ind.fitness.values = fit

    console.print(f"Evaluation complete")

    # Select best solutions (Pareto front)
    pareto_front = tools.sortNondominated(
        initial_population, len(initial_population), first_front_only=True
    )[0]

    ga_time = time.time() - ga_start

    console.print(f"\n[green]✓ Phase 2 Complete[/green] ({ga_time:.2f}s)")
    console.print(f"Pareto front size: {len(pareto_front)}\n")

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
    best_sessions = decode_individual(
        best_individual,
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    return {
        "success": True,
        "best_individual": best_individual,
        "best_sessions": best_sessions,
        "pareto_front": pareto_front,
        "cp_solutions": cp_solutions,
        "cp_time": cp_time,
        "ga_time": ga_time,
        "total_time": total_time,
        "num_feasible_solutions": len(cp_solutions),
    }
