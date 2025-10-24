"""
Standard Workflow Module

Orchestrates full scheduling pipeline: load → validate → schedule → export.
Extracted from main.py for better testability and reusability.
"""

import os
import random
from datetime import datetime
from typing import Dict, Optional
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.encoder.input_encoder import (
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
    link_courses_and_groups,
    link_courses_and_instructors,
)
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.decoder.individual_decoder import decode_individual
from src.core.types import SchedulingContext
from src.core.ga_scheduler import GAScheduler, GAConfig
from src.validation import validate_input
from src.workflows.reporting import generate_reports
from src.utils.logger import GALogger
from config.constraints import HARD_CONSTRAINTS_CONFIG, SOFT_CONSTRAINTS_CONFIG
from config.ga_params import REPAIR_HEURISTICS_CONFIG

console = Console()


def run_standard_workflow(
    pop_size: int,
    generations: int,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    data_dir: str = "data",
    output_dir: Optional[str] = None,
    seed: int = 69,
    validate: bool = True,
    pool=None,  # NEW: Optional multiprocessing Pool for parallel evaluation
) -> Dict:
    """
    Execute standard GA scheduling workflow.

    Pipeline:
        1. Initialize RNG and output directory
        2. Load input data from JSON files
        3. Validate input data (optional)
        4. Setup and run GA scheduler
        5. Decode best solution
        6. Generate reports and exports

    Args:
        pop_size: Population size for GA
        generations: Number of GA generations
        crossover_prob: Crossover probability (0.0-1.0)
        mutation_prob: Mutation probability (0.0-1.0)
        data_dir: Directory containing input JSON files
        output_dir: Output directory (auto-generated if None)
        seed: Random seed for reproducibility
        validate: Whether to validate input before running GA
        pool: Optional multiprocessing.Pool for parallel fitness evaluation

    Returns:
        Dict containing:
            - best_individual: Best GA solution
            - decoded_schedule: List of CourseSession objects
            - metrics: GAMetrics with evolution data
            - output_path: Path to output directory
            - qts: QuantumTimeSystem instance

    Raises:
        ValueError: If input validation fails
    """

    # ========================================
    # Step 1: Initialize
    # ========================================
    console.rule(
        "[bold cyan]SCHEDULE ENGINE - Standard Workflow[/bold cyan]", style="cyan"
    )
    console.print()

    # Set random seed
    random.seed(seed)
    console.print(f"[dim]Random seed: {seed}[/dim]")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    console.print(f"[dim]Output directory: {output_dir}[/dim]")
    console.print()

    # ========================================
    # Step 2: Load Data
    # ========================================
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading Input Data...", total=5)
        qts, context = load_input_data(data_dir)
        progress.update(task, completed=5)

    console.print(f"   [cyan]Courses:[/cyan] {len(context.courses)}")
    console.print(f"   [cyan]Groups:[/cyan] {len(context.groups)}")
    console.print(f"   [cyan]Instructors:[/cyan] {len(context.instructors)}")
    console.print(f"   [cyan]Rooms:[/cyan] {len(context.rooms)}")
    console.print(f"   [cyan]Time quanta:[/cyan] {len(context.available_quanta)}")
    console.print()

    # ========================================
    # Step 3: Validate (Optional)
    # ========================================
    if validate:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Validating Input...", total=None)
            validation_result = validate_input(context, strict=False)
            progress.update(task, completed=1)

        if not validation_result:
            raise ValueError(
                "[X] Input validation failed with ERRORS! Fix errors and try again."
            )

        console.print("   [bold green]✓[/bold green] Input validation passed\n")

    # ========================================
    # Step 4: Configure GA
    # ========================================
    ga_config = GAConfig(
        pop_size=pop_size,
        generations=generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        repair_config=REPAIR_HEURISTICS_CONFIG,
    )

    # Get enabled constraint names
    hard_names = [
        name for name, cfg in HARD_CONSTRAINTS_CONFIG.items() if cfg["enabled"]
    ]
    soft_names = [
        name for name, cfg in SOFT_CONSTRAINTS_CONFIG.items() if cfg["enabled"]
    ]

    # ========================================
    # Step 4.5: Initialize Logger
    # ========================================
    from config.ga_params import (
        USE_MULTIPROCESSING,
        NUM_WORKERS,
        POPULATION_STRATEGY,
        USE_ADAPTIVE_PROBABILITIES,
        ELITE_PRESERVATION,
        ELITE_SIZE,
    )

    logger_config = {
        "pop_size": pop_size,
        "generations": generations,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "seed": seed,
        "use_multiprocessing": USE_MULTIPROCESSING,
        "num_workers": NUM_WORKERS if NUM_WORKERS else "auto",
        "population_strategy": POPULATION_STRATEGY,
        "adaptive_operators": USE_ADAPTIVE_PROBABILITIES,
        "elite_preservation": ELITE_PRESERVATION,
        "elite_size": f"{ELITE_SIZE:.1%}",
        "num_hard_constraints": len(hard_names),
        "num_soft_constraints": len(soft_names),
        "repair_enabled": REPAIR_HEURISTICS_CONFIG.get("enabled", False),
        "repair_max_iterations": REPAIR_HEURISTICS_CONFIG.get("max_iterations", 0),
        "repair_after_mutation": REPAIR_HEURISTICS_CONFIG.get(
            "apply_after_mutation", False
        ),
        "repair_after_crossover": REPAIR_HEURISTICS_CONFIG.get(
            "apply_after_crossover", False
        ),
        "repair_memetic_mode": REPAIR_HEURISTICS_CONFIG.get("memetic_mode", False),
        "repair_memetic_iterations": REPAIR_HEURISTICS_CONFIG.get(
            "memetic_iterations", 0
        ),
        "num_courses": len(context.courses),
        "num_groups": len(context.groups),
        "num_instructors": len(context.instructors),
        "num_rooms": len(context.rooms),
        "num_quanta": len(context.available_quanta),
    }

    logger = GALogger(output_dir, logger_config)
    console.print(f"   [dim]Logger:[/dim] [cyan]{logger.get_log_path()}[/cyan]")

    console.print("[bold]Genetic Algorithm Configuration:[/bold]")
    console.print(
        f"   Population: [cyan]{ga_config.pop_size}[/cyan] | Generations: [cyan]{ga_config.generations}[/cyan]"
    )
    console.print(
        f"   Crossover: [cyan]{ga_config.crossover_prob:.1%}[/cyan] | Mutation: [cyan]{ga_config.mutation_prob:.1%}[/cyan]"
    )
    console.print(
        f"   Constraints: [yellow]{len(hard_names)} hard[/yellow], [blue]{len(soft_names)} soft[/blue]"
    )

    # Display repair configuration status
    if REPAIR_HEURISTICS_CONFIG.get("enabled", False):
        repair_modes = []
        if REPAIR_HEURISTICS_CONFIG.get("apply_after_mutation", False):
            repair_modes.append("mutation")
        if REPAIR_HEURISTICS_CONFIG.get("apply_after_crossover", False):
            repair_modes.append("crossover")
        if REPAIR_HEURISTICS_CONFIG.get("memetic_mode", False):
            repair_modes.append(
                f"memetic({REPAIR_HEURISTICS_CONFIG.get('elite_percentage', 0.2):.0%} elite)"
            )

        modes_str = ", ".join(repair_modes) if repair_modes else "none"
        console.print(
            f"   Repair Heuristics: [green]✓ enabled[/green] (after {modes_str}, max {REPAIR_HEURISTICS_CONFIG.get('max_iterations', 3)} iter)"
        )
    else:
        console.print(f"   Repair Heuristics: [dim]✗ disabled[/dim]")

    console.print()

    # ========================================
    # Step 5: Run GA
    # ========================================
    console.print("[bold green]Running Genetic Algorithm...[/bold green]\n")

    logger.start_run()  # Mark start time

    scheduler = GAScheduler(
        ga_config, context, hard_names, soft_names, pool=pool, logger=logger
    )
    scheduler.setup_toolbox()
    scheduler.initialize_population()
    scheduler.evolve()

    # ========================================
    # Step 6: Decode Best Solution
    # ========================================
    console.print()
    console.print("[bold]Processing Results...[/bold]")

    best_individual = scheduler.get_best_solution()
    decoded_schedule = decode_individual(
        best_individual,
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    console.print(
        f"   Hard Violations: [yellow]{best_individual.fitness.values[0]:.0f}[/yellow]"
    )
    console.print(
        f"   Soft Penalty: [blue]{best_individual.fitness.values[1]:.2f}[/blue]"
    )
    console.print(f"   Schedule sessions: [cyan]{len(decoded_schedule)}[/cyan]")
    console.print()

    # Finalize logger
    logger.end_run(
        best_hard=best_individual.fitness.values[0],
        best_soft=best_individual.fitness.values[1],
        final_schedule_sessions=len(decoded_schedule),
    )
    console.print(f"   [dim]Log saved:[/dim] [cyan]{logger.get_log_path()}[/cyan]")
    console.print()

    # ========================================
    # Step 7: Generate Reports
    # ========================================
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("📊 Generating Reports...", total=None)
        generate_reports(
            decoded_schedule=decoded_schedule,
            metrics=scheduler.metrics,
            population=scheduler.population,
            qts=qts,
            output_dir=output_dir,
            course_map=context.courses,
        )
        progress.update(task, completed=1)

    # ========================================
    # Done!
    # ========================================
    console.print()
    console.rule("[bold green]WORKFLOW COMPLETE[/bold green]", style="green")
    console.print()
    console.print(f"[bold]Results saved to:[/bold] [cyan]{output_dir}[/cyan]")
    console.print(f"   • [dim]schedule.json[/dim]: Schedule data")
    console.print(f"   • [dim]schedule.pdf[/dim]: Visual calendar")
    console.print(f"   • [dim]plots/[/dim]: Evolution charts")
    console.print()
    console.rule(style="green")
    console.print()

    return {
        "best_individual": best_individual,
        "decoded_schedule": decoded_schedule,
        "metrics": scheduler.metrics,
        "output_path": output_dir,
        "qts": qts,
    }


def load_input_data(data_dir: str) -> tuple[QuantumTimeSystem, SchedulingContext]:
    """
    Load and link all input entities.

    Only includes courses that are enrolled by at least one group,
    filtering out the rest of the university course database.

    Args:
        data_dir: Directory containing input JSON files

    Returns:
        Tuple of (QuantumTimeSystem, SchedulingContext)
    """
    # Initialize time system
    qts = QuantumTimeSystem()

    # Step 1: Load groups first to know which course codes are enrolled
    groups = load_groups(os.path.join(data_dir, "Groups.json"), qts)

    # Step 2: Collect all enrolled course codes from groups
    enrolled_course_codes = set()
    for group in groups.values():
        enrolled_course_codes.update(group.enrolled_courses)

    print(
        f"[INFO] Found {len(enrolled_course_codes)} unique course codes enrolled by groups"
    )

    # Step 3: Load ALL courses from database
    all_courses = load_courses(os.path.join(data_dir, "Course.json"))

    # Step 4: Filter to only keep courses whose course_code is enrolled
    # Note: Dict keyed by (course_code, course_type) tuples
    # A single course_code may have both theory and practical versions
    courses = {}
    for course_key, course in all_courses.items():
        # course_key is (course_code, course_type)
        course_code = course_key[0]
        if course_code in enrolled_course_codes:
            courses[course_key] = course

    excluded_count = len(all_courses) - len(courses)
    print(
        f"[INFO] Filtered {len(courses)} course objects from {len(all_courses)} total in database"
    )
    print(f"[INFO] ({excluded_count} courses excluded - not enrolled by any group)")

    # Step 5: Load other entities
    instructors = load_instructors(os.path.join(data_dir, "Instructors.json"), qts)
    rooms = load_rooms(os.path.join(data_dir, "Rooms.json"), qts)

    # Step 6: Link relationships (only for enrolled courses)
    link_courses_and_groups(courses, groups)
    link_courses_and_instructors(courses, instructors)

    # Create context with filtered courses
    context = SchedulingContext(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        available_quanta=qts.get_all_operating_quanta(),
    )

    return qts, context
