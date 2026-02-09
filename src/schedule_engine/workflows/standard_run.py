"""
Standard Workflow Module

Orchestrates full scheduling pipeline: load → validate → schedule → export.
Extracted from main.py for better testability and reusability.
"""

import os
import random
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from schedule_engine.config.models import Config
from schedule_engine.domain.types import SchedulingContext
from schedule_engine.ga.scheduler import GAConfig, GAScheduler
from schedule_engine.io import validate_input
from schedule_engine.io.data_loader import (
    derive_cohort_pairs_from_groups,
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from schedule_engine.io.decoder import decode_individual
from schedule_engine.io.feasibility import (
    check_feasibility,
    generate_feasibility_report_file,
)
from schedule_engine.io.time_system import QuantumTimeSystem
from schedule_engine.utils.console_service import get_console
from schedule_engine.utils.constraint_logger import ConstraintLogger
from schedule_engine.utils.logger import GALogger
from schedule_engine.utils.performance_profiler import cleanup_profiler, init_profiler
from schedule_engine.utils.system_info import get_cpu_count
from schedule_engine.workflows.reporting import generate_reports

console = get_console()


def run_standard_workflow(
    pop_size: int,
    generations: int,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    data_dir: str = "data",
    output_dir: str | None = None,
    seed: int = 69,
    validate: bool = True,
    config: Config | None = None,
) -> dict:
    """
    Execute standard GA scheduling workflow.

    Pipeline:
        1. Initialize RNG and output directory
        2. Load input data from JSON files
        3. Validate input data (optional)
        4. Check feasibility (optional)
        5. Setup and run GA scheduler
        6. Decode best solution
        7. Generate reports and exports

    Args:
        pop_size: Population size for GA
        generations: Number of GA generations
        crossover_prob: Crossover probability (0.0-1.0)
        mutation_prob: Mutation probability (0.0-1.0)
        data_dir: Directory containing input JSON files
        output_dir: Output directory (auto-generated if None)
        seed: Random seed for reproducibility
        validate: Whether to validate input before running GA
        config: Config object (from config.models.Config) - if None, loads from config module

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
    # Load config if not provided
    if config is None:
        from schedule_engine.config import get_config

        config = get_config()
        if config is None:
            raise ValueError(
                "Config not initialized. Call init_config() first or pass config parameter."
            )

    # ═══════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════
    console.print()
    console.print("[bold cyan]schedule engine[/bold cyan]")
    console.print()

    # Set random seed
    random.seed(seed)
    console.print(f"[dim]seed:[/dim] {seed}")

    # Create output directory
    if output_dir is None:
        # Fallback: create simple timestamped directory
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_path = Path("output") / f"evaluation_{timestamp}_auto"
        output_dir = str(output_dir_path)
    else:
        # Ensure directory exists and make sure it's normalized
        output_dir = os.path.normpath(output_dir)
    # os.makedirs already creates all parent directories by default
    os.makedirs(output_dir, exist_ok=True)
    console.print(f"[dim]output:[/dim] {output_dir}")
    console.print()

    # Try to set runtime output dir into config (best-effort), so exporters and
    # other subsystems that read config can also pick it up if they depend on
    # config.output.output_dir field.
    try:
        if hasattr(config, "output"):
            # Mutate config at runtime - many modules only read config at load time
            # but this makes the output directory available for reporters and
            # exporters that check config.output.output_dir
            if hasattr(config.output, "output_dir"):
                config.output.output_dir = output_dir
            else:
                # Some configs only have base_dir - create a new attribute for runtime
                config.output.output_dir = output_dir

        if hasattr(config, "io"):
            # GA scheduler and other systems read io.output_dir; ensure it points
            # to this specific run instead of the global default "output".
            if hasattr(config.io, "output_dir"):
                config.io.output_dir = output_dir
            else:
                config.io.output_dir = output_dir
    except Exception:
        # Don't fail the run if mutation fails; fall back to passing output_dir
        console.print(
            "[dim]warning:[/] Unable to set runtime output_dir in config object"
        )

    # ═══════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════
    console.print("[bold cyan]loading data[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}"),
        BarColumn(),
        TextColumn("[dim]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("reading json files...", total=5)
        qts, context = load_input_data(data_dir, config=config)
        progress.update(task, completed=5)

    # Right-align counts for consistent formatting
    max_count = max(
        len(context.courses),
        len(context.groups),
        len(context.instructors),
        len(context.rooms),
        len(context.available_quanta),
    )
    count_width = len(str(max_count))
    console.print(f"  [dim]courses:[/dim] {len(context.courses):>{count_width}}")
    console.print(f"  [dim]groups:[/dim] {len(context.groups):>{count_width}}")
    console.print(
        f"  [dim]instructors:[/dim] {len(context.instructors):>{count_width}}"
    )
    console.print(f"  [dim]rooms:[/dim] {len(context.rooms):>{count_width}}")
    console.print(
        f"  [dim]quanta:[/dim] {len(context.available_quanta):>{count_width}}"
    )
    console.print()

    # ═══════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════
    if validate:
        console.print("[bold cyan]validating input[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("checking data consistency...", total=None)
            validation_result = validate_input(context, strict=False)
            progress.update(task, completed=1)

        if not validation_result:
            raise ValueError(
                "[!err] input validation failed! fix errors and try again."
            )

        console.print("  [green][!ok] validation passed[/green]")
        console.print()

    # ═══════════════════════════════════════════════════════════════
    # FEASIBILITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    is_feasible, feasibility_report = check_feasibility(
        context.courses, context.instructors, context.rooms, context.groups, qts
    )

    # Generate feasibility report file (always enabled)
    feasibility_report_path = os.path.join(output_dir, "log_feasibility.log")
    generate_feasibility_report_file(feasibility_report, feasibility_report_path)
    console.print(f"  [dim]saved:[/dim] {feasibility_report_path}")
    console.print()

    # If not feasible and FAIL_ON_INFEASIBILITY=True, check_feasibility already exited
    # If we're here, either it's feasible or FAIL_ON_INFEASIBILITY=False
    if not is_feasible:
        console.print("[yellow][!warn] proceeding despite infeasibility[/yellow]")
        console.print()

    # ═══════════════════════════════════════════════════════════════
    # PARALLELIZATION SETUP
    # ═══════════════════════════════════════════════════════════════
    pool = None

    if config.parallel.use_multiprocessing:
        import multiprocessing

        from schedule_engine.utils.parallel_worker import init_worker

        # Determine worker count: None = CPU count (Windows handle limit safe)
        if config.parallel.num_workers is None:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = config.parallel.num_workers

        # Create pool with worker initialization
        # Workers load data from JSON files (no pickling of complex objects!)
        # Also pass serialized config for constraint evaluation
        import dataclasses

        config_dict = dataclasses.asdict(config)
        pool = multiprocessing.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(data_dir, seed, config_dict),
        )
        console.print(f"[cyan][!info] parallel mode:[/cyan] {pool._processes} workers")  # type: ignore[attr-defined]
        console.print()
    else:
        console.print("[yellow][!info] single-threaded mode[/yellow]")
        console.print()

    # ═══════════════════════════════════════════════════════════════
    # GA CONFIGURATION
    # ═══════════════════════════════════════════════════════════════

    # Convert config to dict format for GA scheduler
    repair_config = {
        "enabled": config.repair.enabled,
        "max_iterations": config.repair.max_iterations,
        "apply_after_mutation": config.repair.apply_after_mutation,
        "apply_after_crossover": config.repair.apply_after_crossover,
        "memetic_mode": config.repair.memetic_mode,
        "elite_percentage": config.repair.elite_percentage,
        "memetic_iterations": config.repair.memetic_iterations,
        "violation_threshold": config.repair.violation_threshold,
        "selective_mode": config.repair.selective_mode,
        "detection_strategy": config.repair.detection_strategy,
        "recheck_after_repair": config.repair.recheck_after_repair,
        "adaptive_repair": config.repair.adaptive_repair,
        "heuristics": config.repair.heuristics,
    }

    ga_config = GAConfig(
        pop_size=pop_size,
        generations=generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        repair_config=repair_config,
    )

    # ========================================
    # Get all constraint names (all constraints always enabled)
    # ========================================
    from schedule_engine.constraints.all_constraints import (
        get_all_hard_constraints,
        get_all_soft_constraints,
    )

    # Build constraint lists - all constraints are always enabled now
    all_hard_constraints = get_all_hard_constraints()
    all_soft_constraints = get_all_soft_constraints()

    # All constraints are enabled
    hard_names = list(all_hard_constraints.keys())
    soft_names = list(all_soft_constraints.keys())

    # ========================================
    # Step 4.5: Initialize Logger
    # ========================================

    logger_config = {
        "pop_size": pop_size,
        "generations": generations,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "seed": seed,
        "use_multiprocessing": config.parallel.use_multiprocessing,
        "num_workers": (
            f"{config.parallel.num_workers if config.parallel.num_workers else 'auto (CPU)'}"
        ),
        "population_strategy": config.ga.population_strategy,
        "adaptive_operators": config.ga.use_adaptive_probabilities,
        "elite_preservation": config.ga.elite_preservation,
        "elite_size": f"{config.ga.elite_size:.1%}",
        "num_hard_constraints": len(hard_names),
        "num_soft_constraints": len(soft_names),
        "repair_enabled": config.repair.enabled,
        "repair_max_iterations": config.repair.max_iterations,
        "repair_after_mutation": config.repair.apply_after_mutation,
        "repair_after_crossover": config.repair.apply_after_crossover,
        "repair_memetic_mode": config.repair.memetic_mode,
        "repair_memetic_iterations": config.repair.memetic_iterations,
        "num_courses": len(context.courses),
        "num_groups": len(context.groups),
        "num_instructors": len(context.instructors),
        "num_rooms": len(context.rooms),
        "num_quanta": len(context.available_quanta),
    }

    logger = GALogger(output_dir, logger_config)
    console.print(f"  [dim]run log:[/dim] {logger.get_log_path()}")

    # Initialize ConstraintLogger for detailed per-generation constraint breakdown
    constraint_logger = ConstraintLogger(output_dir, hard_names, soft_names)
    console.print(f"  [dim]metrics csv:[/dim] {constraint_logger.get_log_path()}")
    console.print()

    # ========================================
    # Step 5: Run GA
    # ========================================
    console.print("[bold cyan]genetic algorithm[/bold cyan]")
    console.print(
        f"  [dim]population:[/dim] {ga_config.pop_size} | [dim]generations:[/dim] {ga_config.generations}"
    )
    console.print(
        f"  [dim]crossover:[/dim] {ga_config.crossover_prob:.1%} | [dim]mutation:[/dim] {ga_config.mutation_prob:.1%}"
    )
    console.print(
        f"  [dim]constraints:[/dim] {len(hard_names)} hard, {len(soft_names)} soft"
    )

    # Display repair configuration status
    if config.repair.enabled:
        repair_modes = []
        if config.repair.apply_after_mutation:
            repair_modes.append("mutation")
        if config.repair.apply_after_crossover:
            repair_modes.append("crossover")
        if config.repair.memetic_mode:
            repair_modes.append(f"memetic({config.repair.elite_percentage:.0%})")
        modes_str = ", ".join(repair_modes) if repair_modes else "none"
        console.print(
            f"  [dim]repair:[/dim] [green]enabled[/green] (after {modes_str}, max {config.repair.max_iterations} iter)"
        )
    else:
        console.print("  [dim]repair:[/dim] [dim]disabled[/dim]")

    console.print()

    logger.start_run()  # Mark start time

    # Initialize performance profiler
    profiling_enabled = True
    init_profiler(enabled=True, console=console)
    console.print(
        "[dim]  performance profiling: [green]enabled[/green] (micro-breakdown per generation)[/dim]"
    )

    scheduler = GAScheduler(
        ga_config,
        context,
        hard_names,
        soft_names,
        pool=pool,
        logger=logger,
        constraint_logger=constraint_logger,  # NEW: Pass constraint logger
        seed=seed,
    )
    scheduler.setup_toolbox()
    scheduler.initialize_population()
    scheduler.evolve()

    # Cleanup profiler and show summary
    if profiling_enabled:
        cleanup_profiler()

    # ═══════════════════════════════════════════════════════════════
    # SOLUTION DECODING
    # ═══════════════════════════════════════════════════════════════
    console.print()
    console.print("[bold cyan]solution[/bold cyan]")

    best_individual = scheduler.get_best_solution()
    decoded_schedule = decode_individual(
        best_individual,
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    final_hard = abs(best_individual.fitness.values[0])
    final_soft = abs(best_individual.fitness.values[1])

    console.print(f"  [dim]hard violations:[/dim] {final_hard:.0f}")
    console.print(f"  [dim]soft penalty:[/dim] {final_soft:.2f}")
    console.print(f"  [dim]sessions:[/dim] {len(decoded_schedule)}")

    # Calculate and display improvement percentages
    if (
        hasattr(scheduler, "initial_best_hard")
        and scheduler.initial_best_hard is not None
    ):
        initial_hard = scheduler.initial_best_hard
        initial_soft = scheduler.initial_best_soft

        # Calculate percentage reductions
        if initial_hard > 0:
            hc_reduction_pct = ((initial_hard - final_hard) / initial_hard) * 100
            console.print(
                f"  [dim]hard constraint improvement:[/dim] [green]{hc_reduction_pct:.1f}%[/green] "
                f"[dim](from {initial_hard:.0f})[/dim]"
            )

        if initial_soft > 0:
            sc_reduction_pct = ((initial_soft - final_soft) / initial_soft) * 100
            console.print(
                f"  [dim]soft constraint improvement:[/dim] [green]{sc_reduction_pct:.1f}%[/green] "
                f"[dim](from {initial_soft:.2f})[/dim]"
            )

    console.print()

    # Finalize logger
    logger.end_run(
        best_hard=best_individual.fitness.values[0],
        best_soft=best_individual.fitness.values[1],
        final_schedule_sessions=len(decoded_schedule),
    )

    # ═══════════════════════════════════════════════════════════════
    # RL VISUALIZATION (if RL was used)
    # ═══════════════════════════════════════════════════════════════
    if hasattr(scheduler, "rl_enabled") and scheduler.rl_enabled:
        console.print("[bold cyan]generating rl training visualizations[/bold cyan]")
        try:
            from pathlib import Path as PathLib

            from schedule_engine.rl.training.visualizer import generate_visualizations

            # TensorBoard logs location (in run output directory)
            tb_logdir = PathLib(output_dir) / "logs" / "tensorboard"

            if tb_logdir.exists():
                generate_visualizations(
                    tensorboard_logdir=tb_logdir,
                    output_dir=PathLib(output_dir),
                    experiment_name="rl_guided_ga",
                )
                console.print("  [dim]rl analysis plots saved to rl_analysis/[/dim]")
            else:
                console.print(
                    "  [dim]no tensorboard logs found, skipping rl plots[/dim]"
                )
        except Exception as e:
            console.print(f"  [yellow]warning:[/yellow] rl visualization failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # REPORT GENERATION
    # ═══════════════════════════════════════════════════════════════
    console.print("[bold cyan]generating reports[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("creating visualizations...", total=None)
        # Get population and heuristic_tracker with proper None handling
        population_list: list[Any] = (
            scheduler.population if scheduler.population is not None else []
        )
        heuristic_tracker_obj = getattr(scheduler, "heuristic_tracker", None)

        generate_reports(
            decoded_schedule=decoded_schedule,
            metrics=scheduler.metrics,
            population=population_list,
            qts=qts,
            output_dir=output_dir,
            course_map=context.courses,
            heuristic_tracker=heuristic_tracker_obj,
        )
        progress.update(task, completed=1)

    console.print(f"  [dim]saved:[/dim] {output_dir}")
    console.print()

    # ═══════════════════════════════════════════════════════════════
    # COMPLETE
    # ═══════════════════════════════════════════════════════════════
    console.print("[bold green]complete[/bold green]")
    console.print()
    console.print("[bold cyan]complete[/bold cyan]", style="cyan")
    console.print()
    console.print(f"[bold]output:[/bold] {output_dir}")
    console.print("  [dim]schedule.json[/dim] - final schedule")
    console.print("  [dim]calendar.pdf[/dim] - visual calendar")
    console.print("  [dim]run.log[/dim] - execution summary")
    console.print("  [dim]violations.log[/dim] - constraint violations")
    console.print("  [dim]data/metrics.csv[/dim] - generation metrics")
    console.print("  [dim]plots/[/dim] - visualizations")
    console.print()
    console.print(style="cyan")
    console.print()

    # Clean up multiprocessing pool
    if pool is not None:
        pool.close()
        pool.join()

    return {
        "best_individual": best_individual,
        "decoded_schedule": decoded_schedule,
        "metrics": scheduler.metrics,
        "output_path": output_dir,
        "qts": qts,
    }


def load_input_data(
    data_dir: str,
    config: Config | None = None,
) -> tuple[QuantumTimeSystem, SchedulingContext]:
    """
    Load and link all input entities.

    PARALLELIZED: JSON files are loaded concurrently using ThreadPoolExecutor.
    Expected speedup: 2-3x on I/O-bound operations.

    Only includes courses that are enrolled by at least one group,
    filtering out the rest of the university course database.

    Args:
        data_dir: Directory containing input JSON files

    Returns:
        Tuple of (QuantumTimeSystem, SchedulingContext)
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    start_time = time.time()

    # Initialize time system (must be first, used by other loaders)
    qts = QuantumTimeSystem()

    # ========================================
    # PARALLEL LOADING SECTION
    # ========================================
    # Load JSON files concurrently (I/O-bound operations)
    import os

    max_workers = get_cpu_count()  # Auto-detect all cores
    groups_path = os.path.join(data_dir, "Groups.json")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all loading tasks
        future_groups = executor.submit(load_groups, groups_path, qts)
        future_courses = executor.submit(
            load_courses, os.path.join(data_dir, "Course.json")
        )
        future_instructors = executor.submit(
            load_instructors, os.path.join(data_dir, "Instructors.json"), qts
        )
        future_rooms = executor.submit(
            load_rooms, os.path.join(data_dir, "Rooms.json"), qts
        )

        # Collect results (blocks until all complete)
        groups = future_groups.result()
        all_courses = future_courses.result()
        instructors = future_instructors.result()
        rooms = future_rooms.result()

    # ========================================
    # SEQUENTIAL PROCESSING SECTION
    # ========================================
    # (Depends on loaded data, must be sequential)

    # Step 2: Collect all enrolled course codes from groups
    enrolled_course_codes = set()
    for group in groups.values():
        enrolled_course_codes.update(group.enrolled_courses)

    print(
        f"[!info] Found {len(enrolled_course_codes)} unique course codes enrolled by groups"
    )

    # Step 3: Filter to only keep courses whose course_code is enrolled
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
        f"[!info] Filtered {len(courses)} course objects from {len(all_courses)} total in database"
    )
    print(f"[!info] ({excluded_count} courses excluded - not enrolled by any group)")

    # Step 4: Link relationships (only for enrolled courses)
    link_courses_and_groups(courses, groups)
    link_courses_and_instructors(courses, instructors)

    elapsed = time.time() - start_time
    print(f"[!info] Data loading completed in {elapsed:.2f}s (parallel)")

    # Step 5: Auto-derive cohort pairs from group hierarchy and merge overrides
    derived_pairs = derive_cohort_pairs_from_groups(groups_path)
    if config is None:
        raise ValueError("Config must be provided to derive cohort pairs")

    configured_pairs = list(getattr(config, "cohort_pairs", []))
    cohort_pairs = _merge_cohort_pairs(derived_pairs, configured_pairs)
    config.cohort_pairs = cohort_pairs

    manual_override_count = max(0, len(cohort_pairs) - len(derived_pairs))
    print(
        "[!info] Cohort pairs → derived: "
        f"{len(derived_pairs)} | manual overrides: {manual_override_count}"
    )

    # Create context with filtered courses
    # Type assertion: config must be provided at this point
    assert config is not None, "Config must be provided to load_input_data"

    context = SchedulingContext(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        available_quanta=list(qts.get_all_operating_quanta()),
        config=config,
        cohort_pairs=cohort_pairs,
    )

    return qts, context


def _merge_cohort_pairs(
    derived_pairs: Iterable[tuple[str, str]],
    configured_pairs: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Merge auto-derived cohort pairs with manually configured overrides."""

    merged: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _normalize(pair: tuple[str, str]) -> tuple[str, str] | None:
        left, right = pair
        left_clean = left.strip()
        right_clean = right.strip()
        if not left_clean or not right_clean:
            return None
        return left_clean, right_clean

    for source in (derived_pairs, configured_pairs):
        for pair in source:
            normalized = _normalize(pair)
            if normalized is None:
                continue

            canonical_parts: list[str] = sorted(
                (normalized[0].lower(), normalized[1].lower())
            )
            canonical = (canonical_parts[0], canonical_parts[1])
            if canonical in seen:
                continue

            seen.add(canonical)
            merged.append(normalized)

    return merged
