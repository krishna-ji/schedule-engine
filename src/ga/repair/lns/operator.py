"""LNS repair operator for genetic algorithm.

This module implements the Large Neighborhood Search operator with IGLS
(Iterated Guided Local Search) as the repair strategy.
"""

import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.ga.core.individual import create_individual
from src.ga.repair.conflict_detection import (
    find_hard_conflict_sessions,
    select_worst_conflicts,
)
from src.ga.repair.heuristic_repair import repair_with_heuristic
from src.ga.repair.lns.diagnostics import (
    SubproblemDiagnostics,
    build_conflict_graph,
    expand_neighborhood_bfs,
)
from src.utils.console_service import get_console
from src.utils.parallel_worker import get_worker_context, init_worker
from src.utils.system_info import get_cpu_count

# Logger setup
logger = logging.getLogger(__name__)

# Console initialization
console = get_console()


class LNSRepairStats:
    """Statistics for LNS repair operations."""

    def __init__(self) -> None:
        self.total_attempts: int = 0
        self.successful_repairs: int = 0
        self.failed_repairs: int = 0
        self.total_conflicts_detected: int = 0
        self.total_conflicts_repaired: int = 0
        self.avg_subproblem_size: float = 0.0
        self.total_repair_time: float = 0.0
        # Strategy-specific stats
        self.igls_attempts: int = 0
        self.igls_success: int = 0

    def __repr__(self) -> str:
        success_rate = (
            self.successful_repairs / self.total_attempts * 100
            if self.total_attempts > 0
            else 0.0
        )
        igls_rate = (
            self.igls_success / self.igls_attempts * 100
            if self.igls_attempts > 0
            else 0.0
        )
        return (
            f"LNSRepairStats("
            f"attempts={self.total_attempts}, "
            f"success={self.successful_repairs}, "
            f"failed={self.failed_repairs}, "
            f"success_rate={success_rate:.1f}%, "
            f"igls={self.igls_attempts}({igls_rate:.1f}%), "
            f"avg_subproblem_size={self.avg_subproblem_size:.1f})"
        )


# Global statistics tracker
_lns_stats: LNSRepairStats = LNSRepairStats()


def get_lns_stats() -> LNSRepairStats:
    """Get current LNS repair statistics."""
    return _lns_stats


def reset_lns_stats() -> None:
    """Reset LNS repair statistics."""
    global _lns_stats
    _lns_stats = LNSRepairStats()


def lns_igls_repair(
    individual: list[SessionGene],
    courses: dict[tuple, Course] | None = None,
    instructors: dict[str, Instructor] | None = None,
    groups: dict[str, Group] | None = None,
    rooms: dict[str, Room] | None = None,
    max_subproblem_size: int = 20,
    min_subproblem_size: int = 4,
    expand_hops: int = 0,
    igls_max_iterations: int = 500,
    igls_time_limit: float = 5.0,
    enable_diagnostics: bool = True,
) -> list[SessionGene]:
    """
    Apply LNS-IGLS repair to an individual with hard constraint violations.

    Uses IGLS (Iterated Guided Local Search) as the repair strategy.

    Algorithm:
    1. Detect hard constraint violations
    2. Extract conflicted sessions (destroy phase)
    3. Optionally expand neighborhood via conflict graph
    4. Run diagnostics
    5. Apply IGLS repair to the subproblem
    6. Reintegrate repaired sessions if successful

    Args:
        individual: GA individual to repair
        courses, instructors, groups, rooms: Entity dictionaries (Optional if in worker)
        max_subproblem_size: Maximum sessions to repair at once
        min_subproblem_size: Minimum sessions (skip if below)
        expand_hops: Expand neighborhood N hops in conflict graph (0=disabled)
        igls_max_iterations: IGLS local search iterations
        igls_time_limit: IGLS time limit
        enable_diagnostics: Log detailed diagnostics

    Returns:
        Repaired individual if successful, original otherwise
    """
    # If context is missing, try to get from worker global state
    if courses is None:
        try:
            worker_data = get_worker_context()
            courses = worker_data["courses"]
            instructors = worker_data["instructors"]
            groups = worker_data["groups"]
            rooms = worker_data["rooms"]
        except RuntimeError:
            # Should not happen if initialized correctly
            # But if it does, we can't proceed without context
            logger.error("Context missing in LNS repair worker")
            return individual

    import time

    start_time = time.time()
    _lns_stats.total_attempts += 1

    # Assert non-None for type checking (already guaranteed by context loading above)
    assert (
        courses is not None
        and instructors is not None
        and groups is not None
        and rooms is not None
    )

    # Step 1: Detect conflicts
    conflicted_indices, violations = find_hard_conflict_sessions(
        individual, courses, instructors, groups, rooms
    )

    if not conflicted_indices:
        logger.debug("LNS: No conflicts detected, skipping repair")
        if enable_diagnostics:
            console.print("[dim]   [LNS] No conflicts detected, skipping repair[/dim]")
        return individual

    total_conflicts = sum(v.violation_count for v in violations)
    _lns_stats.total_conflicts_detected += total_conflicts

    logger.info(
        f"LNS: Detected {len(conflicted_indices)} conflicted sessions "
        f"with {total_conflicts} total violations"
    )
    if enable_diagnostics:
        console.print(
            f"[yellow]   [LNS] Detected {len(conflicted_indices)} conflicted sessions with {total_conflicts} violations[/yellow]"
        )
        violation_types: dict[str, int] = {}
        for v in violations:
            vtype = v.constraint_name
            violation_types[vtype] = violation_types.get(vtype, 0) + v.violation_count
        console.print(f"[dim]   [LNS] Violation breakdown: {violation_types}[/dim]")

    # Step 2: Expand neighborhood if requested
    if expand_hops > 0:
        conflict_graph = build_conflict_graph(
            individual, courses, instructors, groups, rooms
        )
        original_size = len(conflicted_indices)
        conflicted_indices = expand_neighborhood_bfs(
            conflicted_indices, conflict_graph, max_subproblem_size, expand_hops
        )
        if len(conflicted_indices) > original_size:
            logger.info(
                f"LNS: Expanded neighborhood from {original_size} to {len(conflicted_indices)} sessions"
            )
            if enable_diagnostics:
                console.print(
                    f"[cyan]   [LNS] Expanded neighborhood: {original_size} -> {len(conflicted_indices)} sessions ({expand_hops} hops)[/cyan]"
                )

    # Step 3: Check size constraints
    if len(conflicted_indices) < min_subproblem_size:
        logger.debug(
            f"LNS: Subproblem too small ({len(conflicted_indices)} < {min_subproblem_size}), skipping"
        )
        if enable_diagnostics:
            console.print(
                f"[dim]   [LNS] Subproblem too small ({len(conflicted_indices)} < {min_subproblem_size}), skipping[/dim]"
            )
        return individual

    if len(conflicted_indices) > max_subproblem_size:
        logger.info(
            f"LNS: Reducing subproblem from {len(conflicted_indices)} to {max_subproblem_size}"
        )
        if enable_diagnostics:
            console.print(
                f"[yellow]   [LNS] Reducing subproblem: {len(conflicted_indices)} -> {max_subproblem_size} (worst conflicts)[/yellow]"
            )
        conflicted_indices = select_worst_conflicts(
            conflicted_indices, violations, max_subproblem_size
        )

    _lns_stats.avg_subproblem_size = (
        _lns_stats.avg_subproblem_size * (_lns_stats.total_attempts - 1)
        + len(conflicted_indices)
    ) / _lns_stats.total_attempts

    # Step 4: Extract conflicted sessions and create partial schedule
    conflicted_sessions = [individual[i] for i in conflicted_indices]
    conflicted_set = set(conflicted_indices)
    partial_schedule = [
        individual[i] for i in range(len(individual)) if i not in conflicted_set
    ]

    logger.info(
        f"LNS-IGLS: Partial={len(partial_schedule)}, repair={len(conflicted_sessions)}"
    )

    # Step 5: Diagnostics
    diagnostics = SubproblemDiagnostics(
        conflicted_sessions, partial_schedule, courses, instructors, groups, rooms
    )

    if enable_diagnostics:
        diagnostics.log_subproblem_summary()

    # Step 6: Apply IGLS repair
    _lns_stats.igls_attempts += 1
    if enable_diagnostics:
        console.print(
            f"[blue]   [LNS-IGLS] Attempting IGLS repair (max_iter={igls_max_iterations}, timeout={igls_time_limit}s)...[/blue]"
        )

    repair_start = time.time()
    repaired_sessions = repair_with_heuristic(
        conflicted_sessions,
        partial_schedule,
        courses,
        instructors,
        groups,
        rooms,
        igls_max_iterations,
        igls_time_limit,
    )
    repair_time_elapsed = time.time() - repair_start

    if repaired_sessions:
        _lns_stats.igls_success += 1
        if enable_diagnostics:
            console.print(
                f"[green]   [LNS-IGLS] IGLS repair: SUCCESS (OK) ({repair_time_elapsed:.2f}s)[/green]"
            )

    elif enable_diagnostics:
        console.print(
            f"[red]   [LNS-IGLS] IGLS repair: FAILED (X) ({repair_time_elapsed:.2f}s)[/red]"
        )

    # Step 7: Reintegrate or return original
    if repaired_sessions is not None:
        new_genes = list(individual)
        for idx, repaired_session in zip(
            conflicted_indices, repaired_sessions, strict=False
        ):
            new_genes[idx] = repaired_session

        # Wrap into DEAP Individual
        new_individual = create_individual(new_genes)

        _lns_stats.successful_repairs += 1
        _lns_stats.total_conflicts_repaired += total_conflicts

        repair_time = time.time() - start_time
        _lns_stats.total_repair_time += repair_time

        logger.info(f"LNS-IGLS: Repair SUCCESS (time={repair_time:.2f}s)")
        if enable_diagnostics:
            console.print(
                f"[bold green]   [LNS-IGLS] (OK) Repair SUCCESSFUL: {len(conflicted_sessions)} sessions repaired "
                f"(total_time={repair_time:.2f}s)[/bold green]"
            )
        # Type cast for mypy (create_individual returns DEAP Individual which inherits from list)
        return list(new_individual)
    _lns_stats.failed_repairs += 1
    repair_time = time.time() - start_time
    _lns_stats.total_repair_time += repair_time

    logger.warning(f"LNS-IGLS: Repair FAILED (time={repair_time:.2f}s)")
    if enable_diagnostics:
        console.print(
            f"[bold red]   [LNS-IGLS] (X) Repair FAILED: IGLS could not repair subproblem "
            f"(total_time={repair_time:.2f}s)[/bold red]"
        )
    return individual


def apply_lns_to_population(
    population: list[list[SessionGene]],
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
    num_individuals: int = 1,
    max_subproblem_size: int = 20,
    igls_time_limit: float = 5.0,
    data_dir: str = "data",  # Added data_dir
) -> list[list[SessionGene]]:
    """
    Apply LNS-IGLS repair to the best individuals in a population.

    Args:
        population: GA population (list of individuals)
        courses: Course dictionary
        instructors: Instructor dictionary
        groups: Group dictionary
        rooms: Room dictionary
        num_individuals: Number of best individuals to repair
        max_subproblem_size: Maximum subproblem size
        igls_time_limit: IGLS time limit
        data_dir: Directory containing input JSON files (for worker init)

    Returns:
        Population with repaired individuals
    """
    if not population:
        return population

    new_population = list(population)  # Copy
    num_to_repair = min(num_individuals, len(population))

    # Parallelize repair for 3-8x speedup (each repair takes ~5-30 seconds)
    if num_to_repair >= 2:
        max_workers = get_cpu_count()
        logger.info(
            f"LNS-IGLS: Repairing {num_to_repair} individuals in parallel ({max_workers} workers)"
        )

        # Prepare repair tasks - DO NOT pass context
        repair_tasks = [
            (
                i,
                population[i],
                None,  # courses
                None,  # instructors
                None,  # groups
                None,  # rooms
                max_subproblem_size,
                igls_time_limit,
            )
            for i in range(num_to_repair)
        ]

        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_worker,
                initargs=(data_dir, random.randint(0, 10000)),
            ) as executor:
                # Submit all repair jobs
                future_to_index = {
                    executor.submit(
                        lns_igls_repair,
                        individual=task[1],
                        courses=task[2],
                        instructors=task[3],
                        groups=task[4],
                        rooms=task[5],
                        max_subproblem_size=task[6],
                        igls_time_limit=task[7],
                    ): task[0]
                    for task in repair_tasks
                }

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        repaired_individual = future.result()
                        new_population[index] = repaired_individual
                        logger.info(
                            f"LNS-IGLS: Completed repair {index + 1}/{num_to_repair}"
                        )
                    except Exception as e:
                        logger.error(f"LNS-IGLS: Repair {index + 1} failed: {e}")
                        # Keep original individual if repair fails

        except Exception as e:
            logger.warning(
                f"Parallel LNS repair failed: {e}, falling back to sequential"
            )
            # Fallback to sequential
            for i in range(num_to_repair):
                logger.info(
                    f"LNS-IGLS: Repairing individual {i + 1}/{num_to_repair} (sequential fallback)"
                )
                new_population[i] = lns_igls_repair(
                    individual=population[i],
                    courses=courses,
                    instructors=instructors,
                    groups=groups,
                    rooms=rooms,
                    max_subproblem_size=max_subproblem_size,
                    igls_time_limit=igls_time_limit,
                )
    else:
        # Single individual - no need to parallelize
        for i in range(num_to_repair):
            logger.info(f"LNS-IGLS: Repairing individual {i + 1}/{num_to_repair}")
            new_population[i] = lns_igls_repair(
                individual=population[i],
                courses=courses,
                instructors=instructors,
                groups=groups,
                rooms=rooms,
                max_subproblem_size=max_subproblem_size,
                igls_time_limit=igls_time_limit,
            )

    return new_population


def should_trigger_lns_repair(
    generation: int,
    trigger_interval: int,
    stagnation_counter: int,
    stagnation_threshold: int,
    force_trigger_generations: list[int] | None = None,
) -> bool:
    """
    Determine if LNS-IGLS repair should be triggered.

    Triggers on:
    - Forced generations (highest priority, for testing/validation)
    - Regular intervals (every trigger_interval generations)
    - Stagnation detection (stagnation_counter >= stagnation_threshold)

    Args:
        generation: Current generation number
        trigger_interval: Number of generations between regular triggers
        stagnation_counter: Current stagnation counter
        stagnation_threshold: Threshold for stagnation-based triggering
        force_trigger_generations: List of generations to force trigger (optional)

    Returns:
        True if LNS-IGLS should be triggered, False otherwise
    """
    # Priority 1: Force trigger on specific generations (for testing/validation)
    if force_trigger_generations and generation in force_trigger_generations:
        logger.info(
            f"LNS-IGLS: FORCED trigger on gen {generation} (validation/testing mode)"
        )
        return True

    # Priority 2: Trigger on interval
    if generation > 0 and generation % trigger_interval == 0:
        logger.info(f"LNS-IGLS: Triggered by interval (gen {generation})")
        return True

    # Priority 3: Trigger on stagnation
    if stagnation_counter >= stagnation_threshold:
        logger.info(
            f"LNS-IGLS: Triggered by stagnation "
            f"(counter={stagnation_counter}, threshold={stagnation_threshold})"
        )
        return True

    return False
