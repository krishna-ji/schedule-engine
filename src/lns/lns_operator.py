"""
LNS repair operator for genetic algorithm.

This module implements the Large Neighborhood Search operator with multiple
repair strategies: CP-SAT, heuristic (greedy/local search), or hybrid.
"""

from typing import List, Dict, Literal
import logging
from rich.console import Console

from src.ga.sessiongene import SessionGene
from src.ga.individual import create_individual
from src.entities.course import Course
from src.entities.instructor import Instructor
from src.entities.group import Group
from src.entities.room import Room
from src.lns.conflict_detection import (
    find_hard_conflict_sessions,
    select_worst_conflicts,
)
from src.lns.cp_repair import repair_with_cp_sat
from src.lns.heuristic_repair import repair_with_heuristic
from src.lns.diagnostics import (
    SubproblemDiagnostics,
    build_conflict_graph,
    expand_neighborhood_bfs,
)

# Logger setup
logger = logging.getLogger(__name__)

# Console initialization
console = Console()


class LNSRepairStats:
    """Statistics for LNS repair operations."""

    def __init__(self):
        self.total_attempts = 0
        self.successful_repairs = 0
        self.failed_repairs = 0
        self.total_conflicts_detected = 0
        self.total_conflicts_repaired = 0
        self.avg_subproblem_size = 0.0
        self.total_repair_time = 0.0
        # Strategy-specific stats
        self.heuristic_attempts = 0
        self.heuristic_success = 0
        self.cp_attempts = 0
        self.cp_success = 0
        self.pre_check_skips = 0

    def __repr__(self):
        success_rate = (
            self.successful_repairs / self.total_attempts * 100
            if self.total_attempts > 0
            else 0.0
        )
        heur_rate = (
            self.heuristic_success / self.heuristic_attempts * 100
            if self.heuristic_attempts > 0
            else 0.0
        )
        cp_rate = (
            self.cp_success / self.cp_attempts * 100 if self.cp_attempts > 0 else 0.0
        )
        return (
            f"LNSRepairStats("
            f"attempts={self.total_attempts}, "
            f"success={self.successful_repairs}, "
            f"failed={self.failed_repairs}, "
            f"success_rate={success_rate:.1f}%, "
            f"heuristic={self.heuristic_attempts}({heur_rate:.1f}%), "
            f"cp={self.cp_attempts}({cp_rate:.1f}%), "
            f"pre_check_skips={self.pre_check_skips}, "
            f"avg_subproblem_size={self.avg_subproblem_size:.1f})"
        )


# Global statistics tracker
_lns_stats = LNSRepairStats()


def get_lns_stats() -> LNSRepairStats:
    """Get current LNS repair statistics."""
    return _lns_stats


def reset_lns_stats():
    """Reset LNS repair statistics."""
    global _lns_stats
    _lns_stats = LNSRepairStats()


def lns_repair(
    individual: List[SessionGene],
    courses: Dict[tuple, Course],
    instructors: Dict[str, Instructor],
    groups: Dict[str, Group],
    rooms: Dict[str, Room],
    repair_strategy: Literal["cp", "heuristic", "hybrid"] = "hybrid",
    max_subproblem_size: int = 20,
    min_subproblem_size: int = 4,
    expand_hops: int = 0,
    cp_time_limit: float = 10.0,
    heuristic_max_iterations: int = 500,
    heuristic_time_limit: float = 5.0,
    enable_diagnostics: bool = True,
    pre_check_feasibility: bool = True,
) -> List[SessionGene]:
    """
    Apply LNS repair to an individual with hard constraint violations.

    Supports three repair strategies:
    - 'cp': CP-SAT only
    - 'heuristic': Greedy/local search only
    - 'hybrid': Try heuristic first, escalate to CP if needed

    Algorithm:
    1. Detect hard constraint violations
    2. Extract conflicted sessions (destroy phase)
    3. Optionally expand neighborhood via conflict graph
    4. Run diagnostics and pre-feasibility check
    5. Apply repair strategy (heuristic/CP/hybrid)
    6. Reintegrate repaired sessions if successful

    Args:
        individual: GA individual to repair
        courses, instructors, groups, rooms: Entity dictionaries
        repair_strategy: 'cp', 'heuristic', or 'hybrid'
        max_subproblem_size: Maximum sessions to repair at once
        min_subproblem_size: Minimum sessions (skip if below)
        expand_hops: Expand neighborhood N hops in conflict graph (0=disabled)
        cp_time_limit: CP-SAT time limit
        heuristic_max_iterations: Heuristic local search iterations
        heuristic_time_limit: Heuristic time limit
        enable_diagnostics: Log detailed diagnostics
        pre_check_feasibility: Run pre-check before CP

    Returns:
        Repaired individual if successful, original otherwise
    """
    import time

    start_time = time.time()
    _lns_stats.total_attempts += 1

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
        violation_types = {}
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
        f"LNS: Partial={len(partial_schedule)}, repair={len(conflicted_sessions)}, strategy={repair_strategy}"
    )

    # Step 5: Diagnostics and pre-check
    diagnostics = SubproblemDiagnostics(
        conflicted_sessions, partial_schedule, courses, instructors, groups, rooms
    )

    if enable_diagnostics:
        diagnostics.log_subproblem_summary()

    feasible = True
    reason = ""
    if pre_check_feasibility and repair_strategy in ("cp", "hybrid"):
        feasible, reason = diagnostics.pre_check_feasibility()
        if enable_diagnostics:
            if feasible:
                console.print("[green]   [LNS] Pre-check: PASSED (OK)[/green]")
            else:
                console.print(f"[red]   [LNS] Pre-check: FAILED (X) - {reason}[/red]")
        if not feasible:
            _lns_stats.pre_check_skips += 1
            logger.warning(f"LNS: Pre-check INFEASIBLE - {reason}")
            if repair_strategy == "cp":
                # Skip CP entirely
                logger.info("LNS: Skipping CP due to pre-check failure")
                if enable_diagnostics:
                    console.print(
                        "[red]   [LNS] Skipping CP-SAT repair (pre-check failed)[/red]"
                    )
                _lns_stats.failed_repairs += 1
                return individual

    # Step 6: Apply repair strategy
    repaired_sessions = None

    if repair_strategy == "heuristic":
        _lns_stats.heuristic_attempts += 1
        if enable_diagnostics:
            console.print(
                f"[blue]   [LNS] Attempting heuristic repair (max_iter={heuristic_max_iterations}, timeout={heuristic_time_limit}s)...[/blue]"
            )
        heur_start = time.time()
        repaired_sessions = repair_with_heuristic(
            conflicted_sessions,
            partial_schedule,
            courses,
            instructors,
            groups,
            rooms,
            heuristic_max_iterations,
            heuristic_time_limit,
        )
        heur_time = time.time() - heur_start
        if repaired_sessions:
            _lns_stats.heuristic_success += 1
            if enable_diagnostics:
                console.print(
                    f"[green]   [LNS] Heuristic repair: SUCCESS (OK) ({heur_time:.2f}s)[/green]"
                )

    elif repair_strategy == "cp":
        if feasible:  # Only try CP if pre-check passed
            _lns_stats.cp_attempts += 1
            if enable_diagnostics:
                console.print(
                    f"[blue]   [LNS] Attempting CP-SAT repair (timeout={cp_time_limit}s)...[/blue]"
                )
            cp_start = time.time()
            repaired_sessions = repair_with_cp_sat(
                conflicted_sessions,
                partial_schedule,
                courses,
                instructors,
                groups,
                rooms,
                cp_time_limit,
            )
            cp_time = time.time() - cp_start
            if repaired_sessions:
                _lns_stats.cp_success += 1
                if enable_diagnostics:
                    console.print(
                        f"[green]   [LNS] CP-SAT repair: SUCCESS (OK) ({cp_time:.2f}s)[/green]"
                    )
            else:
                if enable_diagnostics:
                    console.print(
                        f"[red]   [LNS] CP-SAT repair: FAILED (X) ({cp_time:.2f}s)[/red]"
                    )

    elif repair_strategy == "hybrid":
        # Try heuristic first
        _lns_stats.heuristic_attempts += 1
        if enable_diagnostics:
            console.print(
                f"[blue]   [LNS-Hybrid] Step 1: Attempting heuristic repair (max_iter={heuristic_max_iterations}, timeout={heuristic_time_limit}s)...[/blue]"
            )
        heur_start = time.time()
        repaired_sessions = repair_with_heuristic(
            conflicted_sessions,
            partial_schedule,
            courses,
            instructors,
            groups,
            rooms,
            heuristic_max_iterations,
            heuristic_time_limit,
        )
        heur_time = time.time() - heur_start

        if repaired_sessions:
            _lns_stats.heuristic_success += 1
            logger.info("LNS-Hybrid: Heuristic succeeded")
            if enable_diagnostics:
                console.print(
                    f"[green]   [LNS-Hybrid] Heuristic repair: SUCCESS (OK) ({heur_time:.2f}s, no escalation needed)[/green]"
                )
        elif feasible:
            # Escalate to CP if heuristic failed and pre-check passed
            logger.info("LNS-Hybrid: Heuristic failed, escalating to CP-SAT")
            if enable_diagnostics:
                console.print(
                    f"[yellow]   [LNS-Hybrid] Heuristic failed ({heur_time:.2f}s), escalating to CP-SAT...[/yellow]"
                )
            _lns_stats.cp_attempts += 1
            cp_start = time.time()
            repaired_sessions = repair_with_cp_sat(
                conflicted_sessions,
                partial_schedule,
                courses,
                instructors,
                groups,
                rooms,
                cp_time_limit,
            )
            cp_time = time.time() - cp_start
            if repaired_sessions:
                _lns_stats.cp_success += 1
                if enable_diagnostics:
                    console.print(
                        f"[green]   [LNS-Hybrid] CP-SAT repair: SUCCESS (OK) ({cp_time:.2f}s)[/green]"
                    )
            else:
                if enable_diagnostics:
                    console.print(
                        f"[red]   [LNS-Hybrid] CP-SAT repair: FAILED (X) ({cp_time:.2f}s, all strategies exhausted)[/red]"
                    )
        else:
            if enable_diagnostics:
                console.print(
                    f"[red]   [LNS-Hybrid] Heuristic failed ({heur_time:.2f}s), cannot escalate (pre-check failed)[/red]"
                )

    # Step 7: Reintegrate or return original
    if repaired_sessions is not None:
        new_genes = list(individual)
        for idx, repaired_session in zip(conflicted_indices, repaired_sessions):
            new_genes[idx] = repaired_session

        # Wrap into DEAP Individual
        new_individual = create_individual(new_genes)

        _lns_stats.successful_repairs += 1
        _lns_stats.total_conflicts_repaired += total_conflicts

        repair_time = time.time() - start_time
        _lns_stats.total_repair_time += repair_time

        logger.info(
            f"LNS: Repair SUCCESS (strategy={repair_strategy}, time={repair_time:.2f}s)"
        )
        if enable_diagnostics:
            console.print(
                f"[bold green]   [LNS] (OK) Repair SUCCESSFUL: {len(conflicted_sessions)} sessions repaired "
                f"(strategy={repair_strategy}, total_time={repair_time:.2f}s)[/bold green]"
            )
        return new_individual
    else:
        _lns_stats.failed_repairs += 1
        repair_time = time.time() - start_time
        _lns_stats.total_repair_time += repair_time

        logger.warning(
            f"LNS: Repair FAILED (strategy={repair_strategy}, time={repair_time:.2f}s)"
        )
        if enable_diagnostics:
            console.print(
                f"[bold red]   [LNS] (X) Repair FAILED: All strategies exhausted "
                f"(strategy={repair_strategy}, total_time={repair_time:.2f}s)[/bold red]"
            )
        return individual


# Backward compatibility alias
def lns_cp_repair(
    individual: List[SessionGene],
    courses: Dict[tuple, Course],
    instructors: Dict[str, Instructor],
    groups: Dict[str, Group],
    rooms: Dict[str, Room],
    max_subproblem_size: int = 20,
    cp_time_limit: float = 10.0,
) -> List[SessionGene]:
    """Legacy function - calls lns_repair with 'cp' strategy."""
    return lns_repair(
        individual,
        courses,
        instructors,
        groups,
        rooms,
        repair_strategy="cp",
        max_subproblem_size=max_subproblem_size,
        cp_time_limit=cp_time_limit,
    )


def apply_lns_to_population(
    population: List[List[SessionGene]],
    courses: Dict[tuple, Course],
    instructors: Dict[str, Instructor],
    groups: Dict[str, Group],
    rooms: Dict[str, Room],
    num_individuals: int = 1,
    max_subproblem_size: int = 20,
    cp_time_limit: float = 10.0,
) -> List[List[SessionGene]]:
    """
    Apply LNS-CP repair to the best individuals in a population.

    Args:
        population: GA population (list of individuals)
        courses: Course dictionary
        instructors: Instructor dictionary
        groups: Group dictionary
        rooms: Room dictionary
        num_individuals: Number of best individuals to repair
        max_subproblem_size: Maximum subproblem size
        cp_time_limit: CP-SAT time limit

    Returns:
        Population with repaired individuals
    """
    if not population:
        return population

    # Sort population by fitness (assuming fitness is assigned)
    # Note: In DEAP, fitness is accessed via individual.fitness.values
    # For now, we'll repair the first num_individuals

    new_population = list(population)  # Copy

    for i in range(min(num_individuals, len(population))):
        logger.info(f"LNS-CP: Repairing individual {i+1}/{num_individuals}")
        new_population[i] = lns_cp_repair(
            individual=population[i],
            courses=courses,
            instructors=instructors,
            groups=groups,
            rooms=rooms,
            max_subproblem_size=max_subproblem_size,
            cp_time_limit=cp_time_limit,
        )

    return new_population


def should_trigger_lns_repair(
    generation: int,
    trigger_interval: int,
    stagnation_counter: int,
    stagnation_threshold: int,
    force_trigger_generations: List[int] = None,
) -> bool:
    """
    Determine if LNS-CP repair should be triggered.

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
        True if LNS-CP should be triggered, False otherwise
    """
    # Priority 1: Force trigger on specific generations (for testing/validation)
    if force_trigger_generations and generation in force_trigger_generations:
        logger.info(
            f"LNS: FORCED trigger on gen {generation} (validation/testing mode)"
        )
        return True

    # Priority 2: Trigger on interval
    if generation > 0 and generation % trigger_interval == 0:
        logger.info(f"LNS: Triggered by interval (gen {generation})")
        return True

    # Priority 3: Trigger on stagnation
    if stagnation_counter >= stagnation_threshold:
        logger.info(
            f"LNS: Triggered by stagnation "
            f"(counter={stagnation_counter}, threshold={stagnation_threshold})"
        )
        return True

    return False
