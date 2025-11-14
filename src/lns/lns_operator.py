"""
LNS-CP repair operator for genetic algorithm.

This module implements the Large Neighborhood Search operator that uses
CP-SAT to repair hard constraint violations in GA individuals.
"""

from typing import List, Dict
import logging

from src.ga.sessiongene import SessionGene
from src.entities.course import Course
from src.entities.instructor import Instructor
from src.entities.group import Group
from src.entities.room import Room
from src.lns.conflict_detection import (
    find_hard_conflict_sessions,
    select_worst_conflicts,
)
from src.lns.cp_repair import repair_with_cp_sat

# Logger setup
logger = logging.getLogger(__name__)


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

    def __repr__(self):
        success_rate = (
            self.successful_repairs / self.total_attempts * 100
            if self.total_attempts > 0
            else 0.0
        )
        return (
            f"LNSRepairStats("
            f"attempts={self.total_attempts}, "
            f"success={self.successful_repairs}, "
            f"failed={self.failed_repairs}, "
            f"success_rate={success_rate:.1f}%, "
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


def lns_cp_repair(
    individual: List[SessionGene],
    courses: Dict[tuple, Course],
    instructors: Dict[str, Instructor],
    groups: Dict[str, Group],
    rooms: Dict[str, Room],
    max_subproblem_size: int = 20,
    cp_time_limit: float = 10.0,
) -> List[SessionGene]:
    """
    Apply LNS-CP repair to an individual with hard constraint violations.

    Algorithm (LNS-CP Repair):
    1. Detect hard constraint violations (conflict detection)
    2. Extract conflicted sessions (destroy phase)
    3. If conflicts > max_subproblem_size, select worst subset
    4. Create partial schedule (remove conflicted sessions)
    5. Repair using CP-SAT (repair phase)
    6. Reintegrate repaired sessions if successful
    7. Return repaired individual or original if repair fails

    Args:
        individual: GA individual (chromosome) to repair
        courses: Course dictionary
        instructors: Instructor dictionary
        groups: Group dictionary
        rooms: Room dictionary
        max_subproblem_size: Maximum number of sessions to repair at once
        cp_time_limit: Time limit for CP-SAT solver in seconds

    Returns:
        Repaired individual if successful, original individual otherwise
    """
    import time

    start_time = time.time()

    # Update statistics
    _lns_stats.total_attempts += 1

    # Step 1: Detect conflicts
    conflicted_indices, violations = find_hard_conflict_sessions(
        individual, courses, instructors, groups, rooms
    )

    if not conflicted_indices:
        logger.debug("LNS-CP: No conflicts detected, skipping repair")
        return individual

    total_conflicts = sum(v.violation_count for v in violations)
    _lns_stats.total_conflicts_detected += total_conflicts

    logger.info(
        f"LNS-CP: Detected {len(conflicted_indices)} conflicted sessions "
        f"with {total_conflicts} total violations"
    )

    # Step 2: Select subset if too large
    if len(conflicted_indices) > max_subproblem_size:
        logger.info(
            f"LNS-CP: Reducing subproblem from {len(conflicted_indices)} "
            f"to {max_subproblem_size} sessions"
        )
        conflicted_indices = select_worst_conflicts(
            conflicted_indices, violations, max_subproblem_size
        )

    _lns_stats.avg_subproblem_size = (
        _lns_stats.avg_subproblem_size * (_lns_stats.total_attempts - 1)
        + len(conflicted_indices)
    ) / _lns_stats.total_attempts

    # Step 3: Extract conflicted sessions (destroy phase)
    conflicted_sessions = [individual[i] for i in conflicted_indices]

    # Step 4: Create partial schedule (remove conflicted sessions)
    conflicted_set = set(conflicted_indices)
    partial_schedule = [
        individual[i] for i in range(len(individual)) if i not in conflicted_set
    ]

    logger.info(
        f"LNS-CP: Partial schedule has {len(partial_schedule)} sessions, "
        f"repairing {len(conflicted_sessions)} sessions"
    )

    # Step 5: Repair using CP-SAT (repair phase)
    repaired_sessions = repair_with_cp_sat(
        conflicted_sessions=conflicted_sessions,
        partial_schedule=partial_schedule,
        courses=courses,
        instructors=instructors,
        groups=groups,
        rooms=rooms,
        time_limit_seconds=cp_time_limit,
    )

    # Step 6: Reintegrate or return original
    if repaired_sessions is not None:
        # Success: reintegrate repaired sessions
        new_individual = list(individual)  # Copy
        for idx, repaired_session in zip(conflicted_indices, repaired_sessions):
            new_individual[idx] = repaired_session

        # Update statistics
        _lns_stats.successful_repairs += 1
        _lns_stats.total_conflicts_repaired += total_conflicts

        repair_time = time.time() - start_time
        _lns_stats.total_repair_time += repair_time

        logger.info(
            f"LNS-CP: Repair SUCCESSFUL (time={repair_time:.2f}s, "
            f"repaired {len(conflicted_sessions)} sessions)"
        )

        return new_individual
    else:
        # Failure: return original
        _lns_stats.failed_repairs += 1

        repair_time = time.time() - start_time
        _lns_stats.total_repair_time += repair_time

        logger.warning(
            f"LNS-CP: Repair FAILED (time={repair_time:.2f}s), "
            f"returning original individual"
        )

        return individual


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
) -> bool:
    """
    Determine if LNS-CP repair should be triggered.

    Triggers on:
    - Regular intervals (every trigger_interval generations)
    - Stagnation detection (stagnation_counter >= stagnation_threshold)

    Args:
        generation: Current generation number
        trigger_interval: Number of generations between regular triggers
        stagnation_counter: Current stagnation counter
        stagnation_threshold: Threshold for stagnation-based triggering

    Returns:
        True if LNS-CP should be triggered, False otherwise
    """
    # Trigger on interval
    if generation > 0 and generation % trigger_interval == 0:
        logger.info(f"LNS-CP: Triggered by interval (gen {generation})")
        return True

    # Trigger on stagnation
    if stagnation_counter >= stagnation_threshold:
        logger.info(
            f"LNS-CP: Triggered by stagnation "
            f"(counter={stagnation_counter}, threshold={stagnation_threshold})"
        )
        return True

    return False
