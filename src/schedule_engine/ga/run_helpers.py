"""
Core Notebook Utilities.

Provides essential functions for Jupyter notebook experiments:
- Data loading from JSON files
- Individual creation (random/greedy)
- Fitness evaluation
- DEAP toolbox setup
- Evolution statistics tracking

DRY Principle: All notebooks import from here instead of duplicating code.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from deap import base, creator, tools

from schedule_engine.domain.course import Course
from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.group import Group
from schedule_engine.domain.instructor import Instructor
from schedule_engine.domain.room import Room
from schedule_engine.domain.types import SchedulingContext
from schedule_engine.io.decoder import decode_individual
from schedule_engine.io.time_system import QuantumTimeSystem

__all__ = [
    "NotebookData",
    "EvolutionConfig",
    "EvolutionStats",
    "track_nsga_metrics",
    "stats_to_ga_metrics",
    "load_data",
    "create_random_individual",
    "create_evaluator",
    "course_aware_crossover",
    "smart_mutation",
    "setup_deap",
    "get_constraint_breakdown",
    "run_nsga2",
    "get_best_individual",
    "print_constraint_details",
]


@dataclass
class NotebookData:
    """Container for all scheduling data needed by notebooks."""

    courses: dict[tuple[str, str], Course]
    groups: dict[str, Group]
    instructors: dict[str, Instructor]
    rooms: dict[str, Room]
    qts: QuantumTimeSystem
    context: SchedulingContext

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Courses: {len(self.courses)}, Groups: {len(self.groups)}, "
            f"Instructors: {len(self.instructors)}, Rooms: {len(self.rooms)}, "
            f"Quanta: {self.qts.total_quanta}"
        )


@dataclass
class EvolutionConfig:
    """Configuration for NSGA-II evolution."""

    pop_size: int = 50
    ngen: int = 100
    cxpb: float = 0.9
    mutpb: float = 0.2
    fitness_weights: tuple[float, float] = (-1.0, -0.01)
    verbose: bool = True
    log_interval: int = 20


@dataclass
class EvolutionStats:
    """Tracks evolution statistics across generations."""

    generations: list[int] = field(default_factory=list)
    min_hard: list[float] = field(default_factory=list)
    avg_hard: list[float] = field(default_factory=list)
    max_hard: list[float] = field(default_factory=list)
    min_soft: list[float] = field(default_factory=list)
    avg_soft: list[float] = field(default_factory=list)
    feasible_count: list[int] = field(default_factory=list)
    generation_times: list[float] = field(default_factory=list)
    elapsed_time: float = 0.0
    diversity: list[float] = field(default_factory=list)
    hypervolume: list[float] = field(default_factory=list)
    spacing: list[float] = field(default_factory=list)
    pareto_front_size: list[int] = field(default_factory=list)
    feasibility_rate: list[float] = field(default_factory=list)
    igd: list[float] = field(default_factory=list)
    spread: list[float] = field(default_factory=list)
    detailed_hard: dict[str, list[float]] = field(default_factory=dict)
    detailed_soft: dict[str, list[float]] = field(default_factory=dict)
    hypervolume_ref_point: tuple[float, float] | None = None
    reference_front: list[Any] = field(default_factory=list)


def _init_detailed_metrics(stats: EvolutionStats) -> None:
    """Initialize per-constraint metric storage if needed."""
    if stats.detailed_hard and stats.detailed_soft:
        return

    from schedule_engine.constraints import HARD_CONSTRAINT_NAMES, SOFT_CONSTRAINT_NAMES

    stats.detailed_hard = {name: [] for name in HARD_CONSTRAINT_NAMES}
    stats.detailed_soft = {name: [] for name in SOFT_CONSTRAINT_NAMES}


def track_nsga_metrics(
    population: list[Any],
    stats: EvolutionStats,
    data: NotebookData,
) -> None:
    """Track NSGA-II metrics for the current generation."""
    from deap import tools

    from schedule_engine.metrics import average_pairwise_diversity
    from schedule_engine.metrics.convergence import (
        calculate_constraint_satisfaction_rate,
    )
    from schedule_engine.metrics.hypervolume import (
        calculate_hypervolume,
        get_hypervolume_reference_point,
    )
    from schedule_engine.metrics.pareto_metrics import (
        calculate_inverted_generational_distance,
        calculate_spacing,
        calculate_spread,
        get_pareto_front_size,
    )

    if not population:
        return

    _init_detailed_metrics(stats)

    stats.diversity.append(average_pairwise_diversity(population))

    if stats.hypervolume_ref_point is None:
        stats.hypervolume_ref_point = get_hypervolume_reference_point(
            population, margin=0.1
        )
    stats.hypervolume.append(
        calculate_hypervolume(population, stats.hypervolume_ref_point)
    )

    stats.spacing.append(calculate_spacing(population))
    stats.pareto_front_size.append(get_pareto_front_size(population))
    stats.feasibility_rate.append(calculate_constraint_satisfaction_rate(population))

    if not stats.reference_front:
        pareto_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )[0]
        import copy

        stats.reference_front = [copy.deepcopy(ind) for ind in pareto_front]

    if stats.reference_front:
        stats.igd.append(
            calculate_inverted_generational_distance(population, stats.reference_front)
        )
    else:
        stats.igd.append(0.0)

    stats.spread.append(calculate_spread(population))

    best = tools.selBest(population, 1)[0]
    breakdown = get_constraint_breakdown(list(best), data)

    for name in stats.detailed_hard:
        stats.detailed_hard[name].append(breakdown.get(name, 0))

    for name in stats.detailed_soft:
        stats.detailed_soft[name].append(breakdown.get(name, 0))


def stats_to_ga_metrics(stats: EvolutionStats) -> "GAMetrics":
    """Convert notebook stats into core GA metrics for report export."""
    from schedule_engine.ga.scheduler import GAMetrics

    return GAMetrics(
        hard_violations=list(stats.min_hard),
        soft_penalties=list(stats.min_soft),
        diversity=list(stats.diversity),
        detailed_hard={k: list(v) for k, v in stats.detailed_hard.items()},
        detailed_soft={k: list(v) for k, v in stats.detailed_soft.items()},
        hypervolume=list(stats.hypervolume),
        spacing=list(stats.spacing),
        pareto_front_size=list(stats.pareto_front_size),
        feasibility_rate=list(stats.feasibility_rate),
        generation_times=list(stats.generation_times),
        igd=list(stats.igd),
        spread=list(stats.spread),
    )


def print_constraint_details(
    hard_breakdown: dict[str, float],
    soft_breakdown: dict[str, float],
    gen: int | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Print detailed constraint penalties with fixed-width alignment."""
    prefix = f"Gen {gen:3d}" if gen is not None else "Current"

    # Calculate totals
    hard_total = sum(hard_breakdown.values())
    soft_total = sum(soft_breakdown.values())

    # Sort constraints alphabetically for consistent display
    hard_items = sorted(hard_breakdown.items())
    soft_items = sorted(soft_breakdown.items())

    # Format ALL constraints with fixed width (show zeros too)
    hard_parts = [f"{k[:13]:13s}={int(v):4d}" for k, v in hard_items]
    soft_parts = [f"{k[:13]:13s}={int(v):4d}" for k, v in soft_items]

    # Join with | separator
    hard_str = " | ".join(hard_parts) if hard_parts else "none"
    soft_str = " | ".join(soft_parts) if soft_parts else "none"

    lines = [
        f"  {prefix}:  Hard={int(hard_total):4d}  Soft={int(soft_total):4d}",
        f"         HARD: [{hard_str}]",
        f"         SOFT: [{soft_str}]",
    ]
    if logger is None:
        for line in lines:
            print(line)
    else:
        for line in lines:
            logger.info(line)


def load_data(
    data_dir: Path | str = "data",
    opening_time: str = "10:00",
    closing_time: str = "17:00",
    closed_days: list[str] | None = None,
) -> NotebookData:
    """
    Load all scheduling data from JSON files.

    Delegates to :class:`DataStore.from_json` and wraps the result in a
    ``NotebookData`` for backward compatibility.
    """
    from schedule_engine.io.data_store import DataStore

    store = DataStore.from_json(
        data_dir,
        opening_time=opening_time,
        closing_time=closing_time,
        closed_days=closed_days,
    )

    # Ensure a global Config exists so heuristics and other subsystems
    # that read get_config_or_default() don't crash.
    from schedule_engine.config import Config, get_config_or_default, init_config

    cfg = get_config_or_default()
    if not cfg:
        cfg = Config(
            name="default",
            environment="test",
            heuristics=dict(master_enabled=True),
            repair=dict(
                enabled=True,
                detection_strategy="hybrid",
                heuristics={},
            ),
        )
        init_config(cfg)

    context = store.to_context()

    return NotebookData(
        courses=store.courses,
        groups=store.groups,
        instructors=store.instructors,
        rooms=store.rooms,
        qts=store.qts,
        context=context,
    )


def create_random_individual(
    data: NotebookData, conflict_aware: bool = True
) -> list[SessionGene]:
    """
    Create a random individual (chromosome) for the GA.

    Args:
        data: NotebookData containing all entities
        conflict_aware: If True (default), avoid group time conflicts during
            initialization. This dramatically reduces initial violations
            (~100-200 vs ~600-700 with pure random).

    Preserves:
        - Course-group pairs (no pedagogical violations)
        - Total quanta per course (split into subsessions, e.g. 5 -> 2,2,1 for theory)

    Random (can violate):
        - Instructor assignment (any instructor, can be unqualified)
        - Room assignment (any room, can be wrong type/size)
        - Time assignment: conflict-aware avoids group overlaps but not room/instructor

    Returns:
        List of SessionGene objects forming the chromosome
    """
    from collections import defaultdict

    from schedule_engine.ga.group_hierarchy import get_family_map_from_json
    from schedule_engine.ga.population import (
        analyze_group_hierarchy,
        generate_course_group_pairs,
        get_subsession_durations,
    )

    # Get course-group pairs (preserves pedagogical structure)
    hierarchy = analyze_group_hierarchy(data.context.groups)
    pair_tuples = generate_course_group_pairs(
        data.context.courses,
        data.context.groups,
        hierarchy,
        silent=True,
    )

    # Get all available resources (for random selection)
    all_instructors = list(data.instructors.values())
    all_rooms = list(data.rooms.values())
    total_quanta = data.qts.total_quanta
    all_quanta = list(range(total_quanta))

    # For conflict-aware: track group occupancy
    # Maps group_id -> set of occupied quanta
    group_occupied: dict[str, set[int]] = defaultdict(set)

    # Load family map for hierarchy-aware conflict detection
    family_map: dict[str, set[str]] = {}
    if conflict_aware:
        try:
            family_map = get_family_map_from_json("data/Groups.json")
        except Exception:
            pass  # Fall back to non-hierarchy-aware

    # Shuffle pairs to avoid systematic bias (first pairs always get best slots)
    shuffled_pairs = list(pair_tuples)
    random.shuffle(shuffled_pairs)

    genes = []
    for course_id, group_ids, _session_type, _num_quanta in shuffled_pairs:
        # Get course info for session type
        course = data.courses.get(course_id)
        course_type = course.course_type if course else "theory"
        course_quanta = course.quanta_per_week if course else int(_num_quanta)

        # Break into subsessions (e.g., 5 -> [2,2,1] for theory)
        subsession_durations = get_subsession_durations(course_quanta, course_type)

        for num_quanta in subsession_durations:
            # Random instructor and room (can create violations, easy to fix)
            instructor = random.choice(all_instructors)
            room = random.choice(all_rooms)

            # Find time slot
            if conflict_aware:
                # Find slot where NONE of the gene's groups (or their families) are occupied
                start_quanta = _find_group_free_slot(
                    group_ids, num_quanta, total_quanta, group_occupied, family_map
                )
            else:
                # Pure random
                max_start = total_quanta - num_quanta
                start_quanta = random.randint(0, max(0, max_start))

            gene = SessionGene(
                course_id=course_id[0] if isinstance(course_id, tuple) else course_id,
                course_type=course_type,
                group_ids=group_ids,
                instructor_id=instructor.instructor_id,
                room_id=room.room_id,
                start_quanta=start_quanta,
                num_quanta=num_quanta,
            )
            genes.append(gene)

            # Mark quanta as occupied for ALL groups in this gene (and their families)
            if conflict_aware:
                for gid in group_ids:
                    # Mark this group and all related groups as occupied
                    related = family_map.get(gid, {gid})
                    for related_gid in related:
                        for q in range(start_quanta, start_quanta + num_quanta):
                            group_occupied[related_gid].add(q)

    return genes


def _find_group_free_slot(
    group_ids: list[str],
    duration: int,
    total_quanta: int,
    group_occupied: dict[str, set[int]],
    family_map: dict[str, set[str]],
) -> int:
    """
    Find a time slot where none of the groups (or their families) are occupied.

    Tries random slots first (faster), then systematic search if needed.
    Falls back to least-congested slot if no free slot exists.

    Returns:
        Start quantum for the slot
    """
    max_start = total_quanta - duration
    if max_start <= 0:
        return 0

    # Build set of all related groups to check
    all_related: set[str] = set()
    for gid in group_ids:
        related = family_map.get(gid, {gid})
        all_related.update(related)

    def is_slot_free(start: int) -> bool:
        """Check if all quanta in [start, start+duration) are free for all groups."""
        for q in range(start, start + duration):
            for gid in all_related:
                if q in group_occupied.get(gid, set()):
                    return False
        return True

    # Try random slots first (faster for sparse schedules)
    random_attempts = min(20, max_start + 1)
    for _ in range(random_attempts):
        candidate = random.randint(0, max_start)
        if is_slot_free(candidate):
            return candidate

    # Systematic search: try all slots
    available_starts = [s for s in range(max_start + 1) if is_slot_free(s)]
    if available_starts:
        return random.choice(available_starts)

    # Fallback: find least congested slot
    best_start = 0
    best_conflicts = float("inf")
    for start in range(max_start + 1):
        conflicts = 0
        for q in range(start, start + duration):
            for gid in all_related:
                if q in group_occupied.get(gid, set()):
                    conflicts += 1
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_start = start

    return best_start


def create_evaluator(
    data: NotebookData,
) -> Callable[[list[SessionGene]], tuple[float, float]]:
    """
    Create a fitness evaluation function.

    All hard and soft constraints are always enabled with their default weights.

    Returns:
        Function that takes an individual and returns (hard_violations, soft_penalty)
    """
    from schedule_engine.constraints import HARD_CONSTRAINTS, SOFT_CONSTRAINTS
    from schedule_engine.domain.timetable import Timetable

    # Build context once
    context = data.to_context()

    def evaluate(individual: list[SessionGene]) -> tuple[float, float]:
        """Evaluate fitness: (hard violations, soft penalty)."""
        tt = Timetable(genes=individual, context=context)

        # Calculate hard constraint penalty
        hard_penalty = 0.0
        for c in HARD_CONSTRAINTS:
            hard_penalty += c.weight * c.evaluate(tt)

        # Calculate soft constraint penalty
        soft_penalty = 0.0
        for c in SOFT_CONSTRAINTS:
            soft_penalty += c.weight * c.evaluate(tt)

        return float(hard_penalty), float(soft_penalty)

    return evaluate


def course_aware_crossover(
    ind1: list[SessionGene],
    ind2: list[SessionGene],
    cx_prob: float = 0.5,
) -> tuple[list[SessionGene], list[SessionGene]]:
    """Course-group aware crossover. Delegates to operators/crossover.py."""
    from schedule_engine.ga.operators.crossover import crossover_course_group_aware

    return crossover_course_group_aware(ind1, ind2, cx_prob=cx_prob)


def smart_mutation(
    individual: list[SessionGene],
    data: NotebookData,
    gene_mut_prob: float = 0.2,
) -> list[SessionGene]:
    """Mutation operator. Delegates to operators/mutation.py."""
    from schedule_engine.ga.operators.mutation import mutate_individual

    (mutated,) = mutate_individual(
        individual,
        data.context,
        mut_prob=gene_mut_prob,
        guided=False,
    )
    return list(mutated)


# Track if DEAP types have been created (module-level flag)
_DEAP_TYPES_CREATED = False


def setup_deap(fitness_weights: tuple[float, ...] = (-1.0, -0.01)) -> None:
    """
    Set up DEAP creator types for multi-objective optimization.

    Creates FitnessMulti and Individual types if not already created.

    Args:
        fitness_weights: Weights for fitness objectives (negative = minimize)
    """
    global _DEAP_TYPES_CREATED
    # Create or update fitness class (multi-objective: minimize both)
    if hasattr(creator, "FitnessMulti"):
        if creator.FitnessMulti.weights != fitness_weights:
            creator.FitnessMulti.weights = fitness_weights
    else:
        creator.create("FitnessMulti", base.Fitness, weights=fitness_weights)

    # Create Individual class as list with fitness attribute
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    _DEAP_TYPES_CREATED = True


def get_constraint_breakdown(
    individual: list[SessionGene],
    data: NotebookData,
) -> dict[str, int | float]:
    """
    Get detailed constraint violation breakdown.

    All hard and soft constraints are always enabled.

    Args:
        individual: Individual to analyze
        data: NotebookData for context

    Returns:
        Dict mapping constraint names to violation counts
    """
    from schedule_engine.constraints import HARD_CONSTRAINTS, SOFT_CONSTRAINTS
    from schedule_engine.domain.timetable import Timetable

    # Build timetable from individual
    context = data.to_context()
    tt = Timetable(genes=individual, context=context)

    breakdown: dict[str, int | float] = {}

    for c in HARD_CONSTRAINTS:
        breakdown[c.name] = c.evaluate(tt)

    for c in SOFT_CONSTRAINTS:
        breakdown[c.name] = c.evaluate(tt)

    return breakdown


def get_best_individual(population: list[Any]) -> list[SessionGene]:
    """
    Get the best individual from a population.

    Best = lowest hard violations, then lowest soft penalty.

    Args:
        population: List of DEAP individuals

    Returns:
        Best individual (as list of SessionGene)
    """
    # Sort by fitness (hard violations first, then soft)
    sorted_pop = sorted(
        population,
        key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1]),
    )
    return list(sorted_pop[0])


def run_nsga2(
    data: NotebookData,
    config: EvolutionConfig,
    create_individual_fn: Callable[[NotebookData], list[SessionGene]] | None = None,
    evaluate_fn: Callable[[list[SessionGene]], tuple[float, float]] | None = None,
    crossover_fn: (
        Callable[
            [list[SessionGene], list[SessionGene]],
            tuple[list[SessionGene], list[SessionGene]],
        ]
        | None
    ) = None,
    mutate_fn: Callable[[list[SessionGene]], list[SessionGene]] | None = None,
    seed: int | None = 42,
    logger: logging.Logger | None = None,
) -> tuple[list[Any], EvolutionStats]:
    """
    Run standard NSGA-II evolution.

    IMPORTANT: seed is reset at START of evolution to ensure reproducibility
    regardless of any pre-evolution test code that may have consumed random numbers.

    Args:
        data: NotebookData containing scheduling entities
        config: Evolution configuration
        create_individual_fn: Function to create random individual
        evaluate_fn: Fitness evaluation function
        crossover_fn: Crossover operator
        mutate_fn: Mutation operator
        seed: Random seed for reproducibility (reset at start of evolution)
        logger: Optional logger for constraint detail output

    Returns:
        Tuple of (final_population, evolution_stats)
    """
    import time

    # CRITICAL: Reset random seed at START of evolution for reproducibility
    # This ensures identical Gen 0 across all notebooks regardless of test code
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    setup_deap(config.fitness_weights)

    create_individual_fn = create_individual_fn or create_random_individual
    evaluate_fn = evaluate_fn or create_evaluator(data)
    crossover_fn = crossover_fn or course_aware_crossover
    mutate_fn = mutate_fn or (lambda ind: smart_mutation(ind, data))

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register(
        "individual", lambda: creator.Individual(create_individual_fn(data))
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fn)
    toolbox.register("mate", crossover_fn)
    toolbox.register("mutate", mutate_fn)
    toolbox.register("select", tools.selNSGA2)

    # Initialize population
    pop = toolbox.population(n=config.pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    stats = EvolutionStats()
    start_time = time.time()

    # Evolution loop
    for gen in range(config.ngen):
        gen_start = time.time()
        # Selection + variation
        offspring = [copy.deepcopy(ind) for ind in toolbox.select(pop, len(pop))]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < config.cxpb:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < config.mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Evaluate
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Survivor selection
        pop = toolbox.select(pop + offspring, config.pop_size)

        # Record stats
        hard_vals = [ind.fitness.values[0] for ind in pop]
        soft_vals = [ind.fitness.values[1] for ind in pop]
        stats.generations.append(gen)
        stats.min_hard.append(float(min(hard_vals)))
        stats.avg_hard.append(float(np.mean(hard_vals)))
        stats.max_hard.append(float(max(hard_vals)))
        stats.min_soft.append(float(min(soft_vals)))
        stats.avg_soft.append(float(np.mean(soft_vals)))
        stats.feasible_count.append(sum(1 for h in hard_vals if h == 0))
        track_nsga_metrics(pop, stats, data)
        stats.generation_times.append(time.time() - gen_start)

        if config.verbose and (
            gen % config.log_interval == 0 or gen == config.ngen - 1
        ):
            # Get detailed breakdown for best individual
            best_ind = min(
                pop, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
            )
            breakdown = get_constraint_breakdown(list(best_ind), data)

            # Split into hard and soft
            from schedule_engine.constraints import (
                HARD_CONSTRAINT_NAMES,
                SOFT_CONSTRAINT_NAMES,
            )

            hard_bd = {k: v for k, v in breakdown.items() if k in HARD_CONSTRAINT_NAMES}
            soft_bd = {k: v for k, v in breakdown.items() if k in SOFT_CONSTRAINT_NAMES}

            print_constraint_details(hard_bd, soft_bd, gen, logger=logger)

    stats.elapsed_time = time.time() - start_time

    if config.verbose:
        print(f" Evolution complete in {stats.elapsed_time:.1f}s")

    return pop, stats
