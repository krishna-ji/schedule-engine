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
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from deap import base, creator, tools

from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.domain.types import Individual, SchedulingContext
from src.io.data_loader import (
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from src.io.decoder import decode_individual
from src.io.time_system import QuantumTimeSystem

__all__ = [
    "NotebookData",
    "EvolutionConfig",
    "EvolutionStats",
    "load_data",
    "create_random_individual",
    "create_evaluator",
    "course_aware_crossover",
    "smart_mutation",
    "setup_deap",
    "get_constraint_breakdown",
    "run_nsga2",
    "get_best_individual",
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
    elapsed_time: float = 0.0


def load_data(
    data_dir: Path | str = "data",
    opening_time: str = "10:00",
    closing_time: str = "17:00",
    closed_days: list[str] | None = None,
) -> NotebookData:
    """
    Load all scheduling data from JSON files.

    Args:
        data_dir: Path to data directory containing JSON files
        opening_time: Day start time (HH:MM format)
        closing_time: Day end time (HH:MM format)
        closed_days: List of days when no classes occur

    Returns:
        NotebookData container with all loaded entities
    """
    data_dir = Path(data_dir)
    closed_days = closed_days or ["Saturday"]

    # Create operating hours (same for all open days)
    operating_hours: dict[str, tuple[str, str] | None] = {}
    all_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    for day in all_days:
        if day in closed_days:
            operating_hours[day] = None
        else:
            operating_hours[day] = (opening_time, closing_time)

    # Initialize time system (quantum_minutes is a class constant)
    qts = QuantumTimeSystem(
        operating_hours=operating_hours,
    )

    # Load entities
    instructors = load_instructors(str(data_dir / "Instructors.json"), qts)
    courses = load_courses(str(data_dir / "Course.json"))  # No qts parameter
    groups = load_groups(str(data_dir / "Groups.json"), qts)
    rooms = load_rooms(str(data_dir / "Rooms.json"), qts)

    # Link relationships
    link_courses_and_groups(courses, groups)
    link_courses_and_instructors(courses, instructors)

    # Create scheduling context
    available_quanta = list(range(qts.total_quanta))
    context = SchedulingContext(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        available_quanta=available_quanta,
    )

    return NotebookData(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        qts=qts,
        context=context,
    )


def create_random_individual(data: NotebookData) -> list[SessionGene]:
    """
    Create a random individual (chromosome) for the GA.

    Each gene represents one course-group session assignment.
    Random assignment of instructor, room, and time slot.

    Args:
        data: NotebookData containing all entities

    Returns:
        List of SessionGene objects forming the chromosome
    """
    from src.ga.population import generate_course_group_aware_population

    population = generate_course_group_aware_population(
        n=1,
        context=data.context,
        parallel=False,
    )
    return list(population[0])


def create_evaluator(
    data: NotebookData,
) -> Callable[[list[SessionGene]], tuple[float, float]]:
    """
    Create a fitness evaluation function.

    Returns:
        Function that takes an individual and returns (hard_violations, soft_penalty)
    """

    from src.ga.evaluator.fitness import evaluate as evaluate_fitness

    def evaluate(individual: list[SessionGene]) -> tuple[float, float]:
        """Evaluate fitness: (hard violations, soft penalty)."""
        hard, soft = evaluate_fitness(
            individual,
            data.courses,
            data.instructors,
            data.groups,
            data.rooms,
        )
        return float(hard), float(soft)

    return evaluate


def course_aware_crossover(
    ind1: list[SessionGene],
    ind2: list[SessionGene],
    cx_prob: float = 0.5,
) -> tuple[list[SessionGene], list[SessionGene]]:
    """
    Course-group aware crossover operator.

    Swaps mutable attributes (instructor, room, time) between matching
    course-group pairs while preserving structure.

    Args:
        ind1: First parent individual
        ind2: Second parent individual
        cx_prob: Probability of swapping each gene

    Returns:
        Tuple of (child1, child2)
    """
    from src.ga.operators.crossover import crossover_course_group_aware

    return crossover_course_group_aware(ind1, ind2, cx_prob=cx_prob)


def smart_mutation(
    individual: list[SessionGene],
    data: NotebookData,
    gene_mut_prob: float = 0.2,
) -> list[SessionGene]:
    """
    Smart mutation operator with constraint awareness.

    Mutates time, room, or instructor with some intelligence:
    - Prefers qualified instructors
    - Respects time bounds

    Args:
        individual: Individual to mutate (modified in-place)
        data: NotebookData for context
        gene_mut_prob: Probability of mutating each gene

    Returns:
        The mutated individual
    """
    from src.ga.operators.mutation import mutate_individual

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

    if _DEAP_TYPES_CREATED:
        return

    # Create fitness class (multi-objective: minimize both)
    if not hasattr(creator, "FitnessMulti"):
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

    Args:
        individual: Individual to analyze
        data: NotebookData for context

    Returns:
        Dict mapping constraint names to violation counts
    """
    sessions = decode_individual(
        individual,
        data.courses,
        data.instructors,
        data.groups,
        data.rooms,
    )

    from src.constraints.registry import (
        constraint_needs_courses,
        get_enabled_hard_constraints,
        get_enabled_soft_constraints,
    )

    breakdown: dict[str, int | float] = {}

    for name, info in get_enabled_hard_constraints().items():
        func = info["function"]
        if constraint_needs_courses(name):
            breakdown[name] = func(sessions, data.courses)
        else:
            breakdown[name] = func(sessions)

    for name, info in get_enabled_soft_constraints().items():
        func = info["function"]
        if constraint_needs_courses(name):
            breakdown[name] = func(sessions, data.courses)
        else:
            breakdown[name] = func(sessions)

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
    crossover_fn: Callable[
        [list[SessionGene], list[SessionGene]],
        tuple[list[SessionGene], list[SessionGene]],
    ]
    | None = None,
    mutate_fn: Callable[[list[SessionGene]], list[SessionGene]] | None = None,
) -> tuple[list[Any], EvolutionStats]:
    """
    Run standard NSGA-II evolution.

    Args:
        data: NotebookData containing scheduling entities
        config: Evolution configuration
        create_individual_fn: Function to create random individual
        evaluate_fn: Fitness evaluation function
        crossover_fn: Crossover operator
        mutate_fn: Mutation operator

    Returns:
        Tuple of (final_population, evolution_stats)
    """
    import time

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

        if config.verbose and (
            gen % config.log_interval == 0 or gen == config.ngen - 1
        ):
            print(
                f"  Gen {gen:3d}: min_hard={stats.min_hard[-1]:3.0f}, "
                f"min_soft={stats.min_soft[-1]:5.0f}, "
                f"feasible={stats.feasible_count[-1]}/{config.pop_size}"
            )

    stats.elapsed_time = time.time() - start_time

    if config.verbose:
        print(f" Evolution complete in {stats.elapsed_time:.1f}s")

    return pop, stats
