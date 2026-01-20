"""NSGA-II evolution loop for experiment notebooks.

Provides configurable evolution loop.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from deap import base, creator, tools

from src.ga.sessiongene import SessionGene

if TYPE_CHECKING:
    from src.notebooks.data_loader import ScheduleData


@dataclass
class EvolutionStats:
    """Statistics collected during evolution."""

    generations: list[int] = field(default_factory=list)
    min_hard: list[float] = field(default_factory=list)
    avg_hard: list[float] = field(default_factory=list)
    max_hard: list[float] = field(default_factory=list)
    feasible_count: list[int] = field(default_factory=list)
    min_soft: list[float] = field(default_factory=list)
    avg_soft: list[float] = field(default_factory=list)
    elapsed_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for plotting."""
        return {
            "gen": self.generations,
            "min_hard": self.min_hard,
            "avg_hard": self.avg_hard,
            "max_hard": self.max_hard,
            "feasible": self.feasible_count,
            "min_soft": self.min_soft,
            "avg_soft": self.avg_soft,
        }


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""

    pop_size: int = 50
    ngen: int = 100
    cxpb: float = 0.9
    mutpb: float = 0.2
    fitness_weights: tuple[float, float] = (-1.0, -0.01)
    verbose: bool = True
    log_interval: int = 20


def setup_deap(
    fitness_weights: tuple[float, float] = (-1.0, -0.01),
) -> None:
    """Setup DEAP creator classes.

    Args:
        fitness_weights: Weights for (hard, soft) objectives
    """
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=fitness_weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


def run_nsga2(
    data: ScheduleData,
    config: EvolutionConfig,
    create_individual_fn: Callable[[ScheduleData], list[SessionGene]],
    evaluate_fn: Callable[[list[SessionGene]], tuple[int, int]],
    crossover_fn: Callable[
        [list[SessionGene], list[SessionGene]],
        tuple[list[SessionGene], list[SessionGene]],
    ],
    mutate_fn: Callable[[list[SessionGene]], tuple[list[SessionGene]]],
    generation_callback: Callable[[int, list[Any], EvolutionStats], None] | None = None,
) -> tuple[list[Any], EvolutionStats]:
    """Run NSGA-II evolution.

    Args:
        data: Schedule data (passed to individual creation)
        config: Evolution configuration
        create_individual_fn: Function to create one individual
        evaluate_fn: Fitness evaluation function
        crossover_fn: Crossover operator
        mutate_fn: Mutation operator
        generation_callback: Optional callback after each generation

    Returns:
        Tuple of (final_population, statistics)

    Example:
        >>> from src.notebooks import load_data, create_random_individual
        >>> data = load_data("../data")
        >>> config = EvolutionConfig(pop_size=50, ngen=100)
        >>> pop, stats = run_nsga2(
        ...     data, config,
        ...     lambda d: create_random_individual(d),
        ...     create_evaluator(data),
        ...     course_aware_crossover,
        ...     lambda ind: smart_mutation(ind, data),
        ... )
    """
    # Setup DEAP
    setup_deap(config.fitness_weights)

    # Create toolbox
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
    if config.verbose:
        print(f" NSGA-II: pop={config.pop_size}, ngen={config.ngen}")

    start_time = time.time()
    pop = toolbox.population(n=config.pop_size)

    # Evaluate initial population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # Statistics tracking
    stats = EvolutionStats()

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

        # Evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Survivor selection (NSGA-II)
        pop = toolbox.select(pop + offspring, config.pop_size)

        # Collect statistics
        hard_vals = [ind.fitness.values[0] for ind in pop]
        soft_vals = [ind.fitness.values[1] for ind in pop]

        stats.generations.append(gen)
        stats.min_hard.append(float(min(hard_vals)))
        stats.avg_hard.append(float(np.mean(hard_vals)))
        stats.max_hard.append(float(max(hard_vals)))
        stats.feasible_count.append(sum(1 for h in hard_vals if h == 0))
        stats.min_soft.append(float(min(soft_vals)))
        stats.avg_soft.append(float(np.mean(soft_vals)))

        # Callback
        if generation_callback:
            generation_callback(gen, pop, stats)

        # Logging
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
        print(f" Done in {stats.elapsed_time:.1f}s")

    return pop, stats


def get_best_individual(
    population: list[Any],
) -> Any:
    """Get best individual from population (lexicographic on hard then soft).

    Args:
        population: Final population from evolution

    Returns:
        Best individual
    """
    return min(population, key=lambda x: (x.fitness.values[0], x.fitness.values[1]))


def get_pareto_front(
    population: list[Any],
) -> list[Any]:
    """Get non-dominated solutions (Pareto front).

    Args:
        population: Population to extract front from

    Returns:
        List of non-dominated individuals
    """
    return tools.sortNondominated(population, len(population), first_front_only=True)[0]
