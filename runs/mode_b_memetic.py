#!/usr/bin/env python3
"""
Mode B: Memetic NSGA-II

NSGA-II + Local Search - Applies local search to improve individuals after genetic operators.

Usage:
    python runs/mode_b_memetic.py
"""
from __future__ import annotations

import copy
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from deap import base, creator, tools

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.io.decoder import decode_individual
from schedule_engine.notebooks.core import (
    EvolutionStats,
    course_aware_crossover,
    create_evaluator,
    create_random_individual,
    get_best_individual,
    get_constraint_breakdown,
    load_data,
    print_constraint_details,
    setup_deap,
    smart_mutation,
    stats_to_ga_metrics,
    track_nsga_metrics,
)
from schedule_engine.notebooks.strategies import local_search_individual
from schedule_engine.notebooks.viz import print_summary
from schedule_engine.utils.json_utils import to_jsonable
from schedule_engine.workflows.reporting import generate_reports


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_b_memetic.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("mode_b_memetic")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode B: Memetic NSGA-II experiment."""

    # CONFIGURATION

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters
    POP_SIZE = 100
    NGEN = 500
    CXPB = 0.8
    MUTPB = 0.4
    FITNESS_WEIGHTS = (-1.0, -1.0)  # Align with other Mode B runs for fair comparison

    # MODE B: Local search parameters
    LOCAL_SEARCH_PROB = 0.1
    LOCAL_SEARCH_ITERATIONS = 5
    LOG_INTERVAL = 10

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "mode_b_memetic" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("MODE B: MEMETIC NSGA-II")
    logger.info("=" * 60)
    logger.info(f"Config: pop={POP_SIZE}, ngen={NGEN}, LS_prob={LOCAL_SEARCH_PROB}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # LOAD DATA

    logger.info("Loading data...")
    data = load_data(
        data_dir=DATA_DIR,
        opening_time="10:00",
        closing_time="17:00",
        closed_days=["Saturday"],
    )
    logger.info(f"Data loaded: {data.summary()}")

    evaluate = create_evaluator(data)

    # TEST COMPONENTS

    logger.info("Testing components...")
    test_ind = create_random_individual(data)
    logger.info(f"Individual: {len(test_ind)} genes")
    logger.info(f"Initial fitness: hard={evaluate(test_ind)[0]}")

    improved_ind, improvement = local_search_individual(
        test_ind, data, evaluate, max_iterations=5
    )
    logger.info(
        f"After LS: hard={evaluate(improved_ind)[0]} (improvement={improvement})"
    )

    # RUN MEMETIC NSGA-II

    logger.info("Starting Memetic NSGA-II evolution...")

    # Reset seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    start = time.time()
    setup_deap(FITNESS_WEIGHTS)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", lambda: creator.Individual(create_random_individual(data))
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", course_aware_crossover)
    toolbox.register("mutate", lambda ind: smart_mutation(ind, data))
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    stats = EvolutionStats()

    for gen in range(NGEN):
        offspring = [copy.deepcopy(ind) for ind in toolbox.select(pop, len(pop))]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < CXPB:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        # MODE B: Local Search
        for ind in offspring:
            if random.random() < LOCAL_SEARCH_PROB:
                genes = list(ind)
                improved_genes, _ = local_search_individual(
                    genes, data, evaluate, LOCAL_SEARCH_ITERATIONS
                )
                ind[:] = improved_genes
                del ind.fitness.values

        # Evaluate
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, POP_SIZE)

        # Stats
        hard_vals = [ind.fitness.values[0] for ind in pop]
        soft_vals = [ind.fitness.values[1] for ind in pop]
        stats.generations.append(gen)
        stats.min_hard.append(float(min(hard_vals)))
        stats.avg_hard.append(float(np.mean(hard_vals)))
        stats.max_hard.append(float(max(hard_vals)))
        stats.feasible_count.append(sum(1 for h in hard_vals if h == 0))
        stats.min_soft.append(float(min(soft_vals)))
        stats.avg_soft.append(float(np.mean(soft_vals)))
        track_nsga_metrics(pop, stats, data)

        if gen % LOG_INTERVAL == 0 or gen == NGEN - 1:
            best_ind = min(
                pop, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
            )
            breakdown = get_constraint_breakdown(list(best_ind), data)
            hard_names = {
                "student_group_exclusivity",
                "instructor_exclusivity",
                "instructor_qualifications",
                "room_suitability",
                "room_exclusivity",
                "instructor_time_availability",
                "room_time_availability",
                "course_completeness",
            }
            hard_bd = {k: v for k, v in breakdown.items() if k in hard_names}
            soft_bd = {k: v for k, v in breakdown.items() if k not in hard_names}
            print_constraint_details(hard_bd, soft_bd, gen)
            logger.debug(
                f"Gen {gen}: min_hard={min(hard_vals)}, feasible={stats.feasible_count[-1]}"
            )

    stats.elapsed_time = time.time() - start
    logger.info(f"Evolution completed in {stats.elapsed_time:.1f}s")

    final_pop = pop

    # RESULTS & VISUALIZATION

    logger.info("Generating results and visualizations...")

    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)
    print_summary(final_pop, stats, breakdown)

    # EXPORT RESULTS

    logger.info("Exporting full results...")
    best_schedule = decode_individual(
        best, data.courses, data.instructors, data.groups, data.rooms
    )
    ga_metrics = stats_to_ga_metrics(stats)
    generate_reports(
        decoded_schedule=best_schedule,
        metrics=ga_metrics,
        population=final_pop,
        qts=data.qts,
        output_dir=str(OUTPUT_DIR),
        course_map=data.courses,
    )

    metadata = {
        "experiment": "mode_b_memetic",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "ngen": NGEN,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "fitness_weights": list(FITNESS_WEIGHTS),
            "local_search_prob": LOCAL_SEARCH_PROB,
            "local_search_iterations": LOCAL_SEARCH_ITERATIONS,
        },
        "results": {
            "elapsed_time": stats.elapsed_time,
            "final_min_hard": stats.min_hard[-1] if stats.min_hard else None,
            "final_min_soft": stats.min_soft[-1] if stats.min_soft else None,
            "final_feasible_count": (
                stats.feasible_count[-1] if stats.feasible_count else 0
            ),
        },
        "constraint_breakdown": breakdown,
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE B COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
