#!/usr/bin/env python3
"""
Mode B1: Memetic + Fast Parallel Repair Operators

Enhancement over Mode B: Replaces blind local search with 4 optimized constraint-aware repair heuristics.

Performance Optimizations:
1. Cached Occupation Maps: Build map ONCE per iteration, not per gene (O(n) → O(1))
2. Parallel Processing: Repair multiple individuals simultaneously
3. Fast Conflict Detection: Simplified checks without repeated list comprehensions
4. Early Termination: Skip genes that don't need repair

Usage:
    python runs/mode_b1_repair_operators.py
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

from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.types import SchedulingContext
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
from schedule_engine.notebooks.parallel_repair import (
    RepairStats,
    apply_fast_repair,
    build_occupied_map,
)
from schedule_engine.notebooks.viz import print_summary
from schedule_engine.utils.json_utils import to_jsonable
from schedule_engine.workflows.reporting import generate_reports


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_b1_repair_operators.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("mode_b1_repair_operators")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode B1: Memetic + Fast Parallel Repair Operators."""
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters
    POP_SIZE = 50
    NGEN = 200
    CXPB = 0.9
    MUTPB = 0.2
    FITNESS_WEIGHTS = (-1.0, -1.0)

    # MODE B1: Repair operator parameters
    REPAIR_PROB = 0.3
    REPAIR_ITERATIONS = 2
    LOG_INTERVAL = 20

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "mode_b1_repair_operators" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("MODE B1: MEMETIC + FAST PARALLEL REPAIR OPERATORS")
    logger.info("=" * 60)
    logger.info(f"Config: pop={POP_SIZE}, ngen={NGEN}, repair_prob={REPAIR_PROB}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    logger.info("Loading data...")
    data = load_data(
        data_dir=DATA_DIR,
        opening_time="10:00",
        closing_time="17:00",
        closed_days=["Saturday"],
    )
    logger.info(f"Data loaded: {data.summary()}")

    context = data.context
    evaluate = create_evaluator(data)

    # ==========================================================================
    # RUN MEMETIC FAST REPAIR NSGA-II
    # ==========================================================================
    logger.info("Starting Memetic Fast Repair NSGA-II evolution...")

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
    total_repairs = 0
    total_repair_time = 0.0

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

        # MODE B1: FAST Constraint-Aware Repair
        repair_start = time.time()
        for ind in offspring:
            if random.random() < REPAIR_PROB:
                genes_list = list(ind)
                repair_stats = apply_fast_repair(genes_list, context, REPAIR_ITERATIONS)
                total_repairs += repair_stats.total_fixes

                if repair_stats.total_fixes > 0:
                    ind.clear()
                    ind.extend(genes_list)
                    del ind.fitness.values
        total_repair_time += time.time() - repair_start

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
                f"Gen {gen}: min_hard={min(hard_vals)}, repairs={total_repairs}"
            )

    stats.elapsed_time = time.time() - start
    logger.info(f"Evolution completed in {stats.elapsed_time:.1f}s")
    logger.info(f"Total repairs: {total_repairs}")
    logger.info(
        f"Repair time: {total_repair_time:.1f}s ({100*total_repair_time/stats.elapsed_time:.1f}% of total)"
    )

    final_pop = pop

    # ==========================================================================
    # RESULTS & VISUALIZATION
    # ==========================================================================
    logger.info("Generating results and visualizations...")

    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)
    print_summary(final_pop, stats, breakdown)

    # ==========================================================================
    # EXPORT RESULTS
    # ==========================================================================
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
        "experiment": "mode_b1_repair_operators",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "ngen": NGEN,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "fitness_weights": list(FITNESS_WEIGHTS),
            "repair_prob": REPAIR_PROB,
            "repair_iterations": REPAIR_ITERATIONS,
        },
        "results": {
            "elapsed_time": stats.elapsed_time,
            "total_repairs": total_repairs,
            "repair_time": total_repair_time,
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
    logger.info("MODE B1 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
