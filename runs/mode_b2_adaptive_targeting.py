#!/usr/bin/env python3
"""
Mode B2: Memetic + Adaptive Targeting

Enhancement over Mode B1: Repairs worst 30% of population instead of random 20%.

Usage:
    python runs/mode_b2_adaptive_targeting.py
"""
from __future__ import annotations

import copy
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from deap import base, creator, tools

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.types import SchedulingContext
from schedule_engine.ga.operators.repair import (
    repair_group_overlaps,
    repair_instructor_availability,
    repair_instructor_conflicts,
    repair_instructor_qualifications,
    repair_room_conflicts,
    repair_room_overlap_reassign,
    repair_room_type_mismatches,
)
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
)
from schedule_engine.notebooks.export import export_full_results
from schedule_engine.notebooks.viz import (
    plot_constraint_breakdown,
    plot_convergence,
    print_summary,
)


@dataclass
class RepairStats:
    """Track repair operator statistics."""

    total_fixes: int = 0
    by_operator: dict[str, int] = field(default_factory=dict)


def apply_repair_operators(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 3,
) -> RepairStats:
    """Apply constraint-aware repair operators in priority order."""
    stats = RepairStats()

    repair_operators = [
        ("instructor_availability", repair_instructor_availability),
        ("group_overlaps", repair_group_overlaps),
        ("room_overlap_reassign", repair_room_overlap_reassign),
        ("room_conflicts", repair_room_conflicts),
        ("instructor_conflicts", repair_instructor_conflicts),
        ("instructor_qualifications", repair_instructor_qualifications),
        ("room_type_mismatches", repair_room_type_mismatches),
    ]

    for _ in range(max_iterations):
        iteration_fixes = 0
        for name, operator in repair_operators:
            try:
                fixes = operator(individual, context)
                if fixes > 0:
                    stats.by_operator[name] = stats.by_operator.get(name, 0) + fixes
                    stats.total_fixes += fixes
                    iteration_fixes += fixes
            except Exception:
                pass

        if iteration_fixes == 0:
            break

    return stats


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_b2_adaptive_targeting.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("mode_b2_adaptive_targeting")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode B2: Memetic + Adaptive Targeting."""
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters
    POP_SIZE = 10
    NGEN = 4000
    CXPB = 0.9
    MUTPB = 0.2
    FITNESS_WEIGHTS = (-1.0, -1.0)

    # MODE B2: Adaptive targeting parameters
    REPAIR_FRACTION = 0.3  # Repair worst 30%
    REPAIR_ITERATIONS = 5
    LOG_INTERVAL = 10

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "mode_b2_adaptive_targeting" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("MODE B2: MEMETIC + ADAPTIVE TARGETING")
    logger.info("=" * 60)
    logger.info(
        f"Config: pop={POP_SIZE}, ngen={NGEN}, repair_fraction={REPAIR_FRACTION}"
    )
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
    # RUN ADAPTIVE TARGETING NSGA-II
    # ==========================================================================
    logger.info("Starting Adaptive Targeting NSGA-II evolution...")

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

        # Evaluate all offspring first (needed for ranking)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # MODE B2: Adaptive Targeting - repair worst fraction
        indexed_offspring = [
            (i, ind.fitness.values[0]) for i, ind in enumerate(offspring)
        ]
        indexed_offspring.sort(key=lambda x: x[1], reverse=True)

        n_repair = max(1, int(REPAIR_FRACTION * len(offspring)))
        worst_indices = [idx for idx, _ in indexed_offspring[:n_repair]]

        for idx in worst_indices:
            ind = offspring[idx]
            repair_stats = apply_repair_operators(list(ind), context, REPAIR_ITERATIONS)
            total_repairs += repair_stats.total_fixes
            del ind.fitness.values

        # Re-evaluate repaired individuals
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
            logger.debug(f"Gen {gen}: min_hard={min(hard_vals)}")

    stats.elapsed_time = time.time() - start
    logger.info(
        f"Evolution completed in {stats.elapsed_time:.1f}s (total repairs: {total_repairs})"
    )

    final_pop = pop

    # ==========================================================================
    # RESULTS & VISUALIZATION
    # ==========================================================================
    logger.info("Generating results and visualizations...")

    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)
    print_summary(final_pop, stats, breakdown)

    # Export figures
    logger.info("Exporting figures...")
    plot_convergence(
        stats, OUTPUT_DIR / "mode_b2_convergence.png", title_prefix="Mode B2: "
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_b2_convergence.png'}")

    plot_constraint_breakdown(
        breakdown,
        OUTPUT_DIR / "mode_b2_breakdown.png",
        title="Mode B2: Constraint Violations",
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_b2_breakdown.png'}")

    # ==========================================================================
    # EXPORT RESULTS
    # ==========================================================================
    logger.info("Exporting full results...")
    export_paths = export_full_results(
        population=final_pop,
        stats=stats,
        data=data,
        output_dir=OUTPUT_DIR,
        mode_name="mode_b2_adaptive_targeting",
    )

    metadata = {
        "experiment": "mode_b2_adaptive_targeting",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "ngen": NGEN,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "fitness_weights": list(FITNESS_WEIGHTS),
            "repair_fraction": REPAIR_FRACTION,
            "repair_iterations": REPAIR_ITERATIONS,
        },
        "results": {
            "elapsed_time": stats.elapsed_time,
            "total_repairs": total_repairs,
            "final_min_hard": stats.min_hard[-1] if stats.min_hard else None,
            "final_min_soft": stats.min_soft[-1] if stats.min_soft else None,
            "final_feasible_count": (
                stats.feasible_count[-1] if stats.feasible_count else 0
            ),
        },
        "constraint_breakdown": breakdown,
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE B2 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
