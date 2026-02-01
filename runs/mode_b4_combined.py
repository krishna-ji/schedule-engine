#!/usr/bin/env python3
"""
Mode B4: Memetic + All Enhancements Combined

Combines: B1 (Repair Operators) + B2 (Adaptive Targeting) + B3 (Two-Phase)

| Phase | Generations | Target     | Iterations | Strategy               |
|-------|-------------|------------|------------|------------------------|
| 1     | 0-199       | Worst 20%  | 3          | Light exploration      |
| 2     | 200+        | Worst 40%  | 8          | Intensive exploitation |

Usage:
    python runs/mode_b4_combined.py
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

from schedule_engine.ga.operators.repair_engine import RepairEngine
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
from schedule_engine.notebooks.viz import print_summary
from schedule_engine.utils.json_utils import to_jsonable
from schedule_engine.workflows.reporting import generate_reports


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_b4_combined.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("mode_b4_combined")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode B4: Memetic + All Enhancements Combined."""

    # CONFIGURATION

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters
    POP_SIZE = 10
    NGEN = 4000
    CXPB = 0.9
    MUTPB = 0.2
    FITNESS_WEIGHTS = (-1.0, -1.0)

    # MODE B4: Combined parameters
    PHASE_SWITCH_GEN = 200
    PHASE1_REPAIR_FRACTION = 0.2
    PHASE1_REPAIR_ITERATIONS = 3
    PHASE2_REPAIR_FRACTION = 0.4
    PHASE2_REPAIR_ITERATIONS = 8
    REPAIR_POLICY = "round_robin"
    REPAIR_BUDGET_MS = 50.0
    REPAIR_MAX_STEPS = 5
    REPAIR_MAX_CANDIDATES = 20
    REPAIR_EPSILON = 0.1
    LOG_INTERVAL = 10

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "mode_b4_combined" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("MODE B4: MEMETIC + ALL ENHANCEMENTS COMBINED")
    logger.info("=" * 60)
    logger.info(
        f"Phase 1: worst {PHASE1_REPAIR_FRACTION*100:.0f}%, {PHASE1_REPAIR_ITERATIONS} iterations"
    )
    logger.info(
        f"Phase 2: worst {PHASE2_REPAIR_FRACTION*100:.0f}%, {PHASE2_REPAIR_ITERATIONS} iterations"
    )
    logger.info(
        f"Repair policy={REPAIR_POLICY}, budget_ms={REPAIR_BUDGET_MS}, "
        f"max_steps={REPAIR_MAX_STEPS}, max_candidates={REPAIR_MAX_CANDIDATES}"
    )
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

    context = data.context
    evaluate = create_evaluator(data)
    repair_engine = RepairEngine(
        context=context,
        evaluator=evaluate,
        policy=REPAIR_POLICY,
        max_steps=REPAIR_MAX_STEPS,
        max_candidates=REPAIR_MAX_CANDIDATES,
        budget_ms=REPAIR_BUDGET_MS,
        epsilon=REPAIR_EPSILON,
        rng=random.Random(SEED),
    )

    # RUN COMBINED NSGA-II

    logger.info("Starting Combined NSGA-II evolution...")

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
        # Phase-dependent parameters
        if gen < PHASE_SWITCH_GEN:
            repair_fraction = PHASE1_REPAIR_FRACTION
            repair_iterations = PHASE1_REPAIR_ITERATIONS
            current_phase = 1
        else:
            repair_fraction = PHASE2_REPAIR_FRACTION
            repair_iterations = PHASE2_REPAIR_ITERATIONS
            current_phase = 2

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

        # Evaluate all offspring first (for ranking)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # MODE B4: Adaptive targeting with phase-dependent intensity
        indexed_offspring = [
            (i, ind.fitness.values[0], ind.fitness.values[1])
            for i, ind in enumerate(offspring)
        ]
        indexed_offspring.sort(key=lambda x: (x[1], x[2]), reverse=True)

        n_repair = max(1, int(repair_fraction * len(offspring)))
        worst_indices = [idx for idx, _, _ in indexed_offspring[:n_repair]]

        per_individual_budget = (
            REPAIR_BUDGET_MS / max(1, len(worst_indices)) if REPAIR_BUDGET_MS > 0 else 0
        )
        for idx in worst_indices:
            ind = offspring[idx]
            repair_stats = repair_engine.repair_individual(
                ind, budget_ms=per_individual_budget, max_steps=repair_iterations
            )
            total_repairs += repair_stats.applied_steps
            if repair_stats.applied_steps > 0:
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
        track_nsga_metrics(pop, stats, data)

        if gen % LOG_INTERVAL == 0 or gen == NGEN - 1 or gen == PHASE_SWITCH_GEN:
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

    stats.elapsed_time = time.time() - start
    logger.info(
        f"Evolution completed in {stats.elapsed_time:.1f}s (total repairs: {total_repairs})"
    )

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
        "experiment": "mode_b4_combined",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "ngen": NGEN,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "fitness_weights": list(FITNESS_WEIGHTS),
            "phase_switch_gen": PHASE_SWITCH_GEN,
            "phase1_repair_fraction": PHASE1_REPAIR_FRACTION,
            "phase1_repair_iterations": PHASE1_REPAIR_ITERATIONS,
            "phase2_repair_fraction": PHASE2_REPAIR_FRACTION,
            "phase2_repair_iterations": PHASE2_REPAIR_ITERATIONS,
            "repair_policy": REPAIR_POLICY,
            "repair_budget_ms": REPAIR_BUDGET_MS,
            "repair_max_steps": REPAIR_MAX_STEPS,
            "repair_max_candidates": REPAIR_MAX_CANDIDATES,
            "repair_epsilon": REPAIR_EPSILON,
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
        json.dump(to_jsonable(metadata), f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE B4 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
