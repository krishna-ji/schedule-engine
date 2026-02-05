#!/usr/bin/env python3
"""
Mode B2: Memetic + Adaptive Targeting

Enhancement over Mode B1: Tracks constraint violations over generations and
prioritizes repair operators for constraints that are NOT decreasing.

Constraint-to-Operator Mapping:
- student_group_exclusivity  → move_time
- instructor_exclusivity     → move_time
- room_exclusivity          → move_time
- room_time_availability    → move_time
- instructor_time_availability → move_time, reassign_instructor
- instructor_qualifications → reassign_instructor
- room_suitability          → swap_room

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
from collections import defaultdict
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
from schedule_engine.workflows.feasibility_checks import run_feasibility_checks
from schedule_engine.workflows.reporting import generate_reports

# Constraint-to-Operator mapping: which operator fixes which constraint
CONSTRAINT_TO_OPERATOR: dict[str, list[str]] = {
    # Time-based conflicts → move_time
    "student_group_exclusivity": ["move_time"],
    "instructor_exclusivity": ["move_time"],
    "room_exclusivity": ["move_time"],
    "room_time_availability": ["move_time"],
    # Instructor-related → reassign_instructor (or move_time for availability)
    "instructor_time_availability": ["move_time", "reassign_instructor"],
    "instructor_qualifications": ["reassign_instructor"],
    # Room-related → swap_room
    "room_suitability": ["swap_room"],
    # course_completeness is structural, not repairable by these operators
}


def get_stagnant_constraints(
    violation_history: dict[str, list[int]],
    lookback: int = 5,
) -> list[str]:
    """
    Identify constraints whose violations are NOT decreasing.

    A constraint is stagnant if:
    - It has non-zero violations in the latest generation
    - Its violation count has not decreased over the last `lookback` generations

    Returns:
        List of stagnant constraint names, sorted by current violation count (desc)
    """
    stagnant = []

    for constraint_name, history in violation_history.items():
        if len(history) < 2:
            continue

        current = history[-1]
        if current == 0:
            continue  # No violations, not stagnant

        # Check if decreasing over lookback window
        window = history[-lookback:] if len(history) >= lookback else history
        if len(window) < 2:
            continue

        # Stagnant if first value in window <= last value (not decreasing)
        if window[0] <= window[-1]:
            stagnant.append((constraint_name, current))

    # Sort by violation count descending
    stagnant.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in stagnant]


def get_priority_operators(stagnant_constraints: list[str]) -> list[str]:
    """
    Get prioritized operators based on stagnant constraints.

    Returns:
        List of operator names to prioritize (in order of priority)
    """
    operator_scores: dict[str, int] = defaultdict(int)

    for idx, constraint_name in enumerate(stagnant_constraints):
        operators = CONSTRAINT_TO_OPERATOR.get(constraint_name, [])
        # Higher score for higher-priority (earlier) constraints
        score = len(stagnant_constraints) - idx
        for op in operators:
            operator_scores[op] += score

    # Sort by score descending
    sorted_ops = sorted(operator_scores.items(), key=lambda x: x[1], reverse=True)
    return [op for op, _ in sorted_ops]


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

    logger.handlers.clear()

    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode B2: Memetic + Adaptive Targeting."""

    # CONFIGURATION

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters
    POP_SIZE = 10
    NGEN = 2000
    CXPB = 0.9
    MUTPB = 0.2
    FITNESS_WEIGHTS = (-1.0, -1.0)

    # MODE B2: Adaptive targeting parameters
    REPAIR_FRACTION = 0.3  # Repair worst 30%
    REPAIR_BUDGET_MS = 50.0
    REPAIR_MAX_STEPS = 5
    REPAIR_MAX_CANDIDATES = 20
    STAGNATION_LOOKBACK = 5  # Generations to check for stagnation
    LOG_INTERVAL = 10
    EXPECTED_QUANTA = 42

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
        f"Config: pop={POP_SIZE}, ngen={NGEN}, repair_fraction={REPAIR_FRACTION}, "
        f"budget_ms={REPAIR_BUDGET_MS}, stagnation_lookback={STAGNATION_LOOKBACK}"
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
    run_feasibility_checks(data, OUTPUT_DIR, logger, expected_quanta=EXPECTED_QUANTA)

    context = data.context
    evaluate = create_evaluator(data)
    repair_engine = RepairEngine(
        context=context,
        evaluator=evaluate,
        policy="round_robin",  # Fallback policy when no stagnant constraints
        max_steps=REPAIR_MAX_STEPS,
        max_candidates=REPAIR_MAX_CANDIDATES,
        budget_ms=REPAIR_BUDGET_MS,
        epsilon=0.1,
        rng=random.Random(SEED),
        logger=logger,
        log_steps=True,
        log_candidates=True,
    )

    # RUN ADAPTIVE TARGETING NSGA-II

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
    repair_history: list[dict[str, float | int | str | list[str]]] = []

    # MODE B2: Track violation history for adaptive targeting
    violation_history: dict[str, list[int]] = defaultdict(list)
    hard_constraint_names = {
        "student_group_exclusivity",
        "instructor_exclusivity",
        "instructor_qualifications",
        "room_suitability",
        "room_exclusivity",
        "instructor_time_availability",
        "room_time_availability",
        "course_completeness",
    }

    for gen in range(NGEN):
        gen_start = time.time()
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

        # MODE B2: Adaptive Targeting - repair worst fraction with targeted operators
        repair_start = time.time()
        indexed_offspring = [
            (i, ind.fitness.values[0], ind.fitness.values[1])
            for i, ind in enumerate(offspring)
        ]
        indexed_offspring.sort(key=lambda x: (x[1], x[2]), reverse=True)

        n_repair = max(1, int(REPAIR_FRACTION * len(offspring)))
        worst_indices = [idx for idx, _, _ in indexed_offspring[:n_repair]]

        # Determine which operators to prioritize based on stagnant constraints
        stagnant = get_stagnant_constraints(violation_history, STAGNATION_LOOKBACK)
        priority_operators = get_priority_operators(stagnant)
        # Use first priority operator if available, otherwise None (round-robin fallback)
        forced_operator = priority_operators[0] if priority_operators else None

        per_individual_budget = (
            REPAIR_BUDGET_MS / max(1, len(worst_indices)) if REPAIR_BUDGET_MS > 0 else 0
        )
        gen_repairs = 0
        gen_delta_hard = 0.0
        gen_delta_soft = 0.0
        for idx in worst_indices:
            ind = offspring[idx]
            repair_stats = repair_engine.repair_individual(
                ind, budget_ms=per_individual_budget, forced_operator=forced_operator
            )
            gen_repairs += repair_stats.applied_steps
            gen_delta_hard += repair_stats.total_delta_hard
            gen_delta_soft += repair_stats.total_delta_soft
            if repair_stats.applied_steps > 0:
                del ind.fitness.values
        total_repairs += gen_repairs
        repair_time_ms = (time.time() - repair_start) * 1000

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

        # MODE B2: Track per-constraint violations for adaptive targeting
        best_ind = min(
            pop, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
        )
        breakdown = get_constraint_breakdown(list(best_ind), data)
        for constraint_name in hard_constraint_names:
            violation_count = int(breakdown.get(constraint_name, 0))
            violation_history[constraint_name].append(violation_count)

        repair_history.append(
            {
                "generation": gen,
                "repairs_applied": gen_repairs,
                "delta_hard": gen_delta_hard,
                "delta_soft": gen_delta_soft,
                "repair_time_ms": repair_time_ms,
                "forced_operator": forced_operator,
                "stagnant_constraints": stagnant,
                "priority_operators": priority_operators,
            }
        )

        stats.generation_times.append(time.time() - gen_start)

        if gen % LOG_INTERVAL == 0 or gen == NGEN - 1:
            hard_bd = {k: v for k, v in breakdown.items() if k in hard_constraint_names}
            soft_bd = {
                k: v for k, v in breakdown.items() if k not in hard_constraint_names
            }
            print_constraint_details(hard_bd, soft_bd, gen, logger=logger)
            if stagnant:
                logger.info(
                    f"  Stagnant constraints: {stagnant[:3]} → forcing {forced_operator}"
                )
            logger.debug(
                f"Gen {gen}: min_hard={min(hard_vals)}, repairs={gen_repairs}, "
                f"delta_hard={gen_delta_hard:.2f}, delta_soft={gen_delta_soft:.2f}"
            )

    stats.elapsed_time = time.time() - start
    logger.info(
        f"Evolution completed in {stats.elapsed_time:.1f}s (total repairs: {total_repairs})"
    )

    final_pop = pop

    # RESULTS & VISUALIZATION

    logger.info("Generating results and visualizations...")

    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)
    print_summary(final_pop, stats, breakdown, logger=logger)

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
        repair_history=repair_history,
        generation_times=stats.generation_times,
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
            "repair_budget_ms": REPAIR_BUDGET_MS,
            "repair_max_steps": REPAIR_MAX_STEPS,
            "repair_max_candidates": REPAIR_MAX_CANDIDATES,
            "stagnation_lookback": STAGNATION_LOOKBACK,
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
        "violation_history": dict(violation_history),
        "repair_history": repair_history,
        "generation_times": stats.generation_times,
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE B2 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
