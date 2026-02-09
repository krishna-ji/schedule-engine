#!/usr/bin/env python3
"""
Mode D: Adaptive Heuristics

NSGA-II + Adaptive Selection - Learns which heuristics work best and adapts probabilities.

Usage:
    python runs/mode_d_adaptive.py
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

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.ga.operators.repair_engine import RepairEngine
from schedule_engine.ga.run_helpers import (
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
from schedule_engine.io.decoder import decode_individual
from schedule_engine.utils.json_utils import to_jsonable
from schedule_engine.viz import print_summary
from schedule_engine.workflows.feasibility_checks import run_feasibility_checks
from schedule_engine.workflows.reporting import generate_reports


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_d_adaptive.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("mode_d_adaptive")

    logger.handlers.clear()

    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode D: Adaptive Heuristics."""

    # CONFIGURATION

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters
    POP_SIZE = 50
    NGEN = 100
    CXPB = 0.9
    MUTPB = 0.2
    FITNESS_WEIGHTS = (-1.0, -1.0)

    # MODE D: Adaptive selection
    REPAIR_PROB = 0.45
    REPAIR_POLICY = "epsilon_greedy"
    REPAIR_EPSILON = 0.1
    REPAIR_BUDGET_MS = 120.0
    REPAIR_MAX_STEPS = 3
    REPAIR_MAX_CANDIDATES = 30
    LOG_INTERVAL = 10
    EXPECTED_QUANTA = 42

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "mode_d_adaptive" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("MODE D: ADAPTIVE HEURISTICS")
    logger.info("=" * 60)
    logger.info(
        f"Config: pop={POP_SIZE}, ngen={NGEN}, repair_policy={REPAIR_POLICY}, "
        f"eps={REPAIR_EPSILON}, budget_ms={REPAIR_BUDGET_MS}"
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

    evaluate = create_evaluator(data)
    repair_engine = RepairEngine(
        context=data.context,
        evaluator=evaluate,
        policy=REPAIR_POLICY,
        max_steps=REPAIR_MAX_STEPS,
        max_candidates=REPAIR_MAX_CANDIDATES,
        budget_ms=REPAIR_BUDGET_MS,
        epsilon=REPAIR_EPSILON,
        rng=random.Random(SEED),
        logger=logger,
        log_steps=True,
        log_candidates=True,
    )

    # TEST ADAPTIVE REPAIR ENGINE

    logger.info("Testing adaptive repair engine...")
    test_ind = create_random_individual(data)

    logger.info(f"Initial fitness: hard={evaluate(test_ind)[0]}")

    for _ in range(10):
        step_result = repair_engine.step(test_ind)
        logger.debug(
            f"Applied {step_result.operator}: delta_hard={step_result.delta_hard}, "
            f"delta_soft={step_result.delta_soft}"
        )

    logger.info(f"After 10 applications: fitness={evaluate(test_ind)[0]}")

    # RUN ADAPTIVE NSGA-II

    logger.info("Starting Adaptive NSGA-II evolution...")

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
    operator_history: list[dict[str, float]] = []
    repair_history: list[dict[str, float | int]] = []

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

        # MODE D: Adaptive Repair
        repair_start = time.time()
        repair_indices = [
            idx for idx in range(len(offspring)) if random.random() < REPAIR_PROB
        ]
        per_individual_budget = (
            REPAIR_BUDGET_MS / max(1, len(repair_indices))
            if REPAIR_BUDGET_MS > 0
            else 0
        )
        gen_repairs = 0
        gen_delta_hard = 0.0
        gen_delta_soft = 0.0
        for idx in repair_indices:
            ind = offspring[idx]
            repair_stats = repair_engine.repair_individual(
                ind, budget_ms=per_individual_budget, max_steps=REPAIR_MAX_STEPS
            )
            gen_repairs += repair_stats.applied_steps
            gen_delta_hard += repair_stats.total_delta_hard
            gen_delta_soft += repair_stats.total_delta_soft
            if repair_stats.applied_steps > 0:
                del ind.fitness.values
        total_repairs += gen_repairs
        repair_time_ms = (time.time() - repair_start) * 1000

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
        operator_scores: dict[str, float] = {}
        for name, stats_dict in repair_engine.operator_stats.items():
            applied = stats_dict.get("applied", 0.0)
            if applied > 0:
                operator_scores[name] = (
                    stats_dict.get("delta_hard", 0.0) * 1000
                    + stats_dict.get("delta_soft", 0.0)
                ) / applied
            else:
                operator_scores[name] = 0.0
        operator_history.append(operator_scores)
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
            print_constraint_details(hard_bd, soft_bd, gen, logger=logger)
            scores_str = ", ".join(f"{k}:{v:.2f}" for k, v in operator_scores.items())
            logger.debug(
                f"Gen {gen}: repairs={gen_repairs}, delta_hard={gen_delta_hard:.2f}, "
                f"delta_soft={gen_delta_soft:.2f}, scores=[{scores_str}]"
            )

        repair_history.append(
            {
                "generation": gen,
                "repairs_applied": gen_repairs,
                "delta_hard": gen_delta_hard,
                "delta_soft": gen_delta_soft,
                "repair_time_ms": repair_time_ms,
            }
        )

        stats.generation_times.append(time.time() - gen_start)

    stats.elapsed_time = time.time() - start
    logger.info(
        f"Evolution completed in {stats.elapsed_time:.1f}s (total repairs: {total_repairs})"
    )
    logger.info(f"Final heuristic stats: {repair_engine.operator_stats}")

    final_pop = pop

    # RESULTS & VISUALIZATION

    logger.info("Generating results and visualizations...")

    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)
    print_summary(final_pop, stats, breakdown, logger=logger)

    # Plot operator score evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    heuristic_names = list(operator_history[0].keys())
    for name in heuristic_names:
        scores = [p.get(name, 0.0) for p in operator_history]
        ax.plot(scores, label=name)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Improvement Score")
    ax.set_title("Mode D: Adaptive Operator Score Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mode_d_operator_scores.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_d_operator_scores.png'}")

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
        operator_stats=repair_engine.operator_stats,
    )

    metadata = {
        "experiment": "mode_d_adaptive",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "ngen": NGEN,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "fitness_weights": list(FITNESS_WEIGHTS),
            "repair_prob": REPAIR_PROB,
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
            "final_operator_scores": operator_history[-1] if operator_history else {},
        },
        "constraint_breakdown": breakdown,
        "repair_history": repair_history,
        "generation_times": stats.generation_times,
        "operator_stats": repair_engine.operator_stats,
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE D COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
