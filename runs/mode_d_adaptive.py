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
from schedule_engine.notebooks.strategies import AdaptiveSelector
from schedule_engine.notebooks.viz import (
    plot_constraint_breakdown,
    plot_convergence,
    print_summary,
)


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
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode D: Adaptive Heuristics."""
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
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
    REPAIR_PROB = 0.3
    LEARNING_RATE = 0.1
    MIN_PROB = 0.05
    LOG_INTERVAL = 10

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
    logger.info(f"Config: pop={POP_SIZE}, ngen={NGEN}, learning_rate={LEARNING_RATE}")
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

    evaluate = create_evaluator(data)

    # ==========================================================================
    # TEST ADAPTIVE SELECTOR
    # ==========================================================================
    logger.info("Testing adaptive selector...")
    selector = AdaptiveSelector(learning_rate=LEARNING_RATE, min_prob=MIN_PROB)
    test_ind = create_random_individual(data)

    logger.info(f"Initial probs: {selector.probs}")
    logger.info(f"Initial fitness: hard={evaluate(test_ind)[0]}")

    for _ in range(10):
        name, fixes = selector.apply(test_ind, data)

    logger.info(
        f"After 10 applications: fitness={evaluate(test_ind)[0]}, probs={selector.probs}"
    )

    # ==========================================================================
    # RUN ADAPTIVE NSGA-II
    # ==========================================================================
    logger.info("Starting Adaptive NSGA-II evolution...")

    start = time.time()
    setup_deap(FITNESS_WEIGHTS)
    selector = AdaptiveSelector(
        learning_rate=LEARNING_RATE, min_prob=MIN_PROB
    )  # Fresh selector

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
    prob_history: list[dict[str, float]] = []

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

        # MODE D: Adaptive Repair
        for ind in offspring:
            if random.random() < REPAIR_PROB:
                genes = list(ind)
                _, fixes = selector.apply(genes, data)
                total_repairs += fixes
                ind[:] = genes
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
        prob_history.append(dict(selector.probs))

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
            probs_str = ", ".join(f"{k}:{v:.2f}" for k, v in selector.probs.items())
            logger.debug(f"Gen {gen}: Adaptive probs: [{probs_str}]")

    stats.elapsed_time = time.time() - start
    logger.info(
        f"Evolution completed in {stats.elapsed_time:.1f}s (total repairs: {total_repairs})"
    )
    logger.info(f"Final heuristic stats: {selector.get_stats()}")

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
        stats, OUTPUT_DIR / "mode_d_convergence.png", title_prefix="Mode D: "
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_d_convergence.png'}")

    plot_constraint_breakdown(
        breakdown,
        OUTPUT_DIR / "mode_d_breakdown.png",
        title="Mode D: Constraint Violations",
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_d_breakdown.png'}")

    # Plot probability evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    heuristic_names = list(prob_history[0].keys())
    for name in heuristic_names:
        probs = [p[name] for p in prob_history]
        ax.plot(probs, label=name)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Selection Probability")
    ax.set_title("Mode D: Adaptive Heuristic Probability Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mode_d_probabilities.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_d_probabilities.png'}")

    # ==========================================================================
    # EXPORT RESULTS
    # ==========================================================================
    logger.info("Exporting full results...")
    export_paths = export_full_results(
        population=final_pop,
        stats=stats,
        data=data,
        output_dir=OUTPUT_DIR,
        mode_name="mode_d_adaptive",
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
            "learning_rate": LEARNING_RATE,
            "min_prob": MIN_PROB,
        },
        "results": {
            "elapsed_time": stats.elapsed_time,
            "total_repairs": total_repairs,
            "final_min_hard": stats.min_hard[-1] if stats.min_hard else None,
            "final_min_soft": stats.min_soft[-1] if stats.min_soft else None,
            "final_feasible_count": (
                stats.feasible_count[-1] if stats.feasible_count else 0
            ),
            "final_heuristic_probs": prob_history[-1] if prob_history else {},
        },
        "constraint_breakdown": breakdown,
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE D COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
