#!/usr/bin/env python3
"""
Mode A: Baseline Pure NSGA-II

Pure NSGA-II baseline - No enhancements, no repair heuristics, no RL guidance.
This script is the foundation for comparing all other modes (B, C, D, E).

Usage:
    python runs/mode_a_baseline.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.ga.run_helpers import (
    EvolutionConfig,
    course_aware_crossover,
    create_evaluator,
    create_random_individual,
    get_best_individual,
    get_constraint_breakdown,
    load_data,
    run_nsga2,
    smart_mutation,
    stats_to_ga_metrics,
)
from schedule_engine.io.decoder import decode_individual
from schedule_engine.utils.json_utils import to_jsonable
from schedule_engine.viz import print_summary
from schedule_engine.workflows.feasibility_checks import run_feasibility_checks
from schedule_engine.workflows.reporting import generate_reports


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_a_baseline.log"

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger("mode_a_baseline")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode A: Baseline Pure NSGA-II experiment."""

    # CONFIGURATION

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters - SAME AS MODE B1 for fair comparison
    POP_SIZE = 50
    NGEN = 1000
    CXPB = 0.9
    MUTPB = 0.2
    EXPECTED_QUANTA = 42

    # Fitness weights: -1.0 = minimize both (equal weight)
    FITNESS_WEIGHTS = (-1.0, -1.0)

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "mode_a_baseline" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("MODE A: BASELINE PURE NSGA-II")
    logger.info("=" * 60)
    logger.info(f"Config: pop={POP_SIZE}, ngen={NGEN}, weights={FITNESS_WEIGHTS}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # Evolution config
    config = EvolutionConfig(
        pop_size=POP_SIZE,
        ngen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        fitness_weights=FITNESS_WEIGHTS,
        verbose=True,
        log_interval=20,
    )

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

    # TEST COMPONENTS

    logger.info("Testing components...")
    test_ind = create_random_individual(data)
    logger.info(f"Individual has {len(test_ind)} genes")

    evaluate = create_evaluator(data)
    test_fitness = evaluate(test_ind)
    logger.info(f"Test fitness: hard={test_fitness[0]}, soft={test_fitness[1]}")

    # RUN NSGA-II EVOLUTION

    logger.info("Starting NSGA-II evolution...")
    final_pop, stats = run_nsga2(
        data=data,
        config=config,
        create_individual_fn=create_random_individual,
        evaluate_fn=evaluate,
        crossover_fn=course_aware_crossover,
        mutate_fn=lambda ind: smart_mutation(ind, data),
        seed=SEED,
        logger=logger,
    )
    logger.info(f"Evolution completed in {stats.elapsed_time:.1f}s")

    # RESULTS & VISUALIZATION

    logger.info("Generating results and visualizations...")

    # Get best solution
    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)

    # Print summary
    print_summary(final_pop, stats, breakdown, logger=logger)

    # NSGA-II METRICS SUMMARY

    logger.info("Calculating NSGA-II quality metrics...")

    spacing = stats.spacing[-1] if stats.spacing else 0.0
    logger.info(f"  Spacing: {spacing:.4f} (lower = more uniform)")

    hypervolume = stats.hypervolume[-1] if stats.hypervolume else 0.0
    logger.info(f"  Hypervolume: {hypervolume:.2f} (higher = better)")

    diversity = stats.diversity[-1] if stats.diversity else 0.0
    logger.info(f"  Population Diversity: {diversity:.4f} (higher = more diverse)")

    # EXPORT RESULTS (FULL NSGA REPORTS)

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
        generation_times=stats.generation_times,
    )

    # Save experiment metadata (convert numpy types to native Python)
    def to_native(val: int | float | np.integer | np.floating | None) -> int | float | None:  # type: ignore[type-arg]
        """Convert numpy scalar to native Python type."""
        if val is None:
            return None
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        return val

    metadata = {
        "experiment": "mode_a_baseline",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "ngen": NGEN,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "fitness_weights": list(FITNESS_WEIGHTS),
        },
        "results": {
            "elapsed_time": to_native(stats.elapsed_time),
            "final_min_hard": to_native(stats.min_hard[-1]) if stats.min_hard else None,
            "final_min_soft": to_native(stats.min_soft[-1]) if stats.min_soft else None,
            "final_feasible_count": (
                to_native(stats.feasible_count[-1]) if stats.feasible_count else 0
            ),
        },
        "nsga2_metrics": {
            "spacing": to_native(spacing),
            "hypervolume": to_native(hypervolume),
            "population_diversity": to_native(diversity),
            "pareto_front_size": (
                stats.pareto_front_size[-1] if stats.pareto_front_size else 0
            ),
        },
        "constraint_breakdown": {k: to_native(v) for k, v in breakdown.items()},
        "generation_times": [to_native(v) for v in stats.generation_times],
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE A COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
