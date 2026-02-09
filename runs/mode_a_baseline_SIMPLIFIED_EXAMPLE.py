#!/usr/bin/env python3
"""
Mode A: Baseline Pure NSGA-II (SIMPLIFIED VERSION)

Pure NSGA-II baseline with explicit, self-contained configuration.
No hidden config system, all parameters visible at the top of this file.

Usage:
    python runs/mode_a_baseline_simplified.py
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

# Import instance-level config (institution-specific, rarely changes)
from instance_config import (
    CALENDAR_END_HOUR,
    CALENDAR_QUANTUM_MINUTES,
    CALENDAR_START_HOUR,
    CLOSED_DAYS,
    CLOSING_TIME,
    DATA_DIR,
    DEFAULT_SEED,
    MIDDAY_BREAK_END,
    MIDDAY_BREAK_START,
    NUM_WORKERS,
    OPENING_TIME,
    OUTPUT_DIR,
    QUANTUM_MINUTES,
    USE_MULTIPROCESSING,
)

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
# All experiment-specific parameters are defined here.
# Change these values to configure this specific run.
# =============================================================================

# --- Experiment Metadata ---
EXPERIMENT_NAME = "Mode A: Baseline Pure NSGA-II"
DESCRIPTION = "Pure NSGA-II with no repair, no heuristics, no RL"
SEED = DEFAULT_SEED

# --- Genetic Algorithm ---
NGEN = 100  # Number of generations
POP_SIZE = 50  # Population size
CXPB = 0.90  # Crossover probability
MUTPB = 0.20  # Mutation probability
ELITE_SIZE = 0.05  # Elite preservation ratio (5% of population)
TOURNAMENT_SIZE = 2  # Tournament selection size

# --- Fitness Function ---
FITNESS_WEIGHTS = (-1.0, -0.01)  # (hard, soft) - both minimized, hard prioritized

# --- Repair System ---
REPAIR_ENABLED = False  # Mode A: No repair heuristics
REPAIR_MAX_ITERATIONS = 3
REPAIR_APPLY_AFTER_MUTATION = False
REPAIR_APPLY_AFTER_CROSSOVER = False

# --- Heuristics ---
HEURISTICS_MODE = "off"  # Mode A: No heuristics

# --- LNS (Large Neighborhood Search) ---
LNS_ENABLED = False  # Mode A: No LNS

# --- RL Guidance ---
RL_ENABLED = False  # Mode A: No RL

# --- Population Strategy ---
POPULATION_STRATEGY = "random"  # "random", "smart", or "hybrid"
GREEDY_PERCENTAGE = 0.0  # For hybrid strategy
SMART_PERCENTAGE = 0.0  # For hybrid strategy
RANDOM_PERCENTAGE = 1.0  # Mode A: 100% random

# --- Logging & Output ---
VERBOSE = True
LOG_INTERVAL = 20  # Print stats every N generations
SAVE_CHECKPOINT_INTERVAL = 50  # Save population every N generations

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "experiment.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("mode_a_baseline")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode A: Baseline Pure NSGA-II experiment."""

    # Create output directory with timestamp
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = OUTPUT_DIR / "mode_a_baseline" / TIMESTAMP
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(run_output_dir)
    logger.info("=" * 60)
    logger.info(f"{EXPERIMENT_NAME}")
    logger.info("=" * 60)
    logger.info(f"Description: {DESCRIPTION}")
    logger.info(f"Output: {run_output_dir}")
    logger.info("")

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Seed: {SEED}")
    logger.info(f"  Generations: {NGEN}")
    logger.info(f"  Population Size: {POP_SIZE}")
    logger.info(f"  Crossover Prob: {CXPB}")
    logger.info(f"  Mutation Prob: {MUTPB}")
    logger.info(f"  Elite Size: {ELITE_SIZE}")
    logger.info(f"  Population Strategy: {POPULATION_STRATEGY}")
    logger.info(f"  Repair: {REPAIR_ENABLED}")
    logger.info(f"  Heuristics: {HEURISTICS_MODE}")
    logger.info(f"  LNS: {LNS_ENABLED}")
    logger.info(f"  RL: {RL_ENABLED}")
    logger.info("")

    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)

    # Load data
    logger.info("Loading data...")
    from schedule_engine.ga.run_helpers import (
        EvolutionConfig,
        course_aware_crossover,
        create_evaluator,
        create_random_individual,
        load_data,
        run_nsga2,
        smart_mutation,
    )
    from schedule_engine.io.decoder import decode_individual

    context = load_data(
        data_dir=str(DATA_DIR),
        quantum_minutes=QUANTUM_MINUTES,
        opening_time=OPENING_TIME,
        closing_time=CLOSING_TIME,
        midday_break_start=MIDDAY_BREAK_START,
        midday_break_end=MIDDAY_BREAK_END,
        closed_days=CLOSED_DAYS,
    )

    logger.info(
        f"Loaded: {len(context['courses'])} courses, "
        f"{len(context['groups'])} groups, "
        f"{len(context['instructors'])} instructors, "
        f"{len(context['rooms'])} rooms"
    )
    logger.info("")

    # Create evolution config
    evolution_config = EvolutionConfig()
    evolution_config.pop_size = POP_SIZE
    evolution_config.ngen = NGEN
    evolution_config.cxpb = CXPB
    evolution_config.mutpb = MUTPB
    evolution_config.fitness_weights = FITNESS_WEIGHTS
    evolution_config.verbose = VERBOSE
    evolution_config.log_interval = LOG_INTERVAL

    # Create evaluator
    evaluator = create_evaluator(context)

    # Run NSGA-II evolution
    logger.info("Starting evolution...")
    logger.info("")

    final_population, logbook, stats = run_nsga2(
        context=context,
        config=evolution_config,
        evaluator=evaluator,
        mutation_func=smart_mutation,
        crossover_func=course_aware_crossover,
        individual_creator=create_random_individual,
        # All configuration passed explicitly - no hidden globals!
        elite_size=ELITE_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        use_multiprocessing=USE_MULTIPROCESSING,
        num_workers=NUM_WORKERS,
    )

    logger.info("")
    logger.info("Evolution completed!")
    logger.info("")

    # Get best individual
    from schedule_engine.ga.run_helpers import get_best_individual

    best = get_best_individual(final_population)
    logger.info(
        f"Best solution: Hard={best.fitness.values[0]}, Soft={best.fitness.values[1]}"
    )

    # Save results
    logger.info("Saving results...")

    # Save configuration used for this run
    config_dict = {
        "experiment_name": EXPERIMENT_NAME,
        "description": DESCRIPTION,
        "timestamp": TIMESTAMP,
        "seed": SEED,
        "ga": {
            "ngen": NGEN,
            "pop_size": POP_SIZE,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "elite_size": ELITE_SIZE,
            "tournament_size": TOURNAMENT_SIZE,
            "population_strategy": POPULATION_STRATEGY,
        },
        "repair": {
            "enabled": REPAIR_ENABLED,
            "max_iterations": REPAIR_MAX_ITERATIONS,
        },
        "heuristics": {"mode": HEURISTICS_MODE},
        "lns": {"enabled": LNS_ENABLED},
        "rl": {"enabled": RL_ENABLED},
        "time_system": {
            "quantum_minutes": QUANTUM_MINUTES,
            "opening_time": OPENING_TIME,
            "closing_time": CLOSING_TIME,
        },
    }

    with open(run_output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save best solution
    best_decoded = decode_individual(best, context)
    with open(run_output_dir / "best_solution.json", "w") as f:
        json.dump(
            {
                "fitness": {
                    "hard": best.fitness.values[0],
                    "soft": best.fitness.values[1],
                },
                "schedule": [
                    {
                        "course": s.course_id,
                        "type": s.course_type,
                        "groups": s.group_ids,
                        "instructor": s.instructor_id,
                        "room": s.room.room_id,
                        "quanta": s.session_quanta,
                    }
                    for s in best_decoded
                ],
            },
            f,
            indent=2,
        )

    # Save statistics
    from schedule_engine.ga.run_helpers import stats_to_ga_metrics

    metrics = stats_to_ga_metrics(stats)
    with open(run_output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate summary
    from schedule_engine.viz import print_summary

    print_summary(stats, run_output_dir)

    logger.info(f"Results saved to: {run_output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
