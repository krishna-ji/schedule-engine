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

from schedule_engine.notebooks.core import (EvolutionConfig, course_aware_crossover,
                                            create_evaluator, create_random_individual,
                                            get_best_individual,
                                            get_constraint_breakdown, load_data,
                                            run_nsga2, smart_mutation)
from schedule_engine.notebooks.export import export_full_results
from schedule_engine.notebooks.viz import (plot_constraint_breakdown, plot_convergence,
                                           print_summary)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_a_baseline.log"
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
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
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main() -> None:
    """Run Mode A: Baseline Pure NSGA-II experiment."""
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    
    # GA Parameters - SAME AS MODE B1 for fair comparison
    POP_SIZE = 50
    NGEN = 200
    CXPB = 0.9
    MUTPB = 0.2
    
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
    
    # ==========================================================================
    # TEST COMPONENTS
    # ==========================================================================
    logger.info("Testing components...")
    test_ind = create_random_individual(data)
    logger.info(f"Individual has {len(test_ind)} genes")
    
    evaluate = create_evaluator(data)
    test_fitness = evaluate(test_ind)
    logger.info(f"Test fitness: hard={test_fitness[0]}, soft={test_fitness[1]}")
    
    # ==========================================================================
    # RUN NSGA-II EVOLUTION
    # ==========================================================================
    logger.info("Starting NSGA-II evolution...")
    final_pop, stats = run_nsga2(
        data=data,
        config=config,
        create_individual_fn=create_random_individual,
        evaluate_fn=evaluate,
        crossover_fn=course_aware_crossover,
        mutate_fn=lambda ind: smart_mutation(ind, data),
        seed=SEED,
    )
    logger.info(f"Evolution completed in {stats.elapsed_time:.1f}s")
    
    # ==========================================================================
    # RESULTS & VISUALIZATION
    # ==========================================================================
    logger.info("Generating results and visualizations...")
    
    # Get best solution
    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)
    
    # Print summary
    print_summary(final_pop, stats, breakdown)
    
    # Plot results - AUTO EXPORT FIGURES
    logger.info("Exporting figures...")
    plot_convergence(
        stats,
        OUTPUT_DIR / "mode_a_convergence.png",
        title_prefix="Mode A: "
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_a_convergence.png'}")
    
    plot_constraint_breakdown(
        breakdown,
        OUTPUT_DIR / "mode_a_breakdown.png",
        title="Mode A: Constraint Violations"
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_a_breakdown.png'}")
    
    # ==========================================================================
    # EXPORT RESULTS
    # ==========================================================================
    logger.info("Exporting full results...")
    export_paths = export_full_results(
        population=final_pop,
        stats=stats,
        data=data,
        output_dir=OUTPUT_DIR,
        mode_name="mode_a_baseline",
    )
    
    # Save experiment metadata
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
            "elapsed_time": stats.elapsed_time,
            "final_min_hard": stats.min_hard[-1] if stats.min_hard else None,
            "final_min_soft": stats.min_soft[-1] if stats.min_soft else None,
            "final_feasible_count": stats.feasible_count[-1] if stats.feasible_count else 0,
        },
        "constraint_breakdown": breakdown,
    }
    
    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")
    
    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE A COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
