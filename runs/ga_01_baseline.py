#!/usr/bin/env python3
"""
GA Baseline: Pure NSGA-II

Pure NSGA-II with constraint-guided mutation, group-clash repair, and
parallel fitness evaluation.  No local search, no RL guidance.
This script is the foundation for comparing all other modes.

Usage:
    python runs/ga_01_baseline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import BaselineExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42

# GA Core Parameters
POP_SIZE = 50  # Population size (50 gives healthy diversity for ~550 genes)
NGEN = 2000  # Number of generations
CXPB = 0.9  # Crossover probability
MUTPB = 0.3  # Mutation probability (raised: guided mutation is targeted)
FITNESS_WEIGHTS = (-1.0, -1.0)  # (hard, soft) — minimize both

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated: output/ga_01_baseline/<timestamp>

# Time Configuration
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

# Population Initialization
INIT_STRATEGY = "smart"  # Options: "smart", "hybrid", "random"

# Logging
LOG_INTERVAL = 50  # Generations between detailed logs
VERBOSE = True


def main() -> None:
    """Run GA Baseline: Pure NSGA-II."""
    exp = BaselineExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        ngen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        fitness_weights=FITNESS_WEIGHTS,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        opening_time=OPENING_TIME,
        closing_time=CLOSING_TIME,
        closed_days=CLOSED_DAYS,
        init_strategy=INIT_STRATEGY,
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
