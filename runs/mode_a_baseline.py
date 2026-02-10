#!/usr/bin/env python3
"""
Mode A: Baseline Pure NSGA-II

Pure NSGA-II baseline - No enhancements, no repair heuristics, no RL guidance.
This script is the foundation for comparing all other modes (B, C, D, E).

Usage:
    python runs/mode_a_baseline.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import BaselineExperiment

# CONFIGURATION - All tunable parameters


# Reproducibility
SEED = 42

# GA Core Parameters
POP_SIZE = 50  # Population size
NGEN = 1000  # Number of generations
CXPB = 0.9  # Crossover probability
MUTPB = 0.2  # Mutation probability
FITNESS_WEIGHTS = (-1.0, -1.0)  # (hard, soft) - negative = minimize

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated if None (output/mode_a_baseline/<timestamp>)

# Time Configuration
OPENING_TIME = "10:00"  # Day start time (HH:MM)
CLOSING_TIME = "17:00"  # Day end time (HH:MM)
CLOSED_DAYS = ["Saturday"]  # Days with no classes

# Feasibility Check
EXPECTED_QUANTA = 42  # Expected quanta per week

# Logging
LOG_INTERVAL = 20  # Generations between detailed logs
VERBOSE = True  # Enable console output


def main() -> None:
    """Run Mode A: Baseline Pure NSGA-II experiment."""
    exp = BaselineExperiment(
        # Reproducibility
        seed=SEED,
        # GA Core
        pop_size=POP_SIZE,
        ngen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        fitness_weights=FITNESS_WEIGHTS,
        # Paths
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        # Time
        opening_time=OPENING_TIME,
        closing_time=CLOSING_TIME,
        closed_days=CLOSED_DAYS,
        # Feasibility
        expected_quanta=EXPECTED_QUANTA,
        # Logging
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
