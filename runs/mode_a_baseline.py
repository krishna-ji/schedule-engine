#!/usr/bin/env python3
"""
Mode A: Baseline Pure NSGA-II  (Production)

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

# ── PRODUCTION CONFIGURATION ─────────────────────────────────────────

# Reproducibility
SEED = 42

# GA Core Parameters
POP_SIZE = 100  # Population size (larger = more diversity)
NGEN = 2000  # Number of generations
CXPB = 0.9  # Crossover probability
MUTPB = 0.2  # Mutation probability
FITNESS_WEIGHTS = (-1.0, -1.0)  # (hard, soft) - negative = minimize

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated: output/mode_a_baseline/<timestamp>

# Time Configuration
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

# Feasibility Check
EXPECTED_QUANTA = 42

# Logging
LOG_INTERVAL = 50  # Generations between detailed logs
VERBOSE = True


def main() -> None:
    """Run Mode A: Baseline Pure NSGA-II experiment."""
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
        expected_quanta=EXPECTED_QUANTA,
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
