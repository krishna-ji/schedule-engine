#!/usr/bin/env python3
"""
Quick test for ga_01_baseline - 10 generations to verify PDF export.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import BaselineExperiment

# Quick test configuration
SEED = 42
POP_SIZE = 50
NGEN = 10  # Quick test
CXPB = 0.9
MUTPB = 0.2
FITNESS_WEIGHTS = (-1.0, -1.0)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None

OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

LOG_INTERVAL = 5
VERBOSE = True


def main() -> None:
    """Run quick baseline test."""
    print("\n" + "=" * 70)
    print("TESTING: ga_01_baseline - PDF export")
    print("=" * 70 + "\n")

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
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
    )
    exp.run()

    print("\n" + "=" * 70)
    print("Baseline test complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
