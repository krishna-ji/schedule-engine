#!/usr/bin/env python3
"""GA Mode 03 — Aggressive: 2x offspring, high mutation, full-pop repair.

Every individual is repaired each generation.  Combined with 2x
offspring and 15% mutation, this trades compute for rapid
constraint reduction.

Usage:
    python runs/ga_03_aggressive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import AggressiveExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42

# GA Core Parameters
POP_SIZE = 200  # Large population
NGEN = 100  # Fewer generations (heavy per-gen compute)
CROSSOVER_PROB = 0.7  # Higher crossover
MUTATION_PROB = 0.15  # High mutation rate

# Aggressive Parameters
N_OFFSPRINGS_MULT = 2.0  # 2× offspring per generation
REPAIR_ITERS = 3  # Repair passes per individual

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None

# Logging
LOG_INTERVAL = 5  # More frequent logging for short runs
VERBOSE = True


def main() -> None:
    """Run GA Aggressive: Large offspring + full-pop repair."""
    exp = AggressiveExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        ngen=NGEN,
        crossover_prob=CROSSOVER_PROB,
        mutation_event_prob=MUTATION_PROB,
        n_offsprings_mult=N_OFFSPRINGS_MULT,
        repair_iters=REPAIR_ITERS,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
