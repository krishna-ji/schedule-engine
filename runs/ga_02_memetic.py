#!/usr/bin/env python3
"""GA Mode 02 — Memetic: NSGA-II + elite bitset repair.

Applies bitset-based repair to the top *ELITE_PCT* individuals
each generation, accelerating constraint satisfaction.

Usage:
    python runs/ga_02_memetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import MemeticExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42

# GA Core Parameters
POP_SIZE = 80  # Population size
NGEN = 150  # Number of generations
CROSSOVER_PROB = 0.6  # Per-event crossover probability
MUTATION_PROB = 0.08  # Per-event mutation probability

# Memetic Parameters
ELITE_PCT = 0.05  # Fraction of pop to repair each generation
REPAIR_ITERS = 5  # Repair passes per elite individual

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated

# Logging
LOG_INTERVAL = 10
VERBOSE = True


def main() -> None:
    """Run GA Memetic: NSGA-II + Elite Repair."""
    exp = MemeticExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        ngen=NGEN,
        crossover_prob=CROSSOVER_PROB,
        mutation_event_prob=MUTATION_PROB,
        elite_pct=ELITE_PCT,
        repair_iters=REPAIR_ITERS,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
