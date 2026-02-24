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
POP_SIZE = 120  # Population size (was 50 — more diversity)
NGEN = 200  # Number of generations (was 150 — more time to converge)
CROSSOVER_PROB = 0.4  # Per-event crossover probability (was 0.6 — less destructive)
MUTATION_PROB = (
    0.10  # Per-event mutation probability (was 0.08 — slightly more exploration)
)

# Memetic Parameters
ELITE_PCT = (
    0.15  # Fraction of pop to repair each generation (was 0.05 — wider repair coverage)
)
REPAIR_ITERS = 8  # Repair passes per elite individual (was 5 — deeper local search)

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
