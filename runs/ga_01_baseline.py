#!/usr/bin/env python3
"""GA Mode 01 — Baseline: Pure NSGA-II (no repair, no local search).

Usage:
    python runs/ga_01_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import BaselineExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42

# GA Core Parameters
POP_SIZE = 100  # Population size
NGEN = 200  # Number of generations
CROSSOVER_PROB = 0.5  # Per-event crossover probability
MUTATION_PROB = 0.05  # Per-event mutation probability

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated: output/ga_baseline/<timestamp>

# Logging
LOG_INTERVAL = 1  # Generations between detailed logs
VERBOSE = True


def main() -> None:
    """Run GA Baseline: Pure NSGA-II."""
    exp = BaselineExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        ngen=NGEN,
        crossover_prob=CROSSOVER_PROB,
        mutation_event_prob=MUTATION_PROB,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
        export_pdf=True,
        force_pdf=True,
    )
    exp.run()


if __name__ == "__main__":
    main()
