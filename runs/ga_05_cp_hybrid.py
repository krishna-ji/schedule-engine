#!/usr/bin/env python3
"""GA Mode 05 — CP Hybrid: NSGA-II + periodic CP-SAT deep polish.

Every CP_INTERVAL generations, converts the best individual to
SessionGene representation, runs ortools CP-SAT for deep constraint
satisfaction, then writes the repaired chromosome back.

Requires: pip install ortools>=9.8

Usage:
    python runs/ga_05_cp_hybrid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import CPHybridExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42

# GA Core Parameters
POP_SIZE = 60  # Smaller pop (CP polish is expensive)
NGEN = 100  # Fewer generations
CROSSOVER_PROB = 0.5  # Standard crossover
MUTATION_PROB = 0.05  # Standard mutation

# CP-SAT Parameters
CP_INTERVAL = 10  # Run CP polish every N generations
CP_TIMEOUT = 30.0  # CP-SAT timeout in seconds

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None

# Logging
LOG_INTERVAL = 5
VERBOSE = True


def main() -> None:
    """Run GA CP-Hybrid: NSGA-II + CP-SAT polish."""
    exp = CPHybridExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        ngen=NGEN,
        crossover_prob=CROSSOVER_PROB,
        mutation_event_prob=MUTATION_PROB,
        cp_interval=CP_INTERVAL,
        cp_timeout=CP_TIMEOUT,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
