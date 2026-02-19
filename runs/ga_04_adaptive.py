#!/usr/bin/env python3
"""GA Mode 04 — Adaptive: Stagnation-aware mutation escalation.

Starts conservative (5% mutation).  When best_hard stalls for
STAGNATION_WINDOW generations, ramps to MUTATION_HI and activates
elite repair.  De-escalates on improvement.

Usage:
    python runs/ga_04_adaptive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import AdaptiveExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42

# GA Core Parameters
POP_SIZE = 100  # Population size
NGEN = 300  # More generations for long adaptive runs
CROSSOVER_PROB = 0.5  # Moderate crossover
MUTATION_PROB = 0.05  # Starting mutation (low)

# Adaptive Parameters
STAGNATION_WINDOW = 15  # Gens without improvement before escalation
MUTATION_HI = 0.20  # Escalated mutation rate
ELITE_PCT = 0.10  # Repair top 10% when escalated
REPAIR_ITERS = 5  # Repair passes per elite individual

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None

# Logging
LOG_INTERVAL = 15
VERBOSE = True


def main() -> None:
    """Run GA Adaptive: Stagnation-aware escalation."""
    exp = AdaptiveExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        ngen=NGEN,
        crossover_prob=CROSSOVER_PROB,
        mutation_event_prob=MUTATION_PROB,
        stagnation_window=STAGNATION_WINDOW,
        mutation_hi=MUTATION_HI,
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
