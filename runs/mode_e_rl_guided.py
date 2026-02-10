#!/usr/bin/env python3
"""
Mode E: RL-Guided NSGA-II  (Production)

Full deployment with RL-guided heuristic selection using Q-learning.

Usage:
    python runs/mode_e_rl_guided.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import RLGuidedExperiment

# ── PRODUCTION CONFIGURATION ─────────────────────────────────────────

# Reproducibility
SEED = 42

# GA Core Parameters
POP_SIZE = 100  # Population size
NGEN = 1000  # Number of generations
CXPB = 0.9  # Crossover probability
MUTPB = 0.2  # Mutation probability
FITNESS_WEIGHTS = (-1.0, -1.0)  # (hard, soft) - negative = minimize

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated: output/mode_e_rl_guided/<timestamp>

# Time Configuration
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

# Feasibility Check
EXPECTED_QUANTA = 42

# Logging
LOG_INTERVAL = 25  # Generations between detailed logs
VERBOSE = True

# ── Mode E Specific: RL-Guided Repair ─────────────────────────────────
REPAIR_PROB = 0.3  # Probability of applying repair to offspring
LEARNING_RATE = 0.2  # Q-learning alpha (how fast to update Q-values)
EPSILON_START = 1.0  # Initial exploration rate (1.0 = fully random)
EPSILON_END = 0.1  # Final exploration rate (0.0 = fully greedy)
EPSILON_DECAY = 0.997  # Multiplicative decay per generation (slower for 1000 gens)


def main() -> None:
    """Run Mode E: RL-Guided NSGA-II experiment."""
    exp = RLGuidedExperiment(
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
        repair_prob=REPAIR_PROB,
        learning_rate=LEARNING_RATE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
    )
    exp.run()


if __name__ == "__main__":
    main()
