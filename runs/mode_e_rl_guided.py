#!/usr/bin/env python3
"""
Mode E: RL-Guided NSGA-II

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

# CONFIGURATION - All tunable parameters


# Reproducibility
SEED = 42

# GA Core Parameters
POP_SIZE = 50  # Population size
NGEN = 100  # Number of generations
CXPB = 0.9  # Crossover probability
MUTPB = 0.2  # Mutation probability
FITNESS_WEIGHTS = (-1.0, -1.0)  # (hard, soft) - negative = minimize

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated if None

# Time Configuration
OPENING_TIME = "10:00"  # Day start time (HH:MM)
CLOSING_TIME = "17:00"  # Day end time (HH:MM)
CLOSED_DAYS = ["Saturday"]  # Days with no classes

# Feasibility Check
EXPECTED_QUANTA = 42  # Expected quanta per week

# Logging
LOG_INTERVAL = 10  # Generations between detailed logs
VERBOSE = True  # Enable console output

# --- Mode E Specific: RL-Guided Repair ---
REPAIR_PROB = 0.3  # Probability of applying repair to offspring
LEARNING_RATE = 0.2  # Q-learning alpha (how fast to update Q-values)
EPSILON_START = 1.0  # Initial exploration rate (1.0 = fully random)
EPSILON_END = 0.1  # Final exploration rate (0.0 = fully greedy)
EPSILON_DECAY = 0.995  # Multiplicative decay per generation


def main() -> None:
    """Run Mode E: RL-Guided NSGA-II experiment."""
    exp = RLGuidedExperiment(
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
        # Mode E: RL-Guided
        repair_prob=REPAIR_PROB,
        learning_rate=LEARNING_RATE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
    )
    exp.run()


if __name__ == "__main__":
    main()
