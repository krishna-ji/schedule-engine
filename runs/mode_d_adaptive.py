#!/usr/bin/env python3
"""
Mode D: Adaptive Heuristics

NSGA-II + Adaptive Selection - Uses epsilon-greedy or UCB to learn best heuristics.

Usage:
    python runs/mode_d_adaptive.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import AdaptiveExperiment

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

# --- Mode D Specific: Adaptive Repair ---
REPAIR_PROB = 0.45  # Probability of applying repair to offspring
REPAIR_MAX_STEPS = 3  # Max repair steps per individual
REPAIR_POLICY = "epsilon_greedy"  # Policy: "epsilon_greedy", "ucb", "softmax"
REPAIR_BUDGET_MS = 120.0  # Time budget for repairs per generation (ms)
REPAIR_MAX_CANDIDATES = 30  # Max candidate moves per step
REPAIR_EPSILON = 0.1  # Exploration rate for epsilon-greedy


def main() -> None:
    """Run Mode D: Adaptive Heuristics experiment."""
    exp = AdaptiveExperiment(
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
        # Mode D: Adaptive Repair
        repair_prob=REPAIR_PROB,
        repair_max_steps=REPAIR_MAX_STEPS,
        repair_policy=REPAIR_POLICY,
        repair_budget_ms=REPAIR_BUDGET_MS,
        repair_max_candidates=REPAIR_MAX_CANDIDATES,
        repair_epsilon=REPAIR_EPSILON,
    )
    exp.run()


if __name__ == "__main__":
    main()
