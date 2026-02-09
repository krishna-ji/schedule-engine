#!/usr/bin/env python3
"""
Mode B: Memetic NSGA-II

NSGA-II + Local Search - Applies local search to improve individuals after genetic operators.

Usage:
    python runs/mode_b_memetic.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import MemeticExperiment

# =============================================================================
# CONFIGURATION - All tunable parameters
# =============================================================================

# Reproducibility
SEED = 42

# GA Core Parameters
POP_SIZE = 50  # Population size
NGEN = 200  # Number of generations
CXPB = 0.8  # Crossover probability
MUTPB = 0.4  # Mutation probability
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

# --- Mode B Specific: Local Search / Repair ---
LOCAL_SEARCH_PROB = 0.5  # Probability of applying LS to offspring
LOCAL_SEARCH_ITERATIONS = 15  # Max repair steps per individual
REPAIR_POLICY = "round_robin"  # Policy: "round_robin", "random", "ucb"
REPAIR_BUDGET_MS = 120.0  # Time budget for repairs per generation (ms)
REPAIR_MAX_CANDIDATES = 30  # Max candidate moves per step
REPAIR_EPSILON = 0.1  # Exploration rate for adaptive policies

# =============================================================================


def main() -> None:
    """Run Mode B: Memetic NSGA-II experiment."""
    exp = MemeticExperiment(
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
        # Mode B: Local Search
        local_search_prob=LOCAL_SEARCH_PROB,
        local_search_iterations=LOCAL_SEARCH_ITERATIONS,
        repair_policy=REPAIR_POLICY,
        repair_budget_ms=REPAIR_BUDGET_MS,
        repair_max_candidates=REPAIR_MAX_CANDIDATES,
        repair_epsilon=REPAIR_EPSILON,
    )
    exp.run()


if __name__ == "__main__":
    main()
