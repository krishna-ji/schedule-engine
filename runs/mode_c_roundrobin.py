#!/usr/bin/env python3
"""
Mode C: Round-Robin Heuristics  (Production)

NSGA-II + Round-Robin heuristic selection - applies heuristics in fixed order.

Usage:
    python runs/mode_c_roundrobin.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schedule_engine.experiments import RoundRobinExperiment

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
OUTPUT_DIR = None  # Auto-generated: output/mode_c_roundrobin/<timestamp>

# Time Configuration
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

# Logging
LOG_INTERVAL = 25  # Generations between detailed logs
VERBOSE = True

# ── Mode C Specific: Round-Robin Repair ───────────────────────────────
REPAIR_PROB = 0.3  # Probability of applying repair to offspring
REPAIR_MAX_STEPS = 5  # Max repair steps per individual
REPAIR_BUDGET_MS = 200.0  # Time budget for repairs per generation (ms)
REPAIR_MAX_CANDIDATES = 50  # Max candidate moves per step
REPAIR_EPSILON = 0.1  # Exploration rate (unused in round_robin)


def main() -> None:
    """Run Mode C: Round-Robin NSGA-II experiment."""
    exp = RoundRobinExperiment(
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
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
        repair_prob=REPAIR_PROB,
        repair_max_steps=REPAIR_MAX_STEPS,
        repair_budget_ms=REPAIR_BUDGET_MS,
        repair_max_candidates=REPAIR_MAX_CANDIDATES,
        repair_epsilon=REPAIR_EPSILON,
    )
    exp.run()


if __name__ == "__main__":
    main()
