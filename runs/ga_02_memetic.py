#!/usr/bin/env python3
"""
GA Memetic: NSGA-II + Local Search

Applies local search (greedy hill climbing) after crossover/mutation.
Repair operators fix constraint violations in offspring.

Usage:
    python runs/ga_02_memetic.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

from src.experiments import MemeticExperiment

# ── PRODUCTION CONFIGURATION ─────────────────────────────────────────

# Reproducibility
SEED = 42

# GA Core Parameters
POP_SIZE = 100  # Population size
NGEN = 1000  # Number of generations
CXPB = 0.8  # Crossover probability
MUTPB = 0.4  # Mutation probability
FITNESS_WEIGHTS = (-1.0, -1.0)  # (hard, soft) - negative = minimize

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated: output/ga_02_memetic/<timestamp>

# Time Configuration
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

# Logging
LOG_INTERVAL = 25  # Generations between detailed logs
VERBOSE = True

# ── Mode B Specific: Local Search / Repair ────────────────────────────
LOCAL_SEARCH_PROB = 0.5  # Probability of applying LS to offspring
LOCAL_SEARCH_ITERATIONS = 15  # Max repair steps per individual
REPAIR_POLICY = "round_robin"  # Policy: "round_robin", "random", "ucb"
REPAIR_BUDGET_MS = 200.0  # Time budget for repairs per generation (ms)
REPAIR_MAX_CANDIDATES = 50  # Max candidate moves per step
REPAIR_EPSILON = 0.1  # Exploration rate for adaptive policies


def main() -> None:
    """Run Mode B: Memetic NSGA-II experiment."""
    exp = MemeticExperiment(
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
