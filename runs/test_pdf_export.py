#!/usr/bin/env python3
"""
Quick test to verify schedule.json and calendar.pdf generation.
Runs with only 10 generations to complete quickly.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

from src.experiments import AdaptiveExperiment

# Quick test configuration
SEED = 42
POP_SIZE = 50  # Smaller population
NGEN = 10  # Just 10 generations for quick test
CXPB = 0.9
MUTPB = 0.2
FITNESS_WEIGHTS = (-1.0, -1.0)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated

OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

LOG_INTERVAL = 5
VERBOSE = True

# Repair settings
REPAIR_PROB = 0.45
REPAIR_MAX_STEPS = 5
REPAIR_POLICY = "epsilon_greedy"
REPAIR_BUDGET_MS = 200.0
REPAIR_MAX_CANDIDATES = 50
REPAIR_EPSILON = 0.1


def main() -> None:
    """Run quick test to verify PDF export."""
    print("\n" + "=" * 70)
    print("QUICK TEST: Verifying schedule.json and calendar.pdf generation")
    print("=" * 70 + "\n")

    exp = AdaptiveExperiment(
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
        repair_policy=REPAIR_POLICY,
        repair_budget_ms=REPAIR_BUDGET_MS,
        repair_max_candidates=REPAIR_MAX_CANDIDATES,
        repair_epsilon=REPAIR_EPSILON,
    )
    result = exp.run()

    print("\n" + "=" * 70)
    print("TEST COMPLETE - Check output directory for:")
    print("  - schedule.json")
    print("  - calendar.pdf")
    print(f"\nOutput location: {result['output_dir']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
