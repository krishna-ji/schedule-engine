#!/usr/bin/env python3
"""RL 05 — Reward Shaping: scalar vs hypervolume reward comparison.

Usage:
    python runs/rl_05_compare_rewards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import RLRewardCompareExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42
POP_SIZE = 20
MAX_GENERATIONS = 30
MAX_STEPS = 10
NUM_TRANSITIONS = 10

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None
VERBOSE = True


def main() -> None:
    """Run RL Experiment 05: Reward Shaping Comparison."""
    exp = RLRewardCompareExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        max_steps=MAX_STEPS,
        num_transitions=NUM_TRANSITIONS,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
