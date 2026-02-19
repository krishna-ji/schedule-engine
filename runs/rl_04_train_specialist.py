#!/usr/bin/env python3
"""RL 04 — Specialist Agents: state-based agent selection analysis.

Usage:
    python runs/rl_04_train_specialist.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import RLSpecialistExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42
POP_SIZE = 20
MAX_GENERATIONS = 50
MAX_STEPS = 15
NUM_EPISODES = 5
STRATEGY = "state_based"

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None
VERBOSE = True


def main() -> None:
    """Run RL Experiment 04: Specialist Agents."""
    exp = RLSpecialistExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        max_steps=MAX_STEPS,
        num_episodes=NUM_EPISODES,
        strategy=STRATEGY,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
