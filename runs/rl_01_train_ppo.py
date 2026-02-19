#!/usr/bin/env python3
"""RL 01 — Train PPO: Proximal Policy Optimization baseline.

Usage:
    python runs/rl_01_train_ppo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import RLTrainExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42
POP_SIZE = 20
MAX_GENERATIONS = 50
MAX_STEPS = 20
TIMESTEPS = 5000

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto: output/rl_train_ppo/<timestamp>
VERBOSE = True


def main() -> None:
    """Run RL Experiment 01: PPO Baseline."""
    exp = RLTrainExperiment(
        agent_type="ppo",
        seed=SEED,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        max_steps=MAX_STEPS,
        timesteps=TIMESTEPS,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
