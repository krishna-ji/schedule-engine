#!/usr/bin/env python3
"""RL 07 — Ablation: systematic random / PPO / DQN comparison.

Usage:
    python runs/rl_07_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import RLAblationExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42
POP_SIZE = 20
MAX_GENERATIONS = 50
MAX_STEPS = 20
TIMESTEPS = 3000
TRIALS = 5

METHODS = {
    "random": {"agent_type": "random"},
    "ppo": {"agent_type": "ppo"},
    "dqn": {"agent_type": "dqn"},
}

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None
VERBOSE = True


def main() -> None:
    """Run RL Experiment 07: Full Ablation Study."""
    exp = RLAblationExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        max_steps=MAX_STEPS,
        methods=METHODS,
        trials=TRIALS,
        timesteps=TIMESTEPS,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
