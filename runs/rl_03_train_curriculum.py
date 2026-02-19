#!/usr/bin/env python3
"""RL 03 — Curriculum Learning: progressive-stage PPO training.

Usage:
    python runs/rl_03_train_curriculum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import RLCurriculumExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42
POP_SIZE = 20

STAGES = [
    {"name": "easy", "max_generations": 30, "max_steps": 10, "timesteps": 3000},
    {"name": "medium", "max_generations": 50, "max_steps": 15, "timesteps": 4000},
    {"name": "hard", "max_generations": 80, "max_steps": 20, "timesteps": 5000},
]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None
VERBOSE = True


def main() -> None:
    """Run RL Experiment 03: Curriculum Learning."""
    exp = RLCurriculumExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        stages=STAGES,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
