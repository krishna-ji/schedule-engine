#!/usr/bin/env python3
"""RL 08 — Hyperparameter Sweep: learning-rate sensitivity.

Usage:
    python runs/rl_08_hyperparam_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import RLHyperparamSweepExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42
POP_SIZE = 20
MAX_GENERATIONS = 40
MAX_STEPS = 15
TIMESTEPS = 3000

LEARNING_RATES = [1e-4, 3e-4, 1e-3]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None
VERBOSE = True


def main() -> None:
    """Run RL Experiment 08: Hyperparameter Sensitivity."""
    exp = RLHyperparamSweepExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        max_steps=MAX_STEPS,
        learning_rates=LEARNING_RATES,
        timesteps=TIMESTEPS,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
