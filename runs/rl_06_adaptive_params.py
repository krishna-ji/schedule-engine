#!/usr/bin/env python3
"""RL 06 — Adaptive Params: fixed vs adaptive GA parameter comparison.

Usage:
    python runs/rl_06_adaptive_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import RLAdaptiveParamsExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

SEED = 42
OUTPUT_DIR = None
VERBOSE = True


def main() -> None:
    """Run RL Experiment 06: Adaptive Probabilities."""
    exp = RLAdaptiveParamsExperiment(
        seed=SEED,
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
