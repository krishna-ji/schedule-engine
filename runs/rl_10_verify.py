#!/usr/bin/env python3
"""RL 10 — Verify: component availability check.

Usage:
    python runs/rl_10_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import RLVerifyExperiment

# ── CONFIGURATION ─────────────────────────────────────────────────────

OUTPUT_DIR = None
VERBOSE = True


def main() -> None:
    """Run RL Experiment 10: Component Verification."""
    exp = RLVerifyExperiment(
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE,
    )
    exp.run()


if __name__ == "__main__":
    main()
