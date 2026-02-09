#!/usr/bin/env python3
"""
Mode A: Baseline Pure NSGA-II

Pure NSGA-II baseline - No enhancements, no repair heuristics, no RL guidance.
This script is the foundation for comparing all other modes (B, C, D, E).

Usage:
    python runs/mode_a_baseline.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import BaselineExperiment


def main() -> None:
    """Run Mode A: Baseline Pure NSGA-II experiment."""
    exp = BaselineExperiment(
        seed=42,
        pop_size=50,
        ngen=1000,
        cxpb=0.9,
        mutpb=0.2,
        fitness_weights=(-1.0, -1.0),
        data_dir=PROJECT_ROOT / "data",
        expected_quanta=42,
        log_interval=20,
    )
    exp.run()


if __name__ == "__main__":
    main()
