#!/usr/bin/env python3
"""
Mode E: RL-Guided NSGA-II

Full deployment with RL-guided heuristic selection using Q-learning.

Usage:
    python runs/mode_e_rl_guided.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import RLGuidedExperiment


def main() -> None:
    """Run Mode E: RL-Guided NSGA-II experiment."""
    exp = RLGuidedExperiment(
        seed=42,
        pop_size=50,
        ngen=100,
        cxpb=0.9,
        mutpb=0.2,
        fitness_weights=(-1.0, -1.0),
        data_dir=PROJECT_ROOT / "data",
        expected_quanta=42,
        log_interval=10,
        # RL-specific
        repair_prob=0.3,
        learning_rate=0.2,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
    )
    exp.run()


if __name__ == "__main__":
    main()
