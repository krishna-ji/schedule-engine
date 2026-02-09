#!/usr/bin/env python3
"""
Mode D: Adaptive Heuristics

NSGA-II + Adaptive Selection - Learns which heuristics work best.

Usage:
    python runs/mode_d_adaptive.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import AdaptiveExperiment


def main() -> None:
    """Run Mode D: Adaptive Heuristics experiment."""
    exp = AdaptiveExperiment(
        seed=42,
        pop_size=50,
        ngen=100,
        cxpb=0.9,
        mutpb=0.2,
        fitness_weights=(-1.0, -1.0),
        data_dir=PROJECT_ROOT / "data",
        expected_quanta=42,
        log_interval=10,
        # Adaptive-specific
        repair_prob=0.45,
        repair_max_steps=3,
        repair_policy="epsilon_greedy",
        repair_epsilon=0.1,
        repair_budget_ms=120.0,
        repair_max_candidates=30,
    )
    exp.run()


if __name__ == "__main__":
    main()
