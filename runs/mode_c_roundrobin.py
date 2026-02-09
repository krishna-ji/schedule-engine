#!/usr/bin/env python3
"""
Mode C: Round-Robin Heuristics

NSGA-II + Round-Robin heuristic selection.

Usage:
    python runs/mode_c_roundrobin.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import RoundRobinExperiment


def main() -> None:
    """Run Mode C: Round-Robin NSGA-II experiment."""
    exp = RoundRobinExperiment(
        seed=42,
        pop_size=50,
        ngen=100,
        cxpb=0.9,
        mutpb=0.2,
        fitness_weights=(-1.0, -1.0),
        data_dir=PROJECT_ROOT / "data",
        expected_quanta=42,
        log_interval=10,
        # Round-robin specific
        repair_prob=0.3,
        repair_max_steps=3,
        repair_budget_ms=120.0,
        repair_max_candidates=30,
    )
    exp.run()


if __name__ == "__main__":
    main()
