#!/usr/bin/env python3
"""
Mode B: Memetic NSGA-II

NSGA-II + Local Search - Applies local search to improve individuals after genetic operators.

Usage:
    python runs/mode_b_memetic.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.experiments import MemeticExperiment


def main() -> None:
    """Run Mode B: Memetic NSGA-II experiment."""
    exp = MemeticExperiment(
        seed=42,
        pop_size=50,
        ngen=200,
        cxpb=0.8,
        mutpb=0.4,
        fitness_weights=(-1.0, -1.0),
        data_dir=PROJECT_ROOT / "data",
        expected_quanta=42,
        log_interval=10,
        # Memetic-specific
        local_search_prob=0.5,
        local_search_iterations=15,
        repair_policy="round_robin",
        repair_budget_ms=120.0,
        repair_max_candidates=30,
        repair_epsilon=0.1,
    )
    exp.run()


if __name__ == "__main__":
    main()
