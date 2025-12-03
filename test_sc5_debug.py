"""Quick test script to enable SC5 debug mode and run a baseline evaluation.

This script temporarily enables debug_sc5 flag and runs a minimal test
to track SC5 constraint evaluation.

Usage:
    uv run python test_sc5_debug.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataclasses import dataclass

from configs.profiles import TestConfig
from src.encoder.input_encoder import derive_cohort_pairs_from_groups


@dataclass
class SC5DebugConfig(TestConfig):
    """Test config with SC5 debug enabled."""

    # Enable debug mode
    debug_sc5: bool = True

    # Minimal run for quick testing
    ngen: int = 5
    pop_size: int = 10

    # Disable repair for cleaner output
    repair_enabled: bool = False
    heuristics_master_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False


if __name__ == "__main__":
    # Create debug config instance
    debug_config = SC5DebugConfig()

    groups_path = project_root / "data" / "Groups.json"
    debug_config.cohort_pairs = derive_cohort_pairs_from_groups(str(groups_path))

    print("=" * 80)
    print("SC5 DEBUG MODE TEST")
    print("=" * 80)
    print(f"debug_sc5: {debug_config.debug_sc5}")
    print(f"Generations: {debug_config.ngen}")
    print(f"Population: {debug_config.pop_size}")
    print("=" * 80)
    print()

    # Run experiment with debug config
    # Note: You'll need to modify main.py to accept a config instance
    # For now, let's create a minimal test

    from src.config import set_config_instance
    from src.config.config import Config
    from src.core.ga_scheduler import GAScheduler
    from src.encoder.encoder import load_and_encode_data

    # Convert dataclass to Config model
    config_dict = {
        **debug_config.__dict__,
        "time": {
            "quantum_minutes": debug_config.quantum_minutes,
            "opening_time": debug_config.opening_time,
            "closing_time": debug_config.closing_time,
            "closed_days": debug_config.closed_days,
            "cohort_pairs": debug_config.cohort_pairs,
            "break_window_start": debug_config.break_window_start,
            "break_window_end": debug_config.break_window_end,
        },
        "ga_params": {
            "ngen": debug_config.ngen,
            "pop_size": debug_config.pop_size,
            "cx_prob": debug_config.cx_prob,
            "mut_prob": debug_config.mut_prob,
            "tournament_size": debug_config.tournament_size,
        },
        "soft_constraints": {},
        "hard_constraints": {},
        "parallel": {
            "use_multiprocessing": False,  # Disable for cleaner debug output
        },
        "metrics": {
            "compute_advanced_metrics": True,
            "advanced_metrics_frequency": 1,
        },
    }

    config = Config(**config_dict)
    set_config_instance(config)

    # Load data
    data_dir = Path("data")
    entities = load_and_encode_data(str(data_dir))

    # Create scheduler
    scheduler = GAScheduler(entities)

    # Run evolution
    print("\nStarting evolution with SC5 debug enabled...")
    print("Look for [SC5 DEBUG] messages in the output\n")

    result = scheduler.run()

    print("\n" + "=" * 80)
    print("SC5 DEBUG TEST COMPLETE")
    print("=" * 80)
    print(f"Final population size: {len(result.final_population)}")
    print(f"Best fitness: {result.best_individual.fitness.values}")
    print("=" * 80)
