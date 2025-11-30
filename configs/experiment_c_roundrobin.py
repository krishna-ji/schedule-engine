"""
Experiment C: Round-Robin Heuristics

NSGA-II with fixed rotation through heuristic operators.
Tests effectiveness of heuristic toolbox.
"""

from configs.experiments.roundrobin import (
    EXPERIMENT_DESCRIPTION as _EXPERIMENT_DESCRIPTION,
)
from configs.experiments.roundrobin import EXPERIMENT_ID as _EXPERIMENT_ID
from configs.experiments.roundrobin import EXPERIMENT_NAME as _EXPERIMENT_NAME
from configs.experiments.roundrobin import KILLSWITCHES as _KILLSWITCHES
from configs.experiments.roundrobin import RoundRobinProdConfig, RoundRobinTestConfig
from configs.profiles import Profile

# Use dataclass configs
experiment_c_test = RoundRobinTestConfig()
experiment_c_prod = RoundRobinProdConfig()

# Experiment metadata (imported from dataclass module)
EXPERIMENT_ID = _EXPERIMENT_ID
EXPERIMENT_NAME = _EXPERIMENT_NAME
EXPERIMENT_DESCRIPTION = _EXPERIMENT_DESCRIPTION
KILLSWITCHES = _KILLSWITCHES


# Quick usage
def get_config(profile: Profile = Profile.TEST, **overrides):
    """Get config for Experiment C."""
    # Filter out None values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if profile == Profile.TEST:
        config = RoundRobinTestConfig(**overrides)
    else:
        config = RoundRobinProdConfig(**overrides)
    return config.to_pydantic()


if __name__ == "__main__":
    test_cfg = experiment_c_test
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  repair={test_cfg.repair_enabled}")
    print(f"  heuristics={test_cfg.heuristics_master_enabled}")
    print(f"  adaptive={test_cfg.heuristics_adaptive_priority_enabled}")
