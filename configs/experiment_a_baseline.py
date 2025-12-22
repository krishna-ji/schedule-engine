"""
Experiment A: Pure NSGA-II Baseline

Minimal NSGA-II with all enhancements disabled.
Serves as baseline for comparing other experiments.
"""

from configs.experiments.baseline import (
    EXPERIMENT_DESCRIPTION as _EXPERIMENT_DESCRIPTION,
)
from configs.experiments.baseline import EXPERIMENT_ID as _EXPERIMENT_ID
from configs.experiments.baseline import EXPERIMENT_NAME as _EXPERIMENT_NAME
from configs.experiments.baseline import KILLSWITCHES as _KILLSWITCHES
from configs.experiments.baseline import BaselineProdConfig, BaselineTestConfig
from configs.profiles import Profile

# Use dataclass configs
experiment_a_test = BaselineTestConfig()
experiment_a_prod = BaselineProdConfig()

# Experiment metadata (imported from dataclass module)
EXPERIMENT_ID = _EXPERIMENT_ID
EXPERIMENT_NAME = _EXPERIMENT_NAME
EXPERIMENT_DESCRIPTION = _EXPERIMENT_DESCRIPTION
KILLSWITCHES = _KILLSWITCHES


# Quick usage
def get_config(profile: Profile = Profile.TEST, **overrides):
    """Get config for Experiment A."""
    # Filter out None values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config: BaselineTestConfig | BaselineProdConfig
    if profile == Profile.TEST:
        config = BaselineTestConfig(**overrides)
    else:
        config = BaselineProdConfig(**overrides)
    return config.to_pydantic()


if __name__ == "__main__":
    # Test the config
    test_cfg = experiment_a_test
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  repair={test_cfg.repair_enabled}")
    print(f"  heuristics={test_cfg.heuristics_master_enabled}")
    print(f"  rl={test_cfg.rl_enabled}")
