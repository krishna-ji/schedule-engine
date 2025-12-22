"""
Experiment B: Memetic NSGA-II

NSGA-II with memetic local search repairs enabled.
Tests effectiveness of repair system.
"""

from configs.experiments.memetic import (
    EXPERIMENT_DESCRIPTION as _EXPERIMENT_DESCRIPTION,
)
from configs.experiments.memetic import EXPERIMENT_ID as _EXPERIMENT_ID
from configs.experiments.memetic import EXPERIMENT_NAME as _EXPERIMENT_NAME
from configs.experiments.memetic import KILLSWITCHES as _KILLSWITCHES
from configs.experiments.memetic import MemeticProdConfig, MemeticTestConfig
from configs.profiles import Profile

# Use dataclass configs
experiment_b_test = MemeticTestConfig()
experiment_b_prod = MemeticProdConfig()

# Experiment metadata (imported from dataclass module)
EXPERIMENT_ID = _EXPERIMENT_ID
EXPERIMENT_NAME = _EXPERIMENT_NAME
EXPERIMENT_DESCRIPTION = _EXPERIMENT_DESCRIPTION
KILLSWITCHES = _KILLSWITCHES


# Quick usage
def get_config(profile: Profile = Profile.TEST, **overrides):
    """Get config for Experiment B."""
    # Filter out None values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config: MemeticTestConfig | MemeticProdConfig
    if profile == Profile.TEST:
        config = MemeticTestConfig(**overrides)
    else:
        config = MemeticProdConfig(**overrides)
    return config.to_pydantic()


if __name__ == "__main__":
    test_cfg = experiment_b_test
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  repair={test_cfg.repair_enabled}")
    print(f"  memetic={test_cfg.repair_memetic_mode}")
    print(f"  rl={test_cfg.rl_enabled}")
