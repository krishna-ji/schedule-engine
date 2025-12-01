"""
Experiment E: RL-Guided Hyper-Heuristic

Full NSGA-II stack with RL controlling heuristic selection.
Tests effectiveness of reinforcement learning guidance.
"""

from configs.experiments.rl_guided import (
    EXPERIMENT_DESCRIPTION as _EXPERIMENT_DESCRIPTION,
)
from configs.experiments.rl_guided import EXPERIMENT_ID as _EXPERIMENT_ID
from configs.experiments.rl_guided import EXPERIMENT_NAME as _EXPERIMENT_NAME
from configs.experiments.rl_guided import KILLSWITCHES as _KILLSWITCHES
from configs.experiments.rl_guided import RlGuidedProdConfig, RlGuidedTestConfig
from configs.profiles import Profile

# Use dataclass configs
experiment_e_test = RlGuidedTestConfig()
experiment_e_prod = RlGuidedProdConfig()

# Experiment metadata (imported from dataclass module)
EXPERIMENT_ID = _EXPERIMENT_ID
EXPERIMENT_NAME = _EXPERIMENT_NAME
EXPERIMENT_DESCRIPTION = _EXPERIMENT_DESCRIPTION
KILLSWITCHES = _KILLSWITCHES


# Quick usage
def get_config(profile: Profile = Profile.TEST, **overrides):
    """Get config for Experiment E."""
    # Filter out None values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if profile == Profile.TEST:
        config = RlGuidedTestConfig(**overrides)
    else:
        config = RlGuidedProdConfig(**overrides)
    return config.to_pydantic()


if __name__ == "__main__":
    test_cfg = experiment_e_test
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  rl_enabled={test_cfg.rl_enabled}")
    print(f"  rl_mode={test_cfg.rl_mode}")
    print(f"  lns={test_cfg.lns_enabled}")
    print(f"  LNS enabled: {test_cfg.lns_enabled}")
    print(f"  Memetic mode: {test_cfg.repair_memetic_mode}")
