"""
Experiment E: RL-Guided Hyper-Heuristic

Deploy trained RL agents to control heuristic selection within NSGA-II.
Full-featured configuration with RL killswitch enabled.
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


def get_config(profile: Profile = Profile.TEST, **overrides):
    """Get config for Experiment E (RL-guided)."""
    # Filter out None overrides to avoid clobbering defaults
    filtered_overrides = {k: v for k, v in overrides.items() if v is not None}

    config: RlGuidedTestConfig | RlGuidedProdConfig
    if profile == Profile.TEST:
        config = RlGuidedTestConfig(**filtered_overrides)
    elif profile == Profile.PROD:
        config = RlGuidedProdConfig(**filtered_overrides)
    else:
        raise ValueError(f"Unknown profile: {profile}")

    return config.to_pydantic()


if __name__ == "__main__":
    test_cfg = experiment_e_test
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  rl_enabled={test_cfg.rl_enabled}, mode={test_cfg.rl_mode}")
    print(f"  heuristics_master={test_cfg.heuristics_master_enabled}")
