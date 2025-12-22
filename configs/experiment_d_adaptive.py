"""
Experiment D: Adaptive Heuristic Selection

NSGA-II with performance-based adaptive heuristic priority.
Tests effectiveness of adaptive selection mechanism.
"""

from configs.experiments.adaptive import (
    EXPERIMENT_DESCRIPTION as _EXPERIMENT_DESCRIPTION,
)
from configs.experiments.adaptive import EXPERIMENT_ID as _EXPERIMENT_ID
from configs.experiments.adaptive import EXPERIMENT_NAME as _EXPERIMENT_NAME
from configs.experiments.adaptive import KILLSWITCHES as _KILLSWITCHES
from configs.experiments.adaptive import AdaptiveProdConfig, AdaptiveTestConfig
from configs.profiles import Profile

# Use dataclass configs
experiment_d_test = AdaptiveTestConfig()
experiment_d_prod = AdaptiveProdConfig()

# Experiment metadata (imported from dataclass module)
EXPERIMENT_ID = _EXPERIMENT_ID
EXPERIMENT_NAME = _EXPERIMENT_NAME
EXPERIMENT_DESCRIPTION = _EXPERIMENT_DESCRIPTION
KILLSWITCHES = _KILLSWITCHES


# Quick usage
def get_config(profile: Profile = Profile.TEST, **overrides):
    """Get config for Experiment D."""
    # Filter out None values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config: AdaptiveTestConfig | AdaptiveProdConfig
    if profile == Profile.TEST:
        config = AdaptiveTestConfig(**overrides)
    else:
        config = AdaptiveProdConfig(**overrides)
    return config.to_pydantic()


if __name__ == "__main__":
    test_cfg = experiment_d_test
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  adaptive_priority={test_cfg.heuristics_adaptive_priority_enabled}")
    print(f"  adaptive_prob={test_cfg.ga_use_adaptive_probabilities}")
    print(f"  hypermutation={test_cfg.enhancements_hypermutation_enabled}")
