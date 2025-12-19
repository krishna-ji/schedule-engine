"""
Experiment F: Individual Heuristic Testing

Test individual heuristics in isolation.
Enable/disable specific heuristics via config flags.
"""

from configs.experiments.heuristic_testing import (
    EXPERIMENT_DESCRIPTION as _EXPERIMENT_DESCRIPTION,
)
from configs.experiments.heuristic_testing import EXPERIMENT_ID as _EXPERIMENT_ID
from configs.experiments.heuristic_testing import EXPERIMENT_NAME as _EXPERIMENT_NAME
from configs.experiments.heuristic_testing import KILLSWITCHES as _KILLSWITCHES
from configs.experiments.heuristic_testing import (
    HeuristicTestingProdConfig,
    HeuristicTestingTestConfig,
)
from configs.profiles import Profile

# Use dataclass configs
experiment_f_test = HeuristicTestingTestConfig()
experiment_f_prod = HeuristicTestingProdConfig()

# Experiment metadata (imported from dataclass module)
EXPERIMENT_ID = _EXPERIMENT_ID
EXPERIMENT_NAME = _EXPERIMENT_NAME
EXPERIMENT_DESCRIPTION = _EXPERIMENT_DESCRIPTION
KILLSWITCHES = _KILLSWITCHES


# Quick usage
def get_config(profile: Profile = Profile.TEST, **overrides):
    """Get config for Experiment F."""
    if profile == Profile.TEST:
        config = experiment_f_test
    elif profile == Profile.PROD:
        config = experiment_f_prod
    else:
        raise ValueError(f"Unknown profile: {profile}")

    # Apply overrides if provided
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
