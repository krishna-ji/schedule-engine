"""
Experiment configuration modules.

Each module defines an experiment with:
- Dataclass configs (Test and Prod profiles)
- Killswitches (explicit feature flags)
- Metadata (ID, name, description)
- get_config() helper function
"""

# Import experiment modules
from . import (
    experiment_a_baseline,
    experiment_b_memetic,
    experiment_c_roundrobin,
    experiment_d_adaptive,
    experiment_e_rl_guided,
)

# Dataclass config instances (for backward compatibility)
from .experiment_a_baseline import experiment_a_test as experiment_a
from .experiment_b_memetic import experiment_b_test as experiment_b
from .experiment_c_roundrobin import experiment_c_test as experiment_c
from .experiment_d_adaptive import experiment_d_test as experiment_d
from .experiment_e_rl_guided import experiment_e_test as experiment_e

__all__ = [
    # Modules
    "experiment_a_baseline",
    "experiment_b_memetic",
    "experiment_c_roundrobin",
    "experiment_d_adaptive",
    "experiment_e_rl_guided",
    # Instances (default to test configs)
    "experiment_a",
    "experiment_b",
    "experiment_c",
    "experiment_d",
    "experiment_e",
]
