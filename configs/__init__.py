"""
Experiment configuration modules.

Each module defines an experiment with:
- Blueprint instance (algorithmic configuration)
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

# Experiment instances
from .experiment_a_baseline import experiment_a
from .experiment_b_memetic import experiment_b
from .experiment_c_roundrobin import experiment_c
from .experiment_d_adaptive import experiment_d
from .experiment_e_rl_guided import experiment_e

__all__ = [
    # Modules
    "experiment_a_baseline",
    "experiment_b_memetic",
    "experiment_c_roundrobin",
    "experiment_d_adaptive",
    "experiment_e_rl_guided",
    # Instances
    "experiment_a",
    "experiment_b",
    "experiment_c",
    "experiment_d",
    "experiment_e",
]
