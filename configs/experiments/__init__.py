"""
Experiment configurations using dataclass-based hierarchy.

Import experiment configs from this module:
- baseline: Pure NSGA-II (Mode A)
- memetic: NSGA-II + Memetic local search (Mode B)
- roundrobin: NSGA-II + Round-robin heuristics (Mode C)
- adaptive: NSGA-II + Adaptive heuristics (Mode D)
- rl_guided: NSGA-II + RL-guided control (Mode E)
"""

from .baseline import BaselineProdConfig, BaselineTestConfig
from .memetic import MemeticProdConfig, MemeticTestConfig

__all__ = [
    "BaselineTestConfig",
    "BaselineProdConfig",
    "MemeticTestConfig",
    "MemeticProdConfig",
]
