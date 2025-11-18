"""
Reward calculation modules for RL agent.

ENHANCEMENT #1: Multi-objective reward functions
- Scalar (weighted sum) - default
- Hypervolume indicator - Pareto-aware
- MOEA/D decomposition - multi-agent ensemble
- Adaptive preferences - dynamic weighting
"""

from src.rl.rewards.scalar_reward import ScalarReward
from src.rl.rewards.hypervolume_reward import HypervolumeReward
from src.rl.rewards.decomposed_reward import DecomposedReward
from src.rl.rewards.adaptive_reward import AdaptiveReward

__all__ = [
    "ScalarReward",
    "HypervolumeReward",
    "DecomposedReward",
    "AdaptiveReward",
]
