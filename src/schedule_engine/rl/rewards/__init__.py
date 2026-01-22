"""
Reward calculation modules for RL agent.

ENHANCEMENT #1: Multi-objective reward functions
- Scalar (weighted sum) - default
- Hypervolume indicator - Pareto-aware
- MOEA/D decomposition - multi-agent ensemble
- Adaptive preferences - dynamic weighting
"""

from schedule_engine.rl.rewards.adaptive_reward import AdaptiveReward
from schedule_engine.rl.rewards.decomposed_reward import DecomposedReward
from schedule_engine.rl.rewards.hypervolume_reward import HypervolumeReward
from schedule_engine.rl.rewards.scalar_reward import ScalarReward

__all__ = [
    "ScalarReward",
    "HypervolumeReward",
    "DecomposedReward",
    "AdaptiveReward",
]
