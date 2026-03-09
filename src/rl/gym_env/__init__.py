"""
Gymnasium environment for schedule optimization.

Wraps the GA scheduler as a reinforcement learning environment where:
- State: Population metrics, fitness stats, diversity, progress (39-D)
- Actions: 6 discrete actions (LLH pipeline configurations)
- Rewards: Fitness improvement + PBRS shaping + curriculum bonus
"""

from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

__all__: list[str] = ["PymooHyperHeuristicEnv"]
