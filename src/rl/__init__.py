"""
Reinforcement Learning integration module.

Provides RL-based heuristic selection for the GA scheduler using:
- Gymnasium environment wrapping the scheduling problem
- Stable-Baselines3 agents (PPO, DQN)

Components:
- gym_env: Gymnasium environment (state, action, reward)
- agents: PPO, DQN, random baseline agents
- training: Training loop and utilities
"""

__all__: list[str] = []
