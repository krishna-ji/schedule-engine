"""
Reinforcement Learning integration module.

Provides RL-based heuristic selection for the GA scheduler using:
- Gymnasium environment wrapping the scheduling problem
- Stable-Baselines3 agents (PPO, DQN)
- Hybrid control modes (RL-primary, RL-fallback, RL-assisted)

Phase 2 Components:
- gym_env: Gymnasium environment (state, action, reward)
- agents: PPO, DQN, random baseline agents
- training: Training loop, curriculum, hyperparameter tuning
- deployment: Model loading and inference for production
- hybrid: Hybrid controller for GA integration
- evaluation: Baseline comparison and metrics
- visualization: Training curves and performance plots
"""

from src.rl.gym_env.schedule_env import ScheduleEnv
from src.rl.hybrid.hybrid_controller import HybridController

__all__ = ["ScheduleEnv", "HybridController"]
