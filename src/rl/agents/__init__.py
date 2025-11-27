"""
RL agent implementations and wrappers.

Provides pre-configured Stable-Baselines3 agents:
- PPO (Proximal Policy Optimization) - recommended
- DQN (Deep Q-Network) - discrete actions
- Random (baseline for comparison)

TIER 2 Enhancements:
- Specialist agents (ENHANCEMENT #4): repair vs optimization
- Agent coordinators for multi-agent systems
"""

from src.rl.agents.dqn_agent import create_dqn_agent
from src.rl.agents.ppo_agent import create_ppo_agent
from src.rl.agents.random_agent import RandomAgent
from src.rl.agents.specialist_agents import AgentCoordinator, SpecialistAgents

__all__ = [
    "create_ppo_agent",
    "create_dqn_agent",
    "RandomAgent",
    "SpecialistAgents",
    "AgentCoordinator",
]
