"""
Reinforcement Learning (RL) Hyper-Heuristic Framework.

This module implements the RL-based hyper-heuristic that learns to select
and apply the most effective optimization heuristics at each stage of the
schedule optimization process.

Phase 2 implementation components:
- Environment: Timetabling problem state representation
- State: Feature vector describing current schedule quality
- Actions: Heuristic operators (mutations, crossovers, LNS-IGLS, etc.)
- Reward: Fitness improvement + strategic bonuses
- Agents: Random baseline, Q-Learning, DQN

This is the next evolution beyond pure NSGA-II + IGLS + LNS-IGLS.
"""

__all__ = []
