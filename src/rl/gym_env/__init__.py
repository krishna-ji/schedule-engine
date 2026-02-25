"""
Gymnasium environment for schedule optimization.

Wraps the GA scheduler as a reinforcement learning environment where:
- State: Population metrics, fitness stats, diversity, progress
- Actions: 20 discrete actions (19 heuristics + no-op)
- Rewards: Fitness improvement + diversity bonus - time penalty
"""

# Legacy DEAP‑era imports removed.  The current Pymoo‑native environment
# lives in  src.rl.gym_env.pymoo_env  and the fast state encoder in
# src.rl.gym_env.fast_state_encoder .

__all__: list[str] = []
