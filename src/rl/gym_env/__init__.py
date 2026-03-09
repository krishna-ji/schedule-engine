"""
Gymnasium environment for schedule optimization.

Wraps the GA scheduler as a reinforcement learning environment where:
- State: Population metrics, fitness stats, diversity, progress (39-D)
- Actions: 6 discrete actions (LLH pipeline configurations)
- Rewards: Fitness improvement + PBRS shaping + curriculum bonus
"""

# Legacy DEAP‑era imports removed.  The current Pymoo‑native environment
# lives in  src.rl.gym_env.pymoo_env  and the fast state encoder in
# src.rl.gym_env.fast_state_encoder .

__all__: list[str] = []
