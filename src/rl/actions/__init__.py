"""Atomic vectorized RL actions for Pymoo hyper-heuristic control.

Each action is a standalone ``pymoo.core.repair.Repair`` subclass that
targets a *single* constraint class.  The RL agent selects which action
to inject into ``algorithm.mating.repair`` each generation.

Usage::

    from src.rl.actions.vectorized_ops import VECTORIZED_ACTION_SPACE

    operator_cls = VECTORIZED_ACTION_SPACE[action_id]
    operator = operator_cls(pkl_path)
"""

from src.rl.actions.vectorized_ops import VECTORIZED_ACTION_SPACE

__all__ = ["VECTORIZED_ACTION_SPACE"]
