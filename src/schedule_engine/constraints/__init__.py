"""
Constraint evaluation system for UCTP.

All constraints are always enabled. No registry needed.
"""

from __future__ import annotations

from schedule_engine.constraints.all_constraints import (
    HARD_CONSTRAINTS,
    SOFT_CONSTRAINTS,
    evaluate_all,
    evaluate_hard_constraints,
    evaluate_soft_constraints,
    get_all_hard_constraints,
    get_all_soft_constraints,
    get_hard_constraints,
    get_soft_constraints,
)

__all__ = [
    "HARD_CONSTRAINTS",
    "SOFT_CONSTRAINTS",
    "evaluate_all",
    "evaluate_hard_constraints",
    "evaluate_soft_constraints",
    "get_all_hard_constraints",
    "get_all_soft_constraints",
    "get_hard_constraints",
    "get_soft_constraints",
]
