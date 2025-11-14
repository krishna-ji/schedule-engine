"""
Large Neighborhood Search (LNS) with CP-SAT integration.

This module provides LNS-based repair operators that use CP-SAT as a
subproblem solver to repair hard constraint violations in GA individuals.
"""

from src.lns.conflict_detection import find_hard_conflict_sessions
from src.lns.cp_repair import repair_with_cp_sat
from src.lns.lns_operator import lns_cp_repair

__all__ = [
    "find_hard_conflict_sessions",
    "repair_with_cp_sat",
    "lns_cp_repair",
]
