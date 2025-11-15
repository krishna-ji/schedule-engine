"""
Large Neighborhood Search (LNS) with IGLS integration.

This module provides LNS-based repair operators that use IGLS (Iterated Guided Local Search)
as a subproblem solver to repair hard constraint violations in GA individuals.
"""

from src.lns.conflict_detection import find_hard_conflict_sessions
from src.lns.lns_operator import lns_igls_repair

__all__ = [
    "find_hard_conflict_sessions",
    "lns_igls_repair",
]
