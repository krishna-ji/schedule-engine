"""
Heuristic Toolbox — plain-list registry (no decorators).

All 26 heuristics are declared in ``all_heuristics.py``.
This ``__init__`` re-exports the public helpers so callers can do::

    from schedule_engine.heuristics import get_all_heuristics, get_enabled_heuristics
"""

from __future__ import annotations

from schedule_engine.heuristics.all_heuristics import (
    CATEGORIES,
    HeuristicInfo,
    get_all_heuristics,
    get_enabled_heuristics,
    get_heuristic_by_name,
    get_heuristic_statistics_template,
    get_heuristics_by_category,
)

__all__ = [
    "CATEGORIES",
    "HeuristicInfo",
    "get_all_heuristics",
    "get_enabled_heuristics",
    "get_heuristic_by_name",
    "get_heuristic_statistics_template",
    "get_heuristics_by_category",
]
