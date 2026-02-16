"""
Heuristic Toolbox — OOP and function-based registries.

Provides both:
1. OOP heuristic classes (new API) via ``heuristics.py``
2. Function-based HeuristicInfo (legacy API) via ``all_heuristics.py``

Usage (OOP):
    from src.ga.heuristics import ALL_HEURISTICS, build_heuristics

    # Use defaults
    for h in ALL_HEURISTICS:
        result = h.apply(individual, context)

    # Custom config
    heuristics = build_heuristics(lns_destroy_fraction=0.3)

Usage (Legacy):
    from src.ga.heuristics import get_all_heuristics, get_enabled_heuristics

    for h in get_enabled_heuristics():
        result = h.function(individual, context)
"""

from __future__ import annotations

# Legacy function-based API (all_heuristics.py)
from src.ga.heuristics.all_heuristics import (
    CATEGORIES,
    HeuristicInfo,
    get_all_heuristics,
    get_enabled_heuristics,
    get_heuristic_by_name,
    get_heuristic_statistics_template,
    get_heuristics_by_category,
)

# New OOP API (heuristics.py)
from src.ga.heuristics.heuristics import (
    ALL_HEURISTICS,
    CONSTRUCTION_HEURISTICS,
    DIVERSITY_HEURISTICS,
    ENABLED_HEURISTIC_NAMES,
    HEURISTIC_NAMES,
    IMPROVEMENT_HEURISTICS,
    META_HEURISTICS,
    PERTURBATION_HEURISTICS,
    REPAIR_HEURISTICS,
    FunctionHeuristic,
    Heuristic,
    HeuristicBase,
    build_heuristics,
    get_all_heuristic_objects,
    get_heuristic_by_name_oop,
    get_heuristics_by_category_oop,
)

__all__ = [
    # Legacy API
    "CATEGORIES",
    "HeuristicInfo",
    "get_all_heuristics",
    "get_enabled_heuristics",
    "get_heuristic_by_name",
    "get_heuristic_statistics_template",
    "get_heuristics_by_category",
    # OOP API
    "Heuristic",
    "HeuristicBase",
    "FunctionHeuristic",
    "ALL_HEURISTICS",
    "CONSTRUCTION_HEURISTICS",
    "PERTURBATION_HEURISTICS",
    "IMPROVEMENT_HEURISTICS",
    "DIVERSITY_HEURISTICS",
    "META_HEURISTICS",
    "REPAIR_HEURISTICS",
    "HEURISTIC_NAMES",
    "ENABLED_HEURISTIC_NAMES",
    "build_heuristics",
    "get_all_heuristic_objects",
    "get_heuristics_by_category_oop",
    "get_heuristic_by_name_oop",
]
