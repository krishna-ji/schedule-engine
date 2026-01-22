"""
Heuristic Toolbox Module

Provides decorator-based registration for heuristic operators across six categories:
1. Construction: Build schedules greedily from scratch
2. Perturbation: Shake solutions to escape local optima
3. Improvement: Local search moves for refinement
4. Diversity: Maintain population diversity
5. Meta: High-level search strategies
6. Repair: Fix constraint violations

Architecture:
- Decorator-based registry (like constraints/repair operators)
- Config-driven enable/disable with killswitches
- Priority ordering for execution control
- Unified interface for all heuristics

Usage:
    from schedule_engine.heuristics import (
        construction_heuristic,
        perturbation_heuristic,
        improvement_heuristic,
        diversity_heuristic,
        meta_heuristic
    )

    # Register a construction heuristic
    @construction_heuristic(
        name="largest_degree_first",
        description="Schedule most constrained courses first",
        priority=1
    )
    def largest_degree_first(context):
        # implementation
        return individual
"""

from __future__ import annotations

# Import heuristic implementations to trigger decorator registration
from schedule_engine.heuristics import (
    construction,
    diversity,
    improvement,
    meta,
    perturbation,
    repair,
)
from schedule_engine.heuristics.registry import (  # Decorators; Registry access; Category access
    construction_heuristic,
    diversity_heuristic,
    get_all_heuristics,
    get_construction_heuristics,
    get_diversity_heuristics,
    get_enabled_heuristics,
    get_heuristic_by_name,
    get_improvement_heuristics,
    get_meta_heuristics,
    get_perturbation_heuristics,
    get_repair_heuristics,
    improvement_heuristic,
    list_all_heuristics,
    meta_heuristic,
    perturbation_heuristic,
    repair_heuristic,
)

__all__ = [
    # Decorators
    "construction_heuristic",
    "perturbation_heuristic",
    "improvement_heuristic",
    "diversity_heuristic",
    "meta_heuristic",
    "repair_heuristic",
    # Registry functions
    "get_all_heuristics",
    "get_enabled_heuristics",
    "get_heuristic_by_name",
    "list_all_heuristics",
    "get_construction_heuristics",
    "get_perturbation_heuristics",
    "get_improvement_heuristics",
    "get_diversity_heuristics",
    "get_meta_heuristics",
    "get_repair_heuristics",
    "get_diversity_heuristics",
    "get_meta_heuristics",
]
