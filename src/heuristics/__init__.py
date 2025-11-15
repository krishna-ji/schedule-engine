"""
Heuristic Toolbox Module

Provides decorator-based registration for heuristic operators across five categories:
1. Construction: Build schedules greedily from scratch
2. Perturbation: Shake solutions to escape local optima
3. Improvement: Local search moves for refinement
4. Diversity: Maintain population diversity
5. Meta: High-level search strategies

Architecture:
- Decorator-based registry (like constraints/repair operators)
- Config-driven enable/disable with killswitches
- Priority ordering for execution control
- Unified interface for all heuristics

Usage:
    from src.heuristics import (
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

from src.heuristics.registry import (
    # Decorators
    construction_heuristic,
    perturbation_heuristic,
    improvement_heuristic,
    diversity_heuristic,
    meta_heuristic,
    # Registry access
    get_all_heuristics,
    get_enabled_heuristics,
    get_heuristic_by_name,
    list_all_heuristics,
    # Category access
    get_construction_heuristics,
    get_perturbation_heuristics,
    get_improvement_heuristics,
    get_diversity_heuristics,
    get_meta_heuristics,
)

# Import heuristic implementations to trigger decorator registration
from src.heuristics import construction
from src.heuristics import perturbation
from src.heuristics import improvement
from src.heuristics import diversity
from src.heuristics import meta

__all__ = [
    # Decorators
    "construction_heuristic",
    "perturbation_heuristic",
    "improvement_heuristic",
    "diversity_heuristic",
    "meta_heuristic",
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
]
