"""
All Heuristic Operators — flat list registry.

No decorators, no global mutable state.  Every heuristic is a plain function;
this module collects them into typed lists that the scheduler and RL agent
iterate over.

Pattern mirrors ``constraints/all_constraints.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Metadata


HeuristicFunc = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class HeuristicInfo:
    """Lightweight descriptor for a single heuristic operator."""

    name: str
    function: HeuristicFunc
    description: str
    category: str  # "construction" | "perturbation" | "improvement" | "diversity" | "meta" | "repair"
    priority: int  # lower = higher priority (executed first)
    enabled_by_default: bool = True
    requires_population: bool = False
    modifies_individual: bool = False


# Lazy imports — each function lives in its own module already.
# We import them here so the list is the single source of truth.


def _lazy_construction():
    from schedule_engine.heuristics.construction import (
        earliest_deadline_first,
        largest_degree_first,
        most_constrained_first,
    )

    return [
        HeuristicInfo(
            name="largest_degree_first",
            function=largest_degree_first,
            description="Schedule courses with most conflicts/constraints first (graph coloring heuristic)",
            category="construction",
            priority=1,
        ),
        HeuristicInfo(
            name="most_constrained_first",
            function=most_constrained_first,
            description="Schedule sessions with fewest valid time slots first (minimum remaining values)",
            category="construction",
            priority=2,
        ),
        HeuristicInfo(
            name="earliest_deadline_first",
            function=earliest_deadline_first,
            description="Schedule courses with more sessions per week first (higher frequency = higher priority)",
            category="construction",
            priority=3,
        ),
    ]


def _lazy_perturbation():
    from schedule_engine.heuristics.perturbation import (
        instructor_reassign,
        multi_perturbation,
        random_swap,
        room_shuffle,
        temporal_shift,
    )

    return [
        HeuristicInfo(
            name="random_swap",
            function=random_swap,
            description="Randomly swap time slots or rooms between two compatible sessions",
            category="perturbation",
            priority=1,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="temporal_shift",
            function=temporal_shift,
            description="Shift sessions forward or backward in time by delta quanta",
            category="perturbation",
            priority=2,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="room_shuffle",
            function=room_shuffle,
            description="Randomly reassign rooms to sessions while maintaining compatibility",
            category="perturbation",
            priority=3,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="instructor_reassign",
            function=instructor_reassign,
            description="Reassign instructors to other qualified instructors for courses",
            category="perturbation",
            priority=4,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="multi_perturbation",
            function=multi_perturbation,
            description="Apply multiple perturbation operators in sequence for stronger diversification",
            category="perturbation",
            priority=5,
            enabled_by_default=False,
            modifies_individual=True,
        ),
    ]


def _lazy_improvement():
    from schedule_engine.heuristics.improvement import (
        ejection_chain,
        kempe_chain,
        variable_depth_search,
    )

    return [
        HeuristicInfo(
            name="kempe_chain",
            function=kempe_chain,
            description="Apply Kempe chain moves to resolve time conflicts (graph coloring heuristic)",
            category="improvement",
            priority=1,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="ejection_chain",
            function=ejection_chain,
            description="Apply ejection chain moves with cascading reassignments",
            category="improvement",
            priority=2,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="variable_depth_search",
            function=variable_depth_search,
            description="Multi-move lookahead search with backtracking",
            category="improvement",
            priority=3,
            modifies_individual=True,
        ),
    ]


def _lazy_diversity():
    from schedule_engine.heuristics.diversity import (
        adaptive_diversity_maintenance,
        crowding_mutation,
        distance_preserving_crossover,
        niching_selection,
    )

    return [
        HeuristicInfo(
            name="distance_preserving_crossover",
            function=distance_preserving_crossover,
            description="Crossover operator that maintains phenotypic distance between parents",
            category="diversity",
            priority=1,
            requires_population=True,
        ),
        HeuristicInfo(
            name="crowding_mutation",
            function=crowding_mutation,
            description="Mutation that favors less-explored regions of search space",
            category="diversity",
            priority=2,
            requires_population=True,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="niching_selection",
            function=niching_selection,
            description="Selection operator that promotes diverse individuals (fitness sharing)",
            category="diversity",
            priority=3,
            requires_population=True,
        ),
        HeuristicInfo(
            name="adaptive_diversity_maintenance",
            function=adaptive_diversity_maintenance,
            description="Dynamically adjust diversity based on convergence state",
            category="diversity",
            priority=4,
            enabled_by_default=False,
            requires_population=True,
            modifies_individual=True,
        ),
    ]


def _lazy_meta():
    from schedule_engine.heuristics.meta import (
        adaptive_large_neighborhood,
        guided_local_search,
        iterated_local_search,
        variable_neighborhood_descent,
    )

    return [
        HeuristicInfo(
            name="variable_neighborhood_descent",
            function=variable_neighborhood_descent,
            description="Systematically explore multiple neighborhoods until local optimum",
            category="meta",
            priority=1,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="iterated_local_search",
            function=iterated_local_search,
            description="Alternate between perturbation and local search for global optimization",
            category="meta",
            priority=2,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="adaptive_large_neighborhood",
            function=adaptive_large_neighborhood,
            description="Adaptive destroy-repair with dynamic neighborhood sizing",
            category="meta",
            priority=3,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="guided_local_search",
            function=guided_local_search,
            description="Local search guided by dynamic penalties on solution features",
            category="meta",
            priority=4,
            enabled_by_default=False,
            modifies_individual=True,
        ),
    ]


def _lazy_repair():
    from schedule_engine.heuristics.repair.break_repair import repair_break_placement
    from schedule_engine.heuristics.repair.exhaustive_repair import exhaustive_repair
    from schedule_engine.heuristics.repair.greedy_repair import greedy_repair
    from schedule_engine.heuristics.repair.igls_repair import igls_repair
    from schedule_engine.heuristics.repair.lns_repair import lns_repair
    from schedule_engine.heuristics.repair.memetic_repair import memetic_repair
    from schedule_engine.heuristics.repair.selective_repair import selective_repair

    return [
        HeuristicInfo(
            name="igls_repair",
            function=igls_repair,
            description="Iterative Greedy Local Search repair for constraint violations",
            category="repair",
            priority=1,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="greedy_repair",
            function=greedy_repair,
            description="Fast greedy repair with first-improving moves",
            category="repair",
            priority=2,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="selective_repair",
            function=selective_repair,
            description="Selective repair targeting only violated genes for efficiency",
            category="repair",
            priority=3,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="lns_repair",
            function=lns_repair,
            description="Large Neighborhood Search repair with IGLS subproblem solving",
            category="repair",
            priority=4,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="exhaustive_repair",
            function=exhaustive_repair,
            description="Exhaustive steepest-descent repair (very intensive)",
            category="repair",
            priority=5,
            enabled_by_default=False,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="memetic_repair",
            function=memetic_repair,
            description="Memetic repair (intensive IGLS on elite individuals)",
            category="repair",
            priority=6,
            enabled_by_default=False,
            requires_population=True,
            modifies_individual=True,
        ),
        HeuristicInfo(
            name="repair_break_placement",
            function=repair_break_placement,
            description="Moves sessions to ensure break windows are respected",
            category="repair",
            priority=9,
            modifies_individual=True,
        ),
    ]


# Public API


# Module-level cache so lazy loaders run at most once.
_cache: dict[str, list[HeuristicInfo]] | None = None

CATEGORIES = (
    "construction",
    "perturbation",
    "improvement",
    "diversity",
    "meta",
    "repair",
)


def _ensure_loaded() -> dict[str, list[HeuristicInfo]]:
    global _cache
    if _cache is None:
        _cache = {
            "construction": _lazy_construction(),
            "perturbation": _lazy_perturbation(),
            "improvement": _lazy_improvement(),
            "diversity": _lazy_diversity(),
            "meta": _lazy_meta(),
            "repair": _lazy_repair(),
        }
    return _cache


def get_all_heuristics() -> list[HeuristicInfo]:
    """Return flat list of every registered heuristic, sorted by (category, priority)."""
    by_cat = _ensure_loaded()
    out: list[HeuristicInfo] = []
    for cat in CATEGORIES:
        out.extend(by_cat.get(cat, []))
    return out


def get_heuristics_by_category(category: str) -> list[HeuristicInfo]:
    """Return heuristics for a single category, sorted by priority."""
    return list(_ensure_loaded().get(category, []))


def get_heuristic_by_name(name: str) -> HeuristicInfo | None:
    """Lookup a single heuristic by name (linear scan — 26 items, fine)."""
    for h in get_all_heuristics():
        if h.name == name:
            return h
    return None


def get_enabled_heuristics(category: str | None = None) -> dict[str, HeuristicInfo]:
    """
    Return enabled heuristics honouring ``config.heuristics`` overrides.

    Falls back to ``enabled_by_default`` when no config section exists.
    Returns dict[name → HeuristicInfo] sorted by priority (ascending).
    """
    from schedule_engine.config import get_config_or_default

    if category:
        candidates = get_heuristics_by_category(category)
    else:
        candidates = get_all_heuristics()

    config = get_config_or_default()
    heuristics_config = getattr(config, "heuristics", None)

    if not heuristics_config:
        # No config → use defaults
        return {h.name: h for h in candidates if h.enabled_by_default}

    # Master killswitch
    if not getattr(heuristics_config, "master_enabled", True):
        return {}

    enabled: dict[str, HeuristicInfo] = {}
    for h in candidates:
        cat_cfg = getattr(heuristics_config, h.category, None)
        cat_provided = cat_cfg is not None and bool(cat_cfg)

        if cat_provided:
            assert cat_cfg is not None
            if h.name not in cat_cfg:
                continue
            h_cfg = cat_cfg.get(h.name) or {}
        else:
            h_cfg = {}

        is_enabled = h_cfg.get("enabled", h.enabled_by_default)
        if not is_enabled:
            continue

        priority = h_cfg.get("priority", h.priority)
        if priority != h.priority:
            # Config overrode priority → create new info with updated priority
            from dataclasses import replace

            h = replace(h, priority=priority)

        enabled[h.name] = h

    # Sort by priority
    return dict(sorted(enabled.items(), key=lambda kv: kv[1].priority))


def get_heuristic_statistics_template() -> dict[str, int]:
    """Return a zeroed-out counter dict for telemetry."""
    stats: dict[str, int] = {"total_applications": 0, "total_improvements": 0}
    for h in get_all_heuristics():
        stats[f"{h.name}_applications"] = 0
        stats[f"{h.name}_improvements"] = 0
    return stats
