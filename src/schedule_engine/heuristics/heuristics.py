"""Self-contained OOP heuristic classes with configurable parameters.

Pattern mirrors ``constraints/constraints.py`` for consistency.

Each heuristic class:
- Has `name`, `category`, `priority` class attributes
- Has `__init__` for configurable parameters
- Has `apply(individual, context)` method

Usage:
    # Use default registries
    from schedule_engine.heuristics import ALL_HEURISTICS

    for h in ALL_HEURISTICS:
        if h.category == "repair":
            result = h.apply(individual, context)

    # Custom configuration
    from schedule_engine.heuristics import build_heuristics

    heuristics = build_heuristics(
        lns_destroy_fraction=0.3,
        igls_max_iterations=150,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.types import SchedulingContext

# =============================================================================
# PROTOCOL
# =============================================================================


@runtime_checkable
class Heuristic(Protocol):
    """Protocol for heuristic operators.

    All heuristics must implement:
    - name: Unique string identifier
    - category: One of "construction", "perturbation", "improvement",
                "diversity", "meta", "repair"
    - priority: Lower = higher priority
    - enabled: Whether enabled by default
    - apply: Execute the heuristic
    """

    name: str
    category: Literal[
        "construction", "perturbation", "improvement", "diversity", "meta", "repair"
    ]
    priority: int
    enabled: bool

    def apply(
        self,
        individual: list[SessionGene],
        context: SchedulingContext,
        **kwargs: Any,
    ) -> int:
        """Apply heuristic and return modification count."""
        ...


# =============================================================================
# BASE CLASS
# =============================================================================


@dataclass
class HeuristicBase:
    """Base class providing common heuristic functionality."""

    # Subclasses set these as class attributes
    name: str = ""
    category: str = "repair"
    priority: int = 99
    enabled: bool = True
    description: str = ""
    requires_population: bool = False
    modifies_individual: bool = True

    def apply(
        self,
        individual: list[SessionGene],
        context: SchedulingContext,
        **kwargs: Any,
    ) -> int:
        """Apply heuristic. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement apply()")


# =============================================================================
# FUNCTION ADAPTER
# =============================================================================


class FunctionHeuristic:
    """Adapter that wraps an existing function as a Heuristic object.

    This allows the existing function-based heuristics to work with
    the new OOP interface without rewriting them.
    """

    def __init__(
        self,
        name: str,
        function: Callable[..., int],
        category: str,
        priority: int,
        enabled: bool = True,
        description: str = "",
        requires_population: bool = False,
        modifies_individual: bool = True,
        **default_kwargs: Any,
    ):
        self.name = name
        self._function = function
        self.category = category
        self.priority = priority
        self.enabled = enabled
        self.description = description
        self.requires_population = requires_population
        self.modifies_individual = modifies_individual
        self._default_kwargs = default_kwargs

    def apply(
        self,
        individual: list[SessionGene],
        context: SchedulingContext,
        **kwargs: Any,
    ) -> int:
        """Apply the wrapped function."""
        # Merge default kwargs with runtime kwargs
        merged = {**self._default_kwargs, **kwargs}
        return self._function(individual, context, **merged)

    def __repr__(self) -> str:
        return f"FunctionHeuristic({self.name!r}, category={self.category!r})"


# =============================================================================
# HEURISTIC FACTORIES
# =============================================================================


def _create_construction_heuristics(**params: Any) -> list[Heuristic]:
    """Create construction heuristics with optional parameters."""
    from schedule_engine.heuristics.construction import (
        earliest_deadline_first,
        largest_degree_first,
        most_constrained_first,
    )

    return [
        FunctionHeuristic(
            name="largest_degree_first",
            function=largest_degree_first,
            category="construction",
            priority=1,
            description="Schedule courses with most conflicts/constraints first",
        ),
        FunctionHeuristic(
            name="most_constrained_first",
            function=most_constrained_first,
            category="construction",
            priority=2,
            description="Schedule sessions with fewest valid time slots first",
        ),
        FunctionHeuristic(
            name="earliest_deadline_first",
            function=earliest_deadline_first,
            category="construction",
            priority=3,
            description="Schedule courses with more sessions per week first",
        ),
    ]


def _create_perturbation_heuristics(**params: Any) -> list[Heuristic]:
    """Create perturbation heuristics with optional parameters."""
    from schedule_engine.heuristics.perturbation import (
        instructor_reassign,
        multi_perturbation,
        random_swap,
        room_shuffle,
        temporal_shift,
    )

    # Extract configurable params with defaults
    swap_num_swaps = params.get("swap_num_swaps", 1)
    temporal_shift_delta = params.get("temporal_shift_delta", 3)

    return [
        FunctionHeuristic(
            name="random_swap",
            function=random_swap,
            category="perturbation",
            priority=1,
            description="Randomly swap time slots or rooms between two sessions",
            modifies_individual=True,
            num_swaps=swap_num_swaps,
        ),
        FunctionHeuristic(
            name="temporal_shift",
            function=temporal_shift,
            category="perturbation",
            priority=2,
            description="Shift sessions forward or backward in time",
            modifies_individual=True,
            delta=temporal_shift_delta,
        ),
        FunctionHeuristic(
            name="room_shuffle",
            function=room_shuffle,
            category="perturbation",
            priority=3,
            description="Randomly reassign rooms to sessions",
            modifies_individual=True,
        ),
        FunctionHeuristic(
            name="instructor_reassign",
            function=instructor_reassign,
            category="perturbation",
            priority=4,
            description="Reassign instructors to other qualified instructors",
            modifies_individual=True,
        ),
        FunctionHeuristic(
            name="multi_perturbation",
            function=multi_perturbation,
            category="perturbation",
            priority=5,
            enabled=False,
            description="Apply multiple perturbation operators in sequence",
            modifies_individual=True,
        ),
    ]


def _create_improvement_heuristics(**params: Any) -> list[Heuristic]:
    """Create improvement heuristics with optional parameters."""
    from schedule_engine.heuristics.improvement import (
        ejection_chain,
        kempe_chain,
        variable_depth_search,
    )

    # Extract configurable params
    kempe_max_chain_length = params.get("kempe_max_chain_length", 10)
    vds_max_depth = params.get("vds_max_depth", 5)

    return [
        FunctionHeuristic(
            name="kempe_chain",
            function=kempe_chain,
            category="improvement",
            priority=1,
            description="Apply Kempe chain moves to resolve time conflicts",
            modifies_individual=True,
            max_chain_length=kempe_max_chain_length,
        ),
        FunctionHeuristic(
            name="ejection_chain",
            function=ejection_chain,
            category="improvement",
            priority=2,
            description="Apply ejection chain moves with cascading reassignments",
            modifies_individual=True,
        ),
        FunctionHeuristic(
            name="variable_depth_search",
            function=variable_depth_search,
            category="improvement",
            priority=3,
            description="Multi-move lookahead search with backtracking",
            modifies_individual=True,
            max_depth=vds_max_depth,
        ),
    ]


def _create_diversity_heuristics(**params: Any) -> list[Heuristic]:
    """Create diversity heuristics with optional parameters."""
    from schedule_engine.heuristics.diversity import (
        adaptive_diversity_maintenance,
        crowding_mutation,
        distance_preserving_crossover,
        niching_selection,
    )

    return [
        FunctionHeuristic(
            name="distance_preserving_crossover",
            function=distance_preserving_crossover,
            category="diversity",
            priority=1,
            description="Crossover that maintains phenotypic distance",
            requires_population=True,
        ),
        FunctionHeuristic(
            name="crowding_mutation",
            function=crowding_mutation,
            category="diversity",
            priority=2,
            description="Mutation favoring less-explored regions",
            requires_population=True,
            modifies_individual=True,
        ),
        FunctionHeuristic(
            name="niching_selection",
            function=niching_selection,
            category="diversity",
            priority=3,
            description="Selection promoting diverse individuals",
            requires_population=True,
        ),
        FunctionHeuristic(
            name="adaptive_diversity_maintenance",
            function=adaptive_diversity_maintenance,
            category="diversity",
            priority=4,
            enabled=False,
            description="Dynamically adjust diversity based on convergence",
            requires_population=True,
            modifies_individual=True,
        ),
    ]


def _create_meta_heuristics(**params: Any) -> list[Heuristic]:
    """Create meta heuristics with optional parameters."""
    from schedule_engine.heuristics.meta import (
        adaptive_large_neighborhood,
        guided_local_search,
        iterated_local_search,
        variable_neighborhood_descent,
    )

    # Extract configurable params
    ils_perturbation_strength = params.get("ils_perturbation_strength", 3)
    aln_destroy_fraction = params.get("aln_destroy_fraction", 0.2)

    return [
        FunctionHeuristic(
            name="variable_neighborhood_descent",
            function=variable_neighborhood_descent,
            category="meta",
            priority=1,
            description="Systematically explore multiple neighborhoods",
            modifies_individual=True,
        ),
        FunctionHeuristic(
            name="iterated_local_search",
            function=iterated_local_search,
            category="meta",
            priority=2,
            description="Alternate between perturbation and local search",
            modifies_individual=True,
            perturbation_strength=ils_perturbation_strength,
        ),
        FunctionHeuristic(
            name="adaptive_large_neighborhood",
            function=adaptive_large_neighborhood,
            category="meta",
            priority=3,
            description="Adaptive destroy-repair with dynamic sizing",
            modifies_individual=True,
            destroy_fraction=aln_destroy_fraction,
        ),
        FunctionHeuristic(
            name="guided_local_search",
            function=guided_local_search,
            category="meta",
            priority=4,
            enabled=False,
            description="Local search guided by dynamic penalties",
            modifies_individual=True,
        ),
    ]


def _create_repair_heuristics(**params: Any) -> list[Heuristic]:
    """Create repair heuristics with optional parameters."""
    from schedule_engine.heuristics.repair.break_repair import repair_break_placement
    from schedule_engine.heuristics.repair.exhaustive_repair import exhaustive_repair
    from schedule_engine.heuristics.repair.greedy_repair import greedy_repair
    from schedule_engine.heuristics.repair.igls_repair import igls_repair
    from schedule_engine.heuristics.repair.lns_repair import lns_repair
    from schedule_engine.heuristics.repair.memetic_repair import memetic_repair
    from schedule_engine.heuristics.repair.selective_repair import selective_repair

    # Extract configurable params
    igls_max_iterations = params.get("igls_max_iterations", 100)
    lns_destroy_fraction = params.get("lns_destroy_fraction", 0.2)
    lns_max_iterations = params.get("lns_max_iterations", 50)
    greedy_max_moves = params.get("greedy_max_moves", 20)

    return [
        FunctionHeuristic(
            name="igls_repair",
            function=igls_repair,
            category="repair",
            priority=1,
            description="Iterative Greedy Local Search repair",
            modifies_individual=True,
            max_iterations=igls_max_iterations,
        ),
        FunctionHeuristic(
            name="greedy_repair",
            function=greedy_repair,
            category="repair",
            priority=2,
            description="Fast greedy repair with first-improving moves",
            modifies_individual=True,
            max_moves=greedy_max_moves,
        ),
        FunctionHeuristic(
            name="selective_repair",
            function=selective_repair,
            category="repair",
            priority=3,
            description="Selective repair targeting only violated genes",
            modifies_individual=True,
        ),
        FunctionHeuristic(
            name="lns_repair",
            function=lns_repair,
            category="repair",
            priority=4,
            description="Large Neighborhood Search repair",
            modifies_individual=True,
            destroy_fraction=lns_destroy_fraction,
            max_iterations=lns_max_iterations,
        ),
        FunctionHeuristic(
            name="exhaustive_repair",
            function=exhaustive_repair,
            category="repair",
            priority=5,
            enabled=False,
            description="Exhaustive steepest-descent repair",
            modifies_individual=True,
        ),
        FunctionHeuristic(
            name="memetic_repair",
            function=memetic_repair,
            category="repair",
            priority=6,
            enabled=False,
            description="Memetic repair on elite individuals",
            requires_population=True,
            modifies_individual=True,
        ),
        FunctionHeuristic(
            name="repair_break_placement",
            function=repair_break_placement,
            category="repair",
            priority=9,
            description="Move sessions to ensure break windows",
            modifies_individual=True,
        ),
    ]


# =============================================================================
# FACTORY
# =============================================================================


def build_heuristics(
    # Per-category enable/disable
    enable_construction: bool = True,
    enable_perturbation: bool = True,
    enable_improvement: bool = True,
    enable_diversity: bool = True,
    enable_meta: bool = True,
    enable_repair: bool = True,
    # Perturbation params
    swap_num_swaps: int = 1,
    temporal_shift_delta: int = 3,
    # Improvement params
    kempe_max_chain_length: int = 10,
    vds_max_depth: int = 5,
    # Meta params
    ils_perturbation_strength: int = 3,
    aln_destroy_fraction: float = 0.2,
    # Repair params
    igls_max_iterations: int = 100,
    lns_destroy_fraction: float = 0.2,
    lns_max_iterations: int = 50,
    greedy_max_moves: int = 20,
) -> list[Heuristic]:
    """Build heuristic list with custom configuration.

    Args:
        enable_*: Enable/disable entire category
        swap_num_swaps: Number of swaps for random_swap
        temporal_shift_delta: Max quanta to shift in temporal_shift
        kempe_max_chain_length: Max chain length for Kempe moves
        vds_max_depth: Search depth for variable_depth_search
        ils_perturbation_strength: Perturbation strength for ILS
        aln_destroy_fraction: Fraction to destroy in adaptive LNS
        igls_max_iterations: Max iterations for IGLS repair
        lns_destroy_fraction: Fraction to destroy in LNS repair
        lns_max_iterations: Max iterations for LNS repair
        greedy_max_moves: Max moves for greedy repair

    Returns:
        List of configured Heuristic objects
    """
    params = {
        "swap_num_swaps": swap_num_swaps,
        "temporal_shift_delta": temporal_shift_delta,
        "kempe_max_chain_length": kempe_max_chain_length,
        "vds_max_depth": vds_max_depth,
        "ils_perturbation_strength": ils_perturbation_strength,
        "aln_destroy_fraction": aln_destroy_fraction,
        "igls_max_iterations": igls_max_iterations,
        "lns_destroy_fraction": lns_destroy_fraction,
        "lns_max_iterations": lns_max_iterations,
        "greedy_max_moves": greedy_max_moves,
    }

    heuristics: list[Heuristic] = []

    if enable_construction:
        heuristics.extend(_create_construction_heuristics(**params))
    if enable_perturbation:
        heuristics.extend(_create_perturbation_heuristics(**params))
    if enable_improvement:
        heuristics.extend(_create_improvement_heuristics(**params))
    if enable_diversity:
        heuristics.extend(_create_diversity_heuristics(**params))
    if enable_meta:
        heuristics.extend(_create_meta_heuristics(**params))
    if enable_repair:
        heuristics.extend(_create_repair_heuristics(**params))

    return heuristics


# =============================================================================
# DEFAULT REGISTRIES
# =============================================================================


# Default instances with standard parameters
CONSTRUCTION_HEURISTICS: list[Heuristic] = []
PERTURBATION_HEURISTICS: list[Heuristic] = []
IMPROVEMENT_HEURISTICS: list[Heuristic] = []
DIVERSITY_HEURISTICS: list[Heuristic] = []
META_HEURISTICS: list[Heuristic] = []
REPAIR_HEURISTICS: list[Heuristic] = []
ALL_HEURISTICS: list[Heuristic] = []

# Lazy initialization
_initialized = False


def _ensure_initialized() -> None:
    global _initialized, CONSTRUCTION_HEURISTICS, PERTURBATION_HEURISTICS
    global IMPROVEMENT_HEURISTICS, DIVERSITY_HEURISTICS, META_HEURISTICS
    global REPAIR_HEURISTICS, ALL_HEURISTICS

    if _initialized:
        return

    CONSTRUCTION_HEURISTICS = _create_construction_heuristics()
    PERTURBATION_HEURISTICS = _create_perturbation_heuristics()
    IMPROVEMENT_HEURISTICS = _create_improvement_heuristics()
    DIVERSITY_HEURISTICS = _create_diversity_heuristics()
    META_HEURISTICS = _create_meta_heuristics()
    REPAIR_HEURISTICS = _create_repair_heuristics()

    ALL_HEURISTICS = (
        CONSTRUCTION_HEURISTICS
        + PERTURBATION_HEURISTICS
        + IMPROVEMENT_HEURISTICS
        + DIVERSITY_HEURISTICS
        + META_HEURISTICS
        + REPAIR_HEURISTICS
    )

    _initialized = True


def get_all_heuristic_objects() -> list[Heuristic]:
    """Get all heuristic objects (lazy init)."""
    _ensure_initialized()
    return ALL_HEURISTICS


def get_heuristics_by_category_oop(category: str) -> list[Heuristic]:
    """Get heuristics for a specific category."""
    _ensure_initialized()

    mapping = {
        "construction": CONSTRUCTION_HEURISTICS,
        "perturbation": PERTURBATION_HEURISTICS,
        "improvement": IMPROVEMENT_HEURISTICS,
        "diversity": DIVERSITY_HEURISTICS,
        "meta": META_HEURISTICS,
        "repair": REPAIR_HEURISTICS,
    }
    return mapping.get(category, [])


def get_heuristic_by_name_oop(name: str) -> Heuristic | None:
    """Lookup a single heuristic by name."""
    _ensure_initialized()
    for h in ALL_HEURISTICS:
        if h.name == name:
            return h
    return None


# Convenience names lists
HEURISTIC_NAMES: list[str] = []
ENABLED_HEURISTIC_NAMES: list[str] = []


def _init_name_lists() -> None:
    global HEURISTIC_NAMES, ENABLED_HEURISTIC_NAMES
    _ensure_initialized()
    HEURISTIC_NAMES = [h.name for h in ALL_HEURISTICS]
    ENABLED_HEURISTIC_NAMES = [h.name for h in ALL_HEURISTICS if h.enabled]


# Initialize name lists on import
_init_name_lists()
