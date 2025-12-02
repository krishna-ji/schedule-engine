"""
Base configuration using Python dataclasses.

Provides clean inheritance hierarchy:
- BaseConfig: Shared defaults across all experiments
- TestConfig: Test profile scaling (30 gens, 10 pop)
- ProdConfig: Production profile scaling (2000 gens, 200 pop)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

DEFAULT_REPAIR_HEURISTICS: dict[str, dict[str, int | bool]] = {
    "repair_instructor_availability": {"enabled": True, "priority": 1},
    "repair_group_overlaps": {"enabled": True, "priority": 2},
    "repair_room_overlap_reassign": {"enabled": True, "priority": 3},
    "repair_room_conflicts": {"enabled": True, "priority": 4},
    "repair_instructor_conflicts": {"enabled": True, "priority": 5},
    "repair_instructor_qualifications": {"enabled": True, "priority": 6},
    "repair_room_type_mismatches": {"enabled": True, "priority": 7},
}


@dataclass
class BaseConfig:
    """
    Abstract base config with shared defaults.

    Override fields in profile classes (TestConfig, ProdConfig) or
    experiment classes (BaselineTestConfig, etc.).
    """

    # ==========================================
    # TIME & QUANTA (Shared)
    # ==========================================
    quantum_minutes: int = 60
    opening_time: str = "10:00"
    closing_time: str = "17:00"
    closed_days: list[str] = field(default_factory=lambda: ["Saturday"])

    # Time constraint parameters (for scheduling logic)
    midday_break_start: str = "12:00"
    midday_break_end: str = "13:00"
    max_session_coalescence: int = 3
    max_sessions_per_day: int = 6
    preferred_block_size_min: int = 2
    preferred_block_size_max: int = 3

    # Soft constraint penalties
    theory_isolated_penalty: int = 5
    theory_oversized_penalty_per_quantum: int = 2
    theory_max_excused_isolated: int = 1
    practical_fragmentation_penalty: int = 10

    # ==========================================
    # GA PARAMETERS (Override in profiles)
    # ==========================================
    ngen: int = 100
    pop_size: int = 50
    cxpb: float = 0.70
    mutpb: float = 0.20

    # Elite preservation
    elite_preservation: bool = True
    elite_size: float = 0.05

    # Selection
    tournament_size: int = 2

    # Mutation strategy (DEFAULT: DISABLED - enable per experiment)
    use_constraint_guided_mutation: bool = False

    # ==========================================
    # PARALLEL PROCESSING
    # ==========================================
    use_multiprocessing: bool = True
    num_workers: int | None = None  # None = CPU count

    # ==========================================
    # POPULATION STRATEGY (DEFAULT: 100% RANDOM)
    # ==========================================
    population_strategy: str = "random"  # random, smart, hybrid
    greedy_percentage: float = 0.00  # Greedy initialization disabled
    smart_percentage: float = 0.00  # Smart initialization disabled
    random_percentage: float = 1.00  # 100% random initialization

    # ==========================================
    # KILLSWITCHES (DEFAULT: ALL DISABLED - enable per experiment)
    # ==========================================
    repair_enabled: bool = False
    heuristics_master_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # ==========================================
    # REPAIR CONFIGURATION (DEFAULT: DISABLED - enable per experiment)
    # ==========================================
    repair_max_iterations: int = 100
    repair_apply_after_mutation: bool = False
    repair_apply_after_crossover: bool = False
    repair_memetic_mode: bool = False
    repair_elite_percentage: float = 0.20
    repair_heuristics_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ==========================================
    # HEURISTICS CONFIGURATION
    # ==========================================
    heuristics_adaptive_priority_enabled: bool = False
    heuristics_reorder_interval: int = 10
    heuristics_evaluation_window: int = 10
    heuristics_min_applications: int = 3

    # Individual Heuristic Toggles (default: all enabled via registry)
    # Construction Heuristics
    heuristic_largest_degree_first: bool | None = None
    heuristic_most_constrained_first: bool | None = None
    heuristic_earliest_deadline_first: bool | None = None

    # Perturbation Heuristics
    heuristic_random_swap: bool | None = None
    heuristic_temporal_shift: bool | None = None
    heuristic_room_shuffle: bool | None = None
    heuristic_instructor_reassign: bool | None = None
    heuristic_multi_perturbation: bool | None = None

    # Improvement Heuristics
    heuristic_kempe_chain: bool | None = None
    heuristic_ejection_chain: bool | None = None
    heuristic_variable_depth_search: bool | None = None

    # Diversity Heuristics
    heuristic_distance_preserving_crossover: bool | None = None
    heuristic_crowding_mutation: bool | None = None
    heuristic_niching_selection: bool | None = None
    heuristic_adaptive_diversity_maintenance: bool | None = None

    # Meta Heuristics
    heuristic_variable_neighborhood_descent: bool | None = None
    heuristic_iterated_local_search: bool | None = None
    heuristic_adaptive_large_neighborhood: bool | None = None
    heuristic_guided_local_search: bool | None = None

    # Repair Heuristics
    heuristic_exhaustive_repair: bool | None = None
    heuristic_greedy_repair: bool | None = None
    heuristic_igls_repair: bool | None = None
    heuristic_lns_repair: bool | None = None
    heuristic_memetic_repair: bool | None = None
    heuristic_selective_repair: bool | None = None

    # ==========================================
    # CONSTRAINT WEIGHTS (Fitness function)
    # ==========================================
    hard_weight: float = -1.0
    soft_weight: float = -0.01

    # ==========================================
    # METRICS & REPORTING
    # ==========================================
    advanced_metrics_enabled: bool = True
    advanced_metrics_frequency: int = 10
    hypervolume_enabled: bool = True
    igd_enabled: bool = True
    gd_enabled: bool = True

    # Performance profiling
    performance_profiling_enabled: bool = True

    # ==========================================
    # PATHS (Can be overridden)
    # ==========================================
    data_dir: str = "data"
    output_dir: str = "output"
    output_subdir: str = ""  # Subdirectory within output_dir (e.g., "f-construction")

    # ==========================================
    # METADATA (Experiment tracking)
    # ==========================================
    name: str = "default"
    environment: str = "test"
    notes: str = ""
    experiment_id: str = ""

    # ==========================================
    # CLASS VARIABLES (Truly shared)
    # ==========================================
    WEEKDAYS: ClassVar[list[str]] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        if self.pop_size % 2 != 0:
            raise ValueError("pop_size must be even for NSGA-II")
        if self.ngen < 1:
            raise ValueError("ngen must be positive")
        if not 0 <= self.cxpb <= 1:
            raise ValueError("cxpb must be in [0, 1]")
        if not 0 <= self.mutpb <= 1:
            raise ValueError("mutpb must be in [0, 1]")

    @property
    def total_evaluations(self) -> int:
        """Total fitness evaluations (ngen × pop_size)."""
        return self.ngen * self.pop_size

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for Pydantic conversion."""
        return asdict(self)

    def to_pydantic(self) -> BaseConfig:
        """
        Convert to Pydantic model for validation.

        This bridges dataclass configs → Pydantic models for compatibility
        with existing codebase.
        """
        from src.config.models import Config

        # Map dataclass fields to Pydantic structure
        data = self._build_pydantic_dict()
        return Config(**data)

    def _build_pydantic_dict(self) -> dict[str, Any]:
        """Build nested dict structure for Pydantic Config model."""
        return {
            "name": self.name,
            "environment": self.environment,
            "io": {
                "data_dir": self.data_dir,
                "output_dir": self.output_dir,
            },
            "time": {
                "quantum_minutes": self.quantum_minutes,
                "opening_time": self.opening_time,
                "closing_time": self.closing_time,
                "closed_days": self.closed_days,
                "midday_break_start": self.midday_break_start,
                "midday_break_end": self.midday_break_end,
                "max_session_coalescence": self.max_session_coalescence,
                "max_sessions_per_day": self.max_sessions_per_day,
                "preferred_block_size_min": self.preferred_block_size_min,
                "preferred_block_size_max": self.preferred_block_size_max,
                "theory_isolated_penalty": self.theory_isolated_penalty,
                "theory_oversized_penalty_per_quantum": (
                    self.theory_oversized_penalty_per_quantum
                ),
                "theory_max_excused_isolated": self.theory_max_excused_isolated,
                "practical_fragmentation_penalty": self.practical_fragmentation_penalty,
            },
            "ga": {
                "ngen": self.ngen,
                "pop_size": self.pop_size,
                "cxpb": self.cxpb,
                "mutpb": self.mutpb,
                "tournament_size": self.tournament_size,
                "elite_preservation": self.elite_preservation,
                "elite_size": self.elite_size,
                "population_strategy": self.population_strategy,
                "use_constraint_guided_mutation": self.use_constraint_guided_mutation,
            },
            "parallel": {
                "use_multiprocessing": self.use_multiprocessing,
                "num_workers": self.num_workers,
            },
            "repair": {
                "enabled": self.repair_enabled,
                "max_iterations": self.repair_max_iterations,
                "apply_after_mutation": self.repair_apply_after_mutation,
                "apply_after_crossover": self.repair_apply_after_crossover,
                "memetic_mode": self.repair_memetic_mode,
                "elite_percentage": self.repair_elite_percentage,
                "heuristics": self._build_repair_heuristics_config(),
            },
            "heuristics": {
                "master_enabled": self.heuristics_master_enabled,
                "adaptive_priority": {
                    "enabled": self.heuristics_adaptive_priority_enabled,
                    "reorder_interval": self.heuristics_reorder_interval,
                    "evaluation_window": self.heuristics_evaluation_window,
                    "min_applications": self.heuristics_min_applications,
                },
                "construction": self._build_heuristic_toggles("construction"),
                "perturbation": self._build_heuristic_toggles("perturbation"),
                "improvement": self._build_heuristic_toggles("improvement"),
                "diversity": self._build_heuristic_toggles("diversity"),
                "meta": self._build_heuristic_toggles("meta"),
                "repair": self._build_heuristic_toggles("repair"),
            },
            "lns": {
                "enabled": self.lns_enabled,
            },
            "rl": {
                "enabled": self.rl_enabled,
            },
            "enhancements": {
                "master_enabled": self.enhancements_master_enabled,
            },
            "metrics": {
                "advanced_metrics_enabled": self.advanced_metrics_enabled,
                "advanced_metrics_frequency": self.advanced_metrics_frequency,
                "hypervolume_enabled": self.hypervolume_enabled,
                "igd_enabled": self.igd_enabled,
                "gd_enabled": self.gd_enabled,
            },
            "performance": {
                "profiling_enabled": self.performance_profiling_enabled,
            },
        }

    def _build_heuristic_toggles(self, category: str) -> dict[str, bool]:
        """Extract heuristic toggles for a specific category."""
        category_heuristics = {
            "construction": [
                "largest_degree_first",
                "most_constrained_first",
                "earliest_deadline_first",
            ],
            "perturbation": [
                "random_swap",
                "temporal_shift",
                "room_shuffle",
                "instructor_reassign",
                "multi_perturbation",
            ],
            "improvement": [
                "kempe_chain",
                "ejection_chain",
                "variable_depth_search",
            ],
            "diversity": [
                "distance_preserving_crossover",
                "crowding_mutation",
                "niching_selection",
                "adaptive_diversity_maintenance",
            ],
            "meta": [
                "variable_neighborhood_descent",
                "iterated_local_search",
                "adaptive_large_neighborhood",
                "guided_local_search",
            ],
            "repair": [
                "exhaustive_repair",
                "greedy_repair",
                "igls_repair",
                "lns_repair",
                "memetic_repair",
                "selective_repair",
            ],
        }

        heuristics = category_heuristics.get(category, [])
        toggles = {}

        for heuristic in heuristics:
            field_name = f"heuristic_{heuristic}"
            value = getattr(self, field_name, None)
            # Only include explicit toggles (None means use registry default)
            if value is not None:
                toggles[heuristic] = value

        return toggles

    def _build_repair_heuristics_config(self) -> dict[str, dict[str, Any]]:
        """Merge default repair heuristics with experiment overrides."""

        heuristics = {
            name: values.copy() for name, values in DEFAULT_REPAIR_HEURISTICS.items()
        }

        for name, override in self.repair_heuristics_overrides.items():
            entry = heuristics.setdefault(name, {})
            entry.update(override)

        return heuristics

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseConfig:
        """Create config from dictionary."""
        return cls(**data)
