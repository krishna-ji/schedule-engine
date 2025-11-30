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

    # ==========================================
    # PARALLEL PROCESSING
    # ==========================================
    use_multiprocessing: bool = True
    num_workers: int | None = None  # None = CPU count

    # ==========================================
    # POPULATION STRATEGY
    # ==========================================
    population_strategy: str = "hybrid"  # hybrid, random, greedy
    greedy_percentage: float = 0.25
    smart_percentage: float = 0.50
    random_percentage: float = 0.25

    # ==========================================
    # KILLSWITCHES (Master controls)
    # ==========================================
    repair_enabled: bool = True
    heuristics_master_enabled: bool = True
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # ==========================================
    # REPAIR CONFIGURATION
    # ==========================================
    repair_max_iterations: int = 100
    repair_apply_after_mutation: bool = True
    repair_apply_after_crossover: bool = False
    repair_memetic_mode: bool = False
    repair_elite_percentage: float = 0.20

    # ==========================================
    # HEURISTICS CONFIGURATION
    # ==========================================
    heuristics_adaptive_priority_enabled: bool = False
    heuristics_reorder_interval: int = 10
    heuristics_evaluation_window: int = 10
    heuristics_min_applications: int = 3

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

    def to_pydantic(self) -> Config:
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
            },
            "heuristics": {
                "master_enabled": self.heuristics_master_enabled,
                "adaptive_priority": {
                    "enabled": self.heuristics_adaptive_priority_enabled,
                    "reorder_interval": self.heuristics_reorder_interval,
                    "evaluation_window": self.heuristics_evaluation_window,
                    "min_applications": self.heuristics_min_applications,
                },
                "construction": {},  # Individual heuristics populated by registry
                "perturbation": {},
                "improvement": {},
                "diversity": {},
                "meta": {},
                "repair": {},
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseConfig:
        """Create config from dictionary."""
        return cls(**data)
