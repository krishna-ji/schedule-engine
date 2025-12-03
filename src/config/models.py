"""Pydantic configuration models for Schedule Engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import model_validator as _model_validator

from src.constants import DEFAULT_EARLIEST_TIME, DEFAULT_LATEST_TIME

WEEKDAY_NAMES: list[str] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]

if TYPE_CHECKING:

    def model_validator(
        *args: Any, **kwargs: Any
    ) -> Callable[
        [Callable[..., Any]], Callable[..., Any]
    ]:  # pragma: no cover - typing shim
        ...

else:  # pragma: no cover - runtime path
    model_validator = _model_validator


_WEEKDAY_LOOKUP: dict[str, str] = {name.lower(): name for name in WEEKDAY_NAMES}


def _parse_time_to_minutes(time_str: str) -> int:
    """Convert HH:MM string to minutes since midnight."""

    try:
        hour_str, minute_str = time_str.split(":", maxsplit=1)
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid time format '{time_str}' (expected HH:MM)") from exc

    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"Time '{time_str}' out of 24h bounds")

    return hour * 60 + minute


def _normalize_day_name(day: str | None) -> str:
    """Normalize arbitrary day strings (case-insensitive) to canonical names."""

    if day is None:
        raise ValueError("Day name cannot be null")

    normalized = _WEEKDAY_LOOKUP.get(day.strip().lower())
    if normalized is None:
        raise ValueError(f"Invalid day name '{day}'")
    return normalized


class GAConfig(BaseModel):
    """Genetic Algorithm parameters"""

    ngen: int = Field(default=100, ge=1, le=10000)
    pop_size: int = Field(default=8, ge=2, le=10000)
    cxpb: float = Field(default=0.8, ge=0.0, le=1.0)
    mutpb: float = Field(default=0.3, ge=0.0, le=1.0)
    elite_preservation: bool = True
    elite_size: float = Field(default=0.05, ge=0.0, le=0.5)
    use_adaptive_probabilities: bool = True
    use_constraint_guided_mutation: bool = True
    population_strategy: Literal["hybrid", "smart", "random"] = "hybrid"
    validate_population_integrity: bool = False

    @field_validator("pop_size")
    @classmethod
    def pop_size_must_be_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError(f"Population size must be even for NSGA-II, got {v}")
        return v


class ParallelConfig(BaseModel):
    """Multiprocessing settings"""

    use_multiprocessing: bool = True
    num_workers: int | None = Field(
        default=None,
        description="Number of workers: None=CPU*2 (auto), >0=explicit count",
    )


class PerformanceConfig(BaseModel):
    """Performance profiling configuration"""

    enable_profiling: bool = Field(
        default=True, description="Enable detailed performance profiling"
    )
    show_per_generation: bool = Field(
        default=True, description="Show timing breakdown after each generation"
    )
    show_summary_table: bool = Field(
        default=True, description="Show summary table at end of evolution"
    )


class MetricsConfig(BaseModel):
    """Metrics calculation configuration (performance optimization)"""

    advanced_metrics_frequency: int = Field(
        default=10,
        ge=1,
        description="Calculate expensive metrics (hypervolume, IGD, spread) every N generations",
    )
    always_calculate_basic: bool = Field(
        default=True,
        description="Always calculate basic metrics (hard/soft violations, diversity)",
    )


class GPUConfig(BaseModel):
    """GPU acceleration configuration"""

    enabled: bool = Field(default=False, description="Enable GPU acceleration")
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto", description="Device selection: auto, cuda, or cpu"
    )
    batch_size: int = Field(
        default=128, ge=8, le=1024, description="GPU batch size for evaluation"
    )
    min_population_for_gpu: int = Field(
        default=100,
        ge=1,
        description="Minimum population size to use GPU (smaller uses CPU)",
    )
    fallback_to_cpu: bool = Field(
        default=True, description="Fall back to CPU if GPU fails"
    )
    auto_tune_batch_size: bool = Field(
        default=True, description="Automatically tune batch size for GPU memory"
    )


class ExhaustiveSearchConfig(BaseModel):
    """Exhaustive local search configuration (fixed generations)"""

    enabled: bool = True
    generations: list[int] = Field(
        default=[3, 25],
        description="Generations to trigger exhaustive search (e.g., [3, 25])",
    )
    population_coverage: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of population to optimize (0.3 = top 30%)",
    )
    max_neighborhood_size: int = Field(
        default=100, ge=10, le=500, description="Maximum neighbors to evaluate per gene"
    )
    timeout_seconds: int = Field(
        default=180, ge=30, le=1001, description="Abort if operation exceeds this time"
    )


class StagnationRepairConfig(BaseModel):
    """Stagnation-triggered greedy repair configuration"""

    enabled: bool = True
    patience: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Trigger after N generations without improvement",
    )
    min_generation: int = Field(
        default=8, ge=0, le=100, description="Don't trigger before this generation"
    )
    population_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of population to optimize (0.5 = top 50%)",
    )
    max_iterations: int = Field(
        default=10, ge=1, le=50, description="Max iterations per gene for greedy search"
    )
    timeout_seconds: int = Field(
        default=60, ge=10, le=300, description="Abort if operation exceeds this time"
    )
    cooldown: int = Field(
        default=3, ge=0, le=20, description="Generations to wait before re-triggering"
    )


class SelectiveRepairConfig(BaseModel):
    """Selective repair configuration (post-mutation cleanup)"""

    enabled: bool = True
    apply_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability of applying repair (0.3 = 30% of offspring)",
    )
    apply_after_mutation: bool = True
    apply_after_crossover: bool = False
    detection_strategy: Literal["fast", "full", "hybrid"] = "hybrid"


class LNSConfig(BaseModel):
    """LNS-CP Hybrid repair configuration"""

    enabled: bool = False
    repair_strategy: Literal["cp", "heuristic", "hybrid"] = Field(
        default="hybrid",
        description="Repair strategy: 'cp' (CP-SAT only), 'heuristic' (greedy/local search), 'hybrid' (heuristic first, escalate to CP)",
    )
    trigger_interval: int = Field(default=50, ge=1, le=1000)
    stagnation_threshold: int = Field(default=10, ge=1, le=100)
    force_trigger_generations: list[int] = Field(
        default_factory=list,
        description="Force LNS trigger on these specific generations (e.g., [6, 50] for testing/validation)",
    )
    trigger_before_igls: bool = Field(
        default=False,
        description="If true, trigger LNS before exhaustive search (IGLS) to avoid locked schedules",
    )
    max_subproblem_size: int = Field(default=20, ge=1, le=100)
    min_subproblem_size: int = Field(default=4, ge=1, le=50)
    expand_neighborhood_hops: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Expand conflicted sessions by N hops in conflict graph (0=disabled)",
    )
    cp_time_limit: float = Field(default=10.0, ge=1.0, le=300.0)
    igls_max_iterations: int = Field(
        default=500,
        ge=0,
        le=10000,
        description="Max iterations for IGLS local search repair (set 0 to disable when using `cp` strategy)",
    )
    igls_time_limit: float = Field(
        default=5.0,
        ge=0.0,
        le=3600.0,
        description="Time limit for IGLS repair in seconds (0.0 to disable, max 3600s = 1 hour)",
    )

    @classmethod
    @model_validator(mode="after")
    def _validate_heuristic_thresholds(cls, m: LNSConfig) -> LNSConfig:
        """Allow zero values for heuristics when repair_strategy is 'cp', otherwise apply minimums."""
        if m.repair_strategy != "cp":
            if m.igls_max_iterations < 10:
                raise ValueError(
                    "lns.igls_max_iterations must be >= 10 when using 'hybrid' or 'heuristic' strategies"
                )
            if m.igls_time_limit < 0.5:
                raise ValueError(
                    "lns.igls_time_limit must be >= 0.5 when using 'hybrid' or 'heuristic' strategies"
                )
        return m

    apply_to_best_n: int = Field(default=1, ge=1, le=10)
    enable_diagnostics: bool = Field(
        default=True,
        description="Log detailed subproblem diagnostics and infeasibility reasons",
    )
    pre_check_feasibility: bool = Field(
        default=True, description="Run pre-feasibility check before invoking CP-SAT"
    )


class RepairConfig(BaseModel):
    """Repair heuristics configuration.

    Components:
    - Base repair operators: 7 hard constraint repairs (HC1-HC5, HC8×2, HC4)
    - IGLS system: Exhaustive search + stagnation-triggered greedy repair
    - Selective repair: 3-4× faster violation-targeted mode
    - Memetic mode: Deep repair on elite individuals (Mode B)

    Constraint Coverage: 6 of 8 hard constraints, 1 of 4 soft constraints
    """

    enabled: bool = True
    max_iterations: int = Field(default=3, ge=1, le=500)
    apply_after_mutation: bool = True
    apply_after_crossover: bool = True
    memetic_mode: bool = False
    elite_percentage: float = Field(default=0.1, ge=0.0, le=1.0)
    memetic_iterations: int = Field(default=5, ge=1, le=20)
    violation_threshold: int | None = None
    selective_mode: bool = True
    detection_strategy: Literal["fast", "full", "hybrid"] = "hybrid"
    recheck_after_repair: bool = True
    adaptive_repair: dict[str, Any] = Field(default_factory=dict)
    heuristics: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # NEW: IGLS System
    exhaustive_search: ExhaustiveSearchConfig = Field(
        default_factory=ExhaustiveSearchConfig
    )
    stagnation_repair: StagnationRepairConfig = Field(
        default_factory=StagnationRepairConfig
    )
    selective_repair: SelectiveRepairConfig = Field(
        default_factory=SelectiveRepairConfig
    )


class ConstraintConfig(BaseModel):
    """Single constraint configuration"""

    enabled: bool = True
    weight: float = Field(ge=0.0)


class HardConstraintsConfig(BaseModel):
    """Hard constraints configuration"""

    student_group_exclusivity: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
    )
    instructor_exclusivity: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
    )
    instructor_qualifications: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
    )
    instructor_time_availability: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
    )
    room_suitability: ConstraintConfig = ConstraintConfig(enabled=True, weight=1.0)
    room_exclusivity: ConstraintConfig = ConstraintConfig(enabled=True, weight=1.0)
    room_time_availability: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
    )
    course_completeness: ConstraintConfig = ConstraintConfig(enabled=True, weight=1.0)


class SoftConstraintConfigWithPenalty(BaseModel):
    """Single soft constraint configuration with penalty factors"""

    enabled: bool = True
    weight: float = Field(ge=0.0)
    gap_penalty_per_quantum: int | None = Field(
        default=None,
        ge=0,
        description="Penalty per gap quantum (for gap-based constraints)",
    )
    distance_penalty_per_quantum: int | None = Field(
        default=None,
        ge=0,
        description="Penalty per quantum distance (for distance-based constraints)",
    )


class SoftConstraintsConfig(BaseModel):
    """Soft constraints configuration"""

    student_schedule_compactness: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=1.0, gap_penalty_per_quantum=2
        )
    )
    instructor_schedule_compactness: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=1.0, gap_penalty_per_quantum=1
        )
    )
    student_lunch_break: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=1.0, distance_penalty_per_quantum=2
        )
    )
    session_continuity: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=1.0
        )
    )
    paired_cohort_practical_alignment: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=1.0
        )
    )
    soft_weight_factor: float = Field(default=0.01, ge=0.0, le=1.0)


class FeasibilityConfig(BaseModel):
    """Feasibility checking configuration"""

    enable_checks: bool = True
    fail_on_infeasibility: bool = True
    tolerance_margin: float = Field(default=0.02, ge=0.0, le=0.2)
    generate_report: bool = True
    show_console_output: bool = True
    save_report_on_success: bool = True
    checks: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "instructor_workload": {"enabled": True, "severity": "critical"},
            "instructor_qualification_bottleneck": {
                "enabled": True,
                "severity": "critical",
            },
            "room_capacity_bottleneck": {"enabled": True, "severity": "critical"},
            "room_feature_bottleneck": {"enabled": True, "severity": "critical"},
            "group_pigeonhole": {"enabled": True, "severity": "critical"},
        }
    )


class DayOperatingHours(BaseModel):
    """Operating window for a single day."""

    start: str
    end: str

    @classmethod
    @model_validator(mode="after")
    def validate_range(cls, values: DayOperatingHours) -> DayOperatingHours:
        start_minutes = _parse_time_to_minutes(values.start)
        end_minutes = _parse_time_to_minutes(values.end)
        if start_minutes >= end_minutes:
            raise ValueError("Day operating hours must have end time after start time")
        return values


class TimeConfig(BaseModel):
    """Time and quantum settings"""

    quantum_minutes: int = Field(ge=15, le=120)
    opening_time: str = Field(
        default="10:00", description="Campus opening time (HH:MM)"
    )
    closing_time: str = Field(
        default="17:00", description="Campus closing time (HH:MM)"
    )
    closed_days: list[str] = Field(
        default_factory=lambda: ["Saturday"],
        description="Days where the campus is closed (no operating hours)",
    )
    day_overrides: dict[str, DayOperatingHours | None] = Field(
        default_factory=dict,
        description="Optional per-day overrides (use null to close a day)",
    )
    earliest_preferred_time: str = Field(default=DEFAULT_EARLIEST_TIME)
    cohort_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Paired cohorts for parallel practical scheduling (e.g., [('bei1a', 'bei1b')])",
    )
    latest_preferred_time: str = Field(default=DEFAULT_LATEST_TIME)
    midday_break_start: str
    midday_break_end: str
    max_session_coalescence: int = Field(ge=1, le=6)
    preferred_block_size_min: int = Field(ge=1, le=6)
    preferred_block_size_max: int = Field(ge=1, le=6)

    # Break placement enforcement (soft constraint)
    enforce_break_placement: bool = Field(
        default=True, description="Enable break placement soft constraint"
    )
    break_window_start: str = Field(
        default="12:00", description="Start of daily break window (HH:MM)"
    )
    break_window_end: str = Field(
        default="14:00", description="End of daily break window (HH:MM)"
    )
    break_min_quanta: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Minimum free quanta required in break window",
    )
    break_violation_penalty: int = Field(
        default=8, ge=0, le=100, description="Penalty per missing break quantum"
    )

    # Theory course block penalties
    theory_isolated_penalty: int = Field(
        ge=0,
        le=100,
        description="Penalty for isolated theory sessions (after first)",
    )
    theory_oversized_penalty_per_quantum: int = Field(
        ge=0, le=10, description="Penalty per quantum for theory blocks > 3"
    )
    theory_max_excused_isolated: int = Field(
        ge=0,
        le=3,
        description="Number of isolated sessions excused per course per day",
    )

    # Practical course block penalties
    practical_fragmentation_penalty: int = Field(
        ge=0,
        le=100,
        description="Penalty per split for fragmented practical sessions",
    )

    # Legacy parameters (kept for backward compatibility, may be deprecated)
    isolated_session_penalty: int = Field(default=5, ge=0, le=100)
    oversized_block_penalty_per_quantum: int = Field(default=1, ge=0, le=10)

    max_sessions_per_day: int = Field(ge=1, le=12)

    @classmethod
    @model_validator(mode="after")
    def validate_operating_hours(cls, values: TimeConfig) -> TimeConfig:
        open_minutes = _parse_time_to_minutes(values.opening_time)
        close_minutes = _parse_time_to_minutes(values.closing_time)
        if open_minutes >= close_minutes:
            raise ValueError("time.closing_time must be later than time.opening_time")

        normalized_closed = []
        for day in values.closed_days:
            normalized_closed.append(_normalize_day_name(day))
        values.closed_days = normalized_closed

        normalized_overrides: dict[str, DayOperatingHours | None] = {}
        for day, override in values.day_overrides.items():
            normalized_day = _normalize_day_name(day)
            normalized_overrides[normalized_day] = override
        values.day_overrides = normalized_overrides

        return values


class IOConfig(BaseModel):
    """Input/output paths"""

    data_dir: str
    output_dir: str


class CalendarConfig(BaseModel):
    """Calendar display settings"""

    show_instructor: bool = True  # Keep default (rarely changes)
    show_room: bool = True
    show_group: bool = True


class ColorPaletteConfig(BaseModel):
    """Color palette for visualization"""

    course_colors: dict[str, str] = Field(default_factory=dict)


class HypermutationConfig(BaseModel):
    """Hypermutation settings for escaping local optima"""

    enabled: bool = True
    trigger_on_stagnation: bool = True
    stagnation_window: int = Field(default=5, ge=3, le=20)
    duration_generations: int = Field(default=2, ge=1, le=5)
    mutation_rate: float = Field(default=0.8, ge=0.3, le=1.0)


class ConstraintPrioritiesConfig(BaseModel):
    """Constraint-specific repair priorities"""

    enabled: bool = True
    availability_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    overlap_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    other_weight: float = Field(default=0.05, ge=0.0, le=1.0)


class PopulationRestartConfig(BaseModel):
    """Population restart settings (risky - use as last resort)"""

    enabled: bool = Field(
        default=False, description="Enable population restart (RISKY - destroys 50%)"
    )
    trigger_stagnation_gens: int = Field(
        default=15, ge=10, le=50, description="Stagnation threshold (generations)"
    )
    restart_percentage: float = Field(
        default=0.5, ge=0.3, le=0.7, description="Percentage of population to replace"
    )
    min_interval_gens: int = Field(
        default=50, ge=20, le=200, description="Minimum generations between restarts"
    )


class ViolationHeatmapConfig(BaseModel):
    """Constraint violation heatmap for targeted repair"""

    enabled: bool = Field(
        default=True, description="Track violation frequency per gene"
    )
    target_hot_genes: bool = Field(
        default=True, description="Prioritize repair on frequently-violated genes"
    )
    top_n_hotspots: int = Field(
        default=20, ge=5, le=100, description="Number of hotspots to target"
    )
    persistence_file: str = Field(
        default="violation_heatmap.json", description="File to persist heatmap data"
    )


class MultiNeighborhoodConfig(BaseModel):
    """Multi-neighborhood local search for repair"""

    enabled: bool = Field(
        default=True,
        description="Try combined moves (time+instructor+room simultaneously)",
    )
    max_combinations: int = Field(
        default=50, ge=10, le=200, description="Max instructor-room combinations to try"
    )
    fallback_to_single: bool = Field(
        default=True,
        description="Fall back to single-neighborhood if combined fails",
    )


class EnhancementConfig(BaseModel):
    """
    Master switch for all GA enhancements.

    Phase 1 (Immediate):
    - memetic_mode: Apply light repair every generation to elite
    - increased_population: Use larger population sizes
    - frequent_repair: More aggressive repair intervals

    Phase 2 (High Priority):
    - hypermutation: Escape local optima with temporary high mutation
    - constraint_priorities: Focus repair on worst violations first
    - greedy_initialization: More greedy seeds in hybrid population

    Phase 3 (Advanced):
    - population_restart: Replace worst 50% when stuck (RISKY)
    - violation_heatmap: Track hot genes, target repairs
    - multi_neighborhood: Combined moves in repair (time+instructor+room)
    """

    master_enabled: bool = Field(
        default=True,
        description="Master switch - disable to revert to baseline GA",
    )

    # Phase 1: Immediate wins
    memetic_mode: bool = Field(
        default=True, description="Apply light repair to elite every generation"
    )
    increased_population: bool = Field(
        default=True, description="Use larger population sizes"
    )
    frequent_repair: bool = Field(
        default=True, description="More aggressive repair intervals"
    )

    # Phase 2: High priority
    hypermutation: HypermutationConfig = Field(default_factory=HypermutationConfig)
    constraint_priorities: ConstraintPrioritiesConfig = Field(
        default_factory=ConstraintPrioritiesConfig
    )
    greedy_initialization_percent: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Percentage of population for greedy initialization",
    )

    # Phase 3: Advanced features
    population_restart: PopulationRestartConfig = Field(
        default_factory=PopulationRestartConfig
    )
    violation_heatmap: ViolationHeatmapConfig = Field(
        default_factory=ViolationHeatmapConfig
    )
    multi_neighborhood: MultiNeighborhoodConfig = Field(
        default_factory=MultiNeighborhoodConfig
    )


class HeuristicsConfig(BaseModel):
    """
    Heuristic Toolbox Configuration (Phase 1.5)

    Six categories of heuristic operators:
    - construction: Build schedules greedily from scratch
    - perturbation: Shake solutions to escape local optima
    - improvement: Local search moves for refinement
    - diversity: Maintain population diversity
    - meta: High-level search strategies
    - repair: Fix constraint violations

    Each category contains multiple heuristics with individual killswitches.
    Heuristics are integrated via decorator-based registry (like constraints).
    """

    master_enabled: bool = Field(
        default=True,
        description="Master killswitch for ALL heuristics (overrides individual settings)",
    )
    adaptive_priority: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "reorder_interval": 10,
            "evaluation_window": 10,
            "min_applications": 3,
        },
        description="Adaptive priority adjustment (dynamic heuristic reordering)",
    )
    construction: dict[str, Any] = Field(
        default_factory=dict,
        description="Construction heuristics for greedy schedule building",
    )
    perturbation: dict[str, Any] = Field(
        default_factory=dict,
        description="Perturbation heuristics for diversification",
    )
    improvement: dict[str, Any] = Field(
        default_factory=dict,
        description="Improvement heuristics for local search",
    )
    diversity: dict[str, Any] = Field(
        default_factory=dict,
        description="Diversity heuristics for population maintenance",
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Meta-heuristics for high-level search strategies",
    )
    repair: dict[str, Any] = Field(
        default_factory=dict,
        description="Repair heuristics for constraint violation fixes",
    )


class RLEnvironmentConfig(BaseModel):
    """RL environment configuration"""

    max_steps_per_episode: int = Field(default=100, ge=1, le=1000)
    observation_history_size: int = Field(default=10, ge=1, le=50)
    render_mode: Literal["ansi", "human"] | None = None


class RLRewardConfig(BaseModel):
    """RL reward function configuration"""

    fitness_weight: float = Field(default=1.0, ge=0.0)
    diversity_weight: float = Field(default=0.1, ge=0.0)
    time_weight: float = Field(default=0.01, ge=0.0)
    normalize: bool = True


class RLPPOConfig(BaseModel):
    """PPO agent hyperparameters"""

    learning_rate: float = Field(default=0.0003, gt=0.0)
    n_steps: int = Field(default=2048, ge=1)
    batch_size: int = Field(default=64, ge=1)
    n_epochs: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0.0)
    ent_coef: float = Field(default=0.01, ge=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)


class RLDQNConfig(BaseModel):
    """DQN agent hyperparameters"""

    learning_rate: float = Field(default=0.0001, gt=0.0)
    buffer_size: int = Field(default=100000, ge=1)
    learning_starts: int = Field(default=1000, ge=1)
    batch_size: int = Field(default=32, ge=1)
    tau: float = Field(default=0.005, gt=0.0, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    exploration_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    exploration_initial_eps: float = Field(default=1.0, ge=0.0, le=1.0)
    exploration_final_eps: float = Field(default=0.05, ge=0.0, le=1.0)


class RLAgentConfig(BaseModel):
    """RL agent configuration"""

    type: Literal["ppo", "dqn", "random"] = "ppo"
    model_path: str = "models/rl_agents/best_model.zip"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    ppo: RLPPOConfig = Field(default_factory=RLPPOConfig)
    dqn: RLDQNConfig = Field(default_factory=RLDQNConfig)


class RLCurriculumStage(BaseModel):
    """Curriculum learning stage"""

    name: str
    num_courses: int = Field(ge=1)
    max_generations: int = Field(ge=1)
    total_timesteps: int = Field(ge=1)


class RLTrainingConfig(BaseModel):
    """RL training configuration"""

    total_timesteps: int = Field(default=100000, ge=1)
    checkpoint_interval: int = Field(default=10000, ge=1)
    evaluation_interval: int = Field(default=5000, ge=1)
    tensorboard_log: str = "logs/tensorboard"
    checkpoint_dir: str = "models/rl_agents/checkpoints"
    save_dir: str = "models/rl_agents"
    verbose: int = Field(default=1, ge=0, le=2)
    curriculum: list[dict[str, Any]] = Field(default_factory=list)


class RLInferenceConfig(BaseModel):
    """RL inference configuration"""

    batch_prediction: bool = False
    timeout_ms: int = Field(default=10, ge=1, le=1000)
    fallback_on_timeout: bool = True
    cache_predictions: bool = False


class RLHybridConfig(BaseModel):
    """RL hybrid controller configuration"""

    mode: Literal["rl_primary", "rl_fallback", "rl_assisted"] = "rl_primary"
    fallback_strategy: Literal["random", "greedy", "round_robin"] = "random"
    rl_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    enable_action_masking: bool = True


class RLEvaluationConfig(BaseModel):
    """RL evaluation configuration"""

    baseline_strategies: list[str] = Field(
        default_factory=lambda: ["random", "round_robin", "greedy", "fixed_priority"]
    )
    num_evaluation_episodes: int = Field(default=10, ge=1)
    save_metrics: bool = True
    metrics_dir: str = "output/rl_metrics"


class RLLoggingConfig(BaseModel):
    """RL logging configuration"""

    log_heuristic_usage: bool = True
    log_rewards: bool = True
    log_state_transitions: bool = False
    log_inference_time: bool = True


class RLConfig(BaseModel):
    """Reinforcement Learning integration configuration"""

    enabled: bool = False
    mode: Literal[
        "disabled",
        "training",
        "inference",
        "hybrid",
        "rl_primary",
        "rl_fallback",
        "rl_assisted",
    ] = "disabled"
    environment: RLEnvironmentConfig = Field(default_factory=RLEnvironmentConfig)
    reward: RLRewardConfig = Field(default_factory=RLRewardConfig)
    agent: RLAgentConfig = Field(default_factory=RLAgentConfig)
    training: RLTrainingConfig = Field(default_factory=RLTrainingConfig)
    inference: RLInferenceConfig = Field(default_factory=RLInferenceConfig)
    hybrid: RLHybridConfig = Field(default_factory=RLHybridConfig)
    evaluation: RLEvaluationConfig = Field(default_factory=RLEvaluationConfig)
    logging: RLLoggingConfig = Field(default_factory=RLLoggingConfig)


class Config(BaseModel):
    """Master configuration for Schedule Engine"""

    name: str = "default"
    environment: Literal["test", "prod"] = "test"

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    ga: GAConfig = Field(default_factory=GAConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    lns: LNSConfig = Field(default_factory=LNSConfig)
    hard_constraints: HardConstraintsConfig = Field(
        default_factory=HardConstraintsConfig
    )
    soft_constraints: SoftConstraintsConfig = Field(
        default_factory=SoftConstraintsConfig
    )
    feasibility: FeasibilityConfig = Field(default_factory=FeasibilityConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)  # type: ignore[arg-type]
    io: IOConfig = Field(default_factory=IOConfig)  # type: ignore[arg-type]
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    colors: ColorPaletteConfig = Field(default_factory=ColorPaletteConfig)
    enhancements: EnhancementConfig = Field(default_factory=EnhancementConfig)
    heuristics: HeuristicsConfig = Field(default_factory=HeuristicsConfig)
    rl: RLConfig = Field(default_factory=RLConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load config from YAML file"""
        import yaml  # type: ignore[import-untyped]

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file"""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def summary(self) -> str:
        """Human-readable configuration summary"""
        parallel_mode = "enabled" if self.parallel.use_multiprocessing else "disabled"
        parallel_workers = self.parallel.num_workers or "auto"
        gpu_mode = "enabled" if self.gpu.enabled else "disabled"
        repair_mode = "enabled" if self.repair.enabled else "disabled"
        feasibility_mode = "enabled" if self.feasibility.enable_checks else "disabled"

        return f"""[bold cyan]configuration[/bold cyan]
  [dim]profile:[/dim] {self.name} ({self.environment})
  [dim]genetic algorithm:[/dim] {self.ga.ngen} gen x {self.ga.pop_size} pop | cx={self.ga.cxpb} mut={self.ga.mutpb}
  [dim]parallelization:[/dim] {parallel_mode} ({parallel_workers} workers)
  [dim]gpu acceleration:[/dim] {gpu_mode} (batch={self.gpu.batch_size})
  [dim]repair heuristics:[/dim] {repair_mode} (max {self.repair.max_iterations} iterations)
  [dim]feasibility checks:[/dim] {feasibility_mode}
  [dim]data:[/dim] {self.io.data_dir}/ to {self.io.output_dir}/
        """.strip()
