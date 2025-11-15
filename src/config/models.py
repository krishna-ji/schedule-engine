"""
Pydantic configuration models for Schedule Engine.
All configs loaded from YAML files with full validation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Optional, Literal, Any, List


class GAConfig(BaseModel):
    """Genetic Algorithm parameters"""

    ngen: int = Field(default=100, ge=1, le=10000)
    pop_size: int = Field(default=8, ge=2, le=1000)
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
    def pop_size_must_be_even(cls, v):
        if v % 2 != 0:
            raise ValueError(f"Population size must be even for NSGA-II, got {v}")
        return v


class ParallelConfig(BaseModel):
    """Multiprocessing settings"""

    use_multiprocessing: bool = True
    num_workers: Optional[int] = Field(default=None, ge=1, le=64)


class ExhaustiveSearchConfig(BaseModel):
    """Exhaustive local search configuration (fixed generations)"""

    enabled: bool = True
    generations: List[int] = Field(
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
    force_trigger_generations: List[int] = Field(
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

    @model_validator(mode="after")
    def _validate_heuristic_thresholds(cls, m):
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
    """Repair heuristics configuration (legacy + new IGLS)"""

    enabled: bool = True
    max_iterations: int = Field(default=3, ge=1, le=500)
    apply_after_mutation: bool = True
    apply_after_crossover: bool = True
    memetic_mode: bool = False
    elite_percentage: float = Field(default=0.1, ge=0.0, le=1.0)
    memetic_iterations: int = Field(default=5, ge=1, le=20)
    violation_threshold: Optional[int] = None
    selective_mode: bool = True
    detection_strategy: Literal["fast", "full", "hybrid"] = "hybrid"
    recheck_after_repair: bool = True
    adaptive_repair: Dict[str, Any] = Field(default_factory=dict)
    heuristics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

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
        enabled=True, weight=3.0
    )
    instructor_exclusivity: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=3.0
    )
    instructor_qualifications: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=3.0
    )
    instructor_time_availability: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=3.0
    )
    room_suitability: ConstraintConfig = ConstraintConfig(enabled=True, weight=2.5)
    room_exclusivity: ConstraintConfig = ConstraintConfig(enabled=True, weight=3.0)
    room_time_availability: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=2.5
    )
    course_completeness: ConstraintConfig = ConstraintConfig(enabled=True, weight=2.0)


class SoftConstraintConfigWithPenalty(BaseModel):
    """Single soft constraint configuration with penalty factors"""

    enabled: bool = True
    weight: float = Field(ge=0.0)
    gap_penalty_per_quantum: Optional[int] = Field(
        default=None,
        ge=0,
        description="Penalty per gap quantum (for gap-based constraints)",
    )
    distance_penalty_per_quantum: Optional[int] = Field(
        default=None,
        ge=0,
        description="Penalty per quantum distance (for distance-based constraints)",
    )


class SoftConstraintsConfig(BaseModel):
    """Soft constraints configuration"""

    student_schedule_compactness: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=1.5, gap_penalty_per_quantum=2
        )
    )
    instructor_schedule_compactness: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=1.0, gap_penalty_per_quantum=1
        )
    )
    student_lunch_break: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=1.2, distance_penalty_per_quantum=2
        )
    )
    session_continuity: SoftConstraintConfigWithPenalty = Field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            enabled=True, weight=2.0
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
    checks: Dict[str, Dict[str, Any]] = Field(
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


class TimeConfig(BaseModel):
    """Time and quantum settings"""

    quantum_minutes: int = Field(ge=15, le=120)
    earliest_preferred_time: str
    latest_preferred_time: str
    midday_break_start: str
    midday_break_end: str
    max_session_coalescence: int = Field(ge=1, le=6)
    preferred_block_size_min: int = Field(ge=1, le=6)
    preferred_block_size_max: int = Field(ge=1, le=6)

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

    course_colors: Dict[str, str] = Field(default_factory=dict)


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


class Config(BaseModel):
    """Master configuration for Schedule Engine"""

    name: str = "default"
    environment: Literal["test", "prod"] = "test"

    ga: GAConfig = Field(default_factory=GAConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    lns: LNSConfig = Field(default_factory=LNSConfig)
    hard_constraints: HardConstraintsConfig = Field(
        default_factory=HardConstraintsConfig
    )
    soft_constraints: SoftConstraintsConfig = Field(
        default_factory=SoftConstraintsConfig
    )
    feasibility: FeasibilityConfig = Field(default_factory=FeasibilityConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    colors: ColorPaletteConfig = Field(default_factory=ColorPaletteConfig)
    enhancements: EnhancementConfig = Field(default_factory=EnhancementConfig)

    class Config:
        validate_assignment = True
        extra = "allow"  # Allow extra fields for flexibility

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file"""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save config to YAML file"""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)

    def summary(self) -> str:
        """Human-readable configuration summary"""
        parallel_mode = "enabled" if self.parallel.use_multiprocessing else "disabled"
        parallel_workers = self.parallel.num_workers or "auto"
        repair_mode = "enabled" if self.repair.enabled else "disabled"
        feasibility_mode = "enabled" if self.feasibility.enable_checks else "disabled"

        return f"""[bold cyan]configuration[/bold cyan]
  [dim]profile:[/dim] {self.name} ({self.environment})
  [dim]genetic algorithm:[/dim] {self.ga.ngen} gen x {self.ga.pop_size} pop | cx={self.ga.cxpb} mut={self.ga.mutpb}
  [dim]parallelization:[/dim] {parallel_mode} ({parallel_workers} workers)
  [dim]repair heuristics:[/dim] {repair_mode} (max {self.repair.max_iterations} iterations)
  [dim]feasibility checks:[/dim] {feasibility_mode}
  [dim]data:[/dim] {self.io.data_dir}/ to {self.io.output_dir}/
        """.strip()
