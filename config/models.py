"""
Pydantic configuration models for Schedule Engine.
All configs loaded from YAML files with full validation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Optional, Literal, Any
from pathlib import Path


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


class RepairConfig(BaseModel):
    """Repair heuristics configuration"""

    enabled: bool = True
    max_iterations: int = Field(default=3, ge=1, le=10)
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


class ConstraintConfig(BaseModel):
    """Single constraint configuration"""

    enabled: bool = True
    weight: float = Field(ge=0.0)


class HardConstraintsConfig(BaseModel):
    """Hard constraints configuration"""

    no_group_overlap: ConstraintConfig = ConstraintConfig(enabled=True, weight=2.0)
    no_instructor_conflict: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=2.0
    )
    instructor_not_qualified: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=2.0
    )
    room_type_mismatch: ConstraintConfig = ConstraintConfig(enabled=True, weight=2.0)
    availability_violations: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=2.0
    )
    incomplete_or_extra_sessions: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
    )


class SoftConstraintsConfig(BaseModel):
    """Soft constraints configuration"""

    group_gaps_penalty: ConstraintConfig = ConstraintConfig(enabled=True, weight=1.0)
    instructor_gaps_penalty: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
    )
    group_midday_break_violation: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
    )
    session_block_clustering_penalty: ConstraintConfig = ConstraintConfig(
        enabled=True, weight=1.0
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

    quantum_minutes: int = Field(default=60, ge=15, le=120)
    earliest_preferred_time: str = "10:00"
    latest_preferred_time: str = "17:00"
    midday_break_start: str = "12:00"
    midday_break_end: str = "14:00"
    max_session_coalescence: int = Field(default=3, ge=1, le=6)
    preferred_block_size_min: int = Field(default=2, ge=1, le=6)
    preferred_block_size_max: int = Field(default=3, ge=1, le=6)
    isolated_session_penalty: int = Field(default=5, ge=0, le=100)
    oversized_block_penalty_per_quantum: int = Field(default=1, ge=0, le=10)
    max_sessions_per_day: int = Field(default=5, ge=1, le=12)


class IOConfig(BaseModel):
    """Input/output paths"""

    data_dir: str = "data"
    output_dir: str = "output"


class CalendarConfig(BaseModel):
    """Calendar display settings"""

    show_instructor: bool = True
    show_room: bool = True
    show_group: bool = True


class ColorPaletteConfig(BaseModel):
    """Color palette for visualization"""

    course_colors: Dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    """Master configuration for Schedule Engine"""

    name: str = "default"
    environment: Literal["test", "dev", "prod"] = "dev"

    ga: GAConfig = Field(default_factory=GAConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
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
        """Human-readable summary"""
        return f"""
Configuration: {self.name} ({self.environment})
- GA: {self.ga.ngen} gen x {self.ga.pop_size} pop (CX={self.ga.cxpb}, MUT={self.ga.mutpb})
- Parallel: {"ON" if self.parallel.use_multiprocessing else "OFF"} ({self.parallel.num_workers or "auto"} workers)
- Repair: {"ON" if self.repair.enabled else "OFF"} (max {self.repair.max_iterations} iter)
- Feasibility: {"ON" if self.feasibility.enable_checks else "OFF"} (fail={self.feasibility.fail_on_infeasibility})
- I/O: {self.io.data_dir} -> {self.io.output_dir}
        """.strip()
