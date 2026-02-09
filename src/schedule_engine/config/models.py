"""
Simple dataclass-based configuration for Schedule Engine.

No Pydantic, no validation, no transformation layers.
Just data holders with sensible defaults.

Run files create Config() directly with overrides.
Internal code accesses via get_config() / get_config_or_default().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# GA (Genetic Algorithm)
# =============================================================================


@dataclass
class GAConfig:
    ngen: int = 100
    pop_size: int = 50
    cxpb: float = 0.8
    mutpb: float = 0.3
    elite_preservation: bool = True
    elite_size: float = 0.05
    use_adaptive_probabilities: bool = True
    use_constraint_guided_mutation: bool = True
    population_strategy: str = "hybrid"  # "hybrid", "smart", "random"
    validate_population_integrity: bool = False


# =============================================================================
# REPAIR
# =============================================================================


@dataclass
class ExhaustiveSearchConfig:
    enabled: bool = True
    generations: list[int] = field(default_factory=lambda: [3, 25])
    population_coverage: float = 0.3
    max_neighborhood_size: int = 100
    timeout_seconds: int = 180


@dataclass
class StagnationRepairConfig:
    enabled: bool = True
    patience: int = 5
    min_generation: int = 8
    population_coverage: float = 0.5
    max_iterations: int = 10
    timeout_seconds: int = 60
    cooldown: int = 3


@dataclass
class SelectiveRepairConfig:
    enabled: bool = True
    apply_probability: float = 0.3
    apply_after_mutation: bool = True
    apply_after_crossover: bool = False
    detection_strategy: str = "hybrid"


@dataclass
class RepairConfig:
    enabled: bool = True
    max_iterations: int = 3
    apply_after_mutation: bool = True
    apply_after_crossover: bool = True
    memetic_mode: bool = False
    elite_percentage: float = 0.1
    memetic_iterations: int = 5
    violation_threshold: int | None = None
    selective_mode: bool = True
    detection_strategy: str = "hybrid"
    recheck_after_repair: bool = True
    adaptive_repair: dict[str, Any] = field(default_factory=dict)
    heuristics: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget_ms_per_generation: int = 50
    max_steps_per_individual: int = 5
    max_candidates_per_operator: int = 20
    policy: str = "round_robin"
    epsilon: float = 0.1
    exhaustive_search: ExhaustiveSearchConfig = field(
        default_factory=ExhaustiveSearchConfig
    )
    stagnation_repair: StagnationRepairConfig = field(
        default_factory=StagnationRepairConfig
    )
    selective_repair: SelectiveRepairConfig = field(
        default_factory=SelectiveRepairConfig
    )


# =============================================================================
# CONSTRAINTS
# =============================================================================


@dataclass
class ConstraintConfig:
    enabled: bool = True
    weight: float = 1.0


@dataclass
class SoftConstraintConfigWithPenalty:
    enabled: bool = True
    weight: float = 1.0
    gap_penalty_per_quantum: int | None = None
    distance_penalty_per_quantum: int | None = None


@dataclass
class HardConstraintsConfig:
    student_group_exclusivity: ConstraintConfig = field(
        default_factory=ConstraintConfig
    )
    instructor_exclusivity: ConstraintConfig = field(default_factory=ConstraintConfig)
    instructor_qualifications: ConstraintConfig = field(
        default_factory=ConstraintConfig
    )
    instructor_time_availability: ConstraintConfig = field(
        default_factory=ConstraintConfig
    )
    room_suitability: ConstraintConfig = field(default_factory=ConstraintConfig)
    room_exclusivity: ConstraintConfig = field(default_factory=ConstraintConfig)
    room_time_availability: ConstraintConfig = field(default_factory=ConstraintConfig)
    course_completeness: ConstraintConfig = field(default_factory=ConstraintConfig)


@dataclass
class SoftConstraintsConfig:
    student_schedule_compactness: SoftConstraintConfigWithPenalty = field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            gap_penalty_per_quantum=1
        )
    )
    instructor_schedule_compactness: SoftConstraintConfigWithPenalty = field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            gap_penalty_per_quantum=1
        )
    )
    student_lunch_break: SoftConstraintConfigWithPenalty = field(
        default_factory=lambda: SoftConstraintConfigWithPenalty(
            distance_penalty_per_quantum=1
        )
    )
    session_continuity: SoftConstraintConfigWithPenalty = field(
        default_factory=SoftConstraintConfigWithPenalty
    )
    paired_cohort_practical_alignment: SoftConstraintConfigWithPenalty = field(
        default_factory=SoftConstraintConfigWithPenalty
    )
    soft_weight_factor: float = 1.0


# =============================================================================
# LNS (Large Neighborhood Search)
# =============================================================================


@dataclass
class LNSConfig:
    enabled: bool = False
    repair_strategy: str = "hybrid"
    trigger_interval: int = 50
    stagnation_threshold: int = 10
    force_trigger_generations: list[int] = field(default_factory=list)
    trigger_before_igls: bool = False
    max_subproblem_size: int = 20
    min_subproblem_size: int = 4
    expand_neighborhood_hops: int = 0
    cp_time_limit: float = 10.0
    igls_max_iterations: int = 500
    igls_time_limit: float = 5.0
    apply_to_best_n: int = 1
    enable_diagnostics: bool = True
    pre_check_feasibility: bool = True


# =============================================================================
# ENHANCEMENTS
# =============================================================================


@dataclass
class HypermutationConfig:
    enabled: bool = True
    trigger_on_stagnation: bool = True
    stagnation_window: int = 5
    duration_generations: int = 2
    mutation_rate: float = 0.8


@dataclass
class ConstraintPrioritiesConfig:
    enabled: bool = True
    availability_weight: float = 0.8
    overlap_weight: float = 0.15
    other_weight: float = 0.05


@dataclass
class PopulationRestartConfig:
    enabled: bool = False
    trigger_stagnation_gens: int = 15
    restart_percentage: float = 0.5
    min_interval_gens: int = 50


@dataclass
class ViolationHeatmapConfig:
    enabled: bool = True
    target_hot_genes: bool = True
    top_n_hotspots: int = 20
    persistence_file: str = "violation_heatmap.json"


@dataclass
class MultiNeighborhoodConfig:
    enabled: bool = True
    max_combinations: int = 50
    fallback_to_single: bool = True


@dataclass
class EnhancementConfig:
    master_enabled: bool = True
    memetic_mode: bool = True
    increased_population: bool = True
    frequent_repair: bool = True
    hypermutation: HypermutationConfig = field(default_factory=HypermutationConfig)
    constraint_priorities: ConstraintPrioritiesConfig = field(
        default_factory=ConstraintPrioritiesConfig
    )
    greedy_initialization_percent: float = 0.4
    population_restart: PopulationRestartConfig = field(
        default_factory=PopulationRestartConfig
    )
    violation_heatmap: ViolationHeatmapConfig = field(
        default_factory=ViolationHeatmapConfig
    )
    multi_neighborhood: MultiNeighborhoodConfig = field(
        default_factory=MultiNeighborhoodConfig
    )


# =============================================================================
# HEURISTICS
# =============================================================================


@dataclass
class HeuristicsConfig:
    master_enabled: bool = True
    adaptive_priority: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "reorder_interval": 10,
            "evaluation_window": 10,
            "min_applications": 3,
        }
    )
    construction: dict[str, Any] = field(default_factory=dict)
    perturbation: dict[str, Any] = field(default_factory=dict)
    improvement: dict[str, Any] = field(default_factory=dict)
    diversity: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    repair: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RL (Reinforcement Learning)
# =============================================================================


@dataclass
class RLPPOConfig:
    learning_rate: float = 0.0003
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class RLDQNConfig:
    learning_rate: float = 0.0001
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 32
    tau: float = 0.005
    gamma: float = 0.99
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05


@dataclass
class RLAgentConfig:
    type: str = "ppo"
    model_path: str = "models/rl_agents/best_model.zip"
    device: str = "cpu"
    ppo: RLPPOConfig = field(default_factory=RLPPOConfig)
    dqn: RLDQNConfig = field(default_factory=RLDQNConfig)


@dataclass
class RLEnvironmentConfig:
    max_steps_per_episode: int = 100
    observation_history_size: int = 10
    diversity_update_interval: int = 1
    diversity_sample_size: int | None = None
    action_id_map: dict[str, int] = field(default_factory=dict)
    render_mode: str | None = None


@dataclass
class RLRewardConfig:
    fitness_weight: float = 10.0
    diversity_weight: float = 1.0
    time_weight: float = 0.01
    normalize: bool = False


@dataclass
class RLTrainingConfig:
    total_timesteps: int = 100000
    checkpoint_interval: int = 10000
    evaluation_interval: int = 5000
    tensorboard_log: str = "logs/tensorboard"
    checkpoint_dir: str = "models/rl_agents/checkpoints"
    save_dir: str = "models/rl_agents"
    verbose: int = 1
    curriculum: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RLInferenceConfig:
    batch_prediction: bool = False
    timeout_ms: int = 10
    fallback_on_timeout: bool = True
    cache_predictions: bool = False


@dataclass
class RLHybridConfig:
    mode: str = "rl_primary"
    fallback_strategy: str = "random"
    rl_probability: float = 0.8
    enable_action_masking: bool = True


@dataclass
class RLEvaluationConfig:
    baseline_strategies: list[str] = field(
        default_factory=lambda: ["random", "round_robin", "greedy", "fixed_priority"]
    )
    num_evaluation_episodes: int = 10
    save_metrics: bool = True
    metrics_dir: str = "output/rl_metrics"


@dataclass
class RLLoggingConfig:
    log_heuristic_usage: bool = True
    log_rewards: bool = True
    log_state_transitions: bool = False
    log_inference_time: bool = True


@dataclass
class RLConfig:
    enabled: bool = False
    mode: str = "disabled"
    environment: RLEnvironmentConfig = field(default_factory=RLEnvironmentConfig)
    reward: RLRewardConfig = field(default_factory=RLRewardConfig)
    agent: RLAgentConfig = field(default_factory=RLAgentConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    inference: RLInferenceConfig = field(default_factory=RLInferenceConfig)
    hybrid: RLHybridConfig = field(default_factory=RLHybridConfig)
    evaluation: RLEvaluationConfig = field(default_factory=RLEvaluationConfig)
    logging: RLLoggingConfig = field(default_factory=RLLoggingConfig)


# =============================================================================
# PARALLEL, PERFORMANCE, METRICS, IO, EXPORT, GPU
# =============================================================================


@dataclass
class ParallelConfig:
    use_multiprocessing: bool = True
    num_workers: int | None = None


@dataclass
class IOConfig:
    data_dir: str = "data"
    output_dir: str = "output"


# =============================================================================
# CALENDAR EXPORT CONSTANTS (used by io/export/exporter.py)
# =============================================================================

EXCAL_QUANTUM_MINUTES: int = 15
EXCAL_START_HOUR: int = 7
EXCAL_END_HOUR: int = 20
EXCAL_DEFAULT_OUTPUT_PDF: str = "calendar.pdf"


# =============================================================================
# MASTER CONFIG
# =============================================================================


@dataclass
class Config:
    """Master configuration for Schedule Engine.

    Create directly in run files:

        config = Config()
        config.ga.ngen = 200
        config.ga.pop_size = 100
        config.repair.enabled = False

    Or with nested construction:

        config = Config(
            ga=GAConfig(ngen=200, pop_size=100),
            repair=RepairConfig(enabled=False),
        )
    """

    name: str = "default"
    environment: str = "test"

    ga: GAConfig = field(default_factory=GAConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    lns: LNSConfig = field(default_factory=LNSConfig)
    enhancements: EnhancementConfig = field(default_factory=EnhancementConfig)
    heuristics: HeuristicsConfig = field(default_factory=HeuristicsConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    hard_constraints: HardConstraintsConfig = field(
        default_factory=HardConstraintsConfig
    )
    soft_constraints: SoftConstraintsConfig = field(
        default_factory=SoftConstraintsConfig
    )
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    io: IOConfig = field(default_factory=IOConfig)

    # Derived at runtime by load_input_data / load_data
    cohort_pairs: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable configuration summary."""
        parallel_mode = "enabled" if self.parallel.use_multiprocessing else "disabled"
        parallel_workers = self.parallel.num_workers or "auto"
        repair_mode = "enabled" if self.repair.enabled else "disabled"

        return (
            f"Config: {self.name} ({self.environment})\n"
            f"  GA: {self.ga.ngen} gen x {self.ga.pop_size} pop | "
            f"cx={self.ga.cxpb} mut={self.ga.mutpb}\n"
            f"  Parallel: {parallel_mode} ({parallel_workers} workers)\n"
            f"  Repair: {repair_mode} (max {self.repair.max_iterations} iter)\n"
            f"  Data: {self.io.data_dir}/ -> {self.io.output_dir}/"
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Reconstruct a Config from a dict (e.g. from dataclasses.asdict).

        Handles nested sub-config dicts automatically.
        """
        import dataclasses as _dc

        def _rebuild(klass: type, data: dict[str, Any]) -> Any:
            field_types = {f.name: f.type for f in _dc.fields(klass)}
            kwargs: dict[str, Any] = {}
            for key, val in data.items():
                if key in field_types and isinstance(val, dict):
                    # Resolve the actual class from the type annotation
                    ft = field_types[key]
                    if isinstance(ft, str):
                        ft = globals().get(ft, ft)
                    if isinstance(ft, type) and _dc.is_dataclass(ft):
                        kwargs[key] = _rebuild(ft, val)
                    else:
                        kwargs[key] = val
                else:
                    kwargs[key] = val
            return klass(**kwargs)

        return _rebuild(cls, d)
