"""
Experiment A: Pure NSGA-II Baseline

Standalone configuration with no shared dependencies.
Everything defaults to OFF unless explicitly enabled here.

Test vs Prod: Only ngen and pop_size differ.
"""

from dataclasses import dataclass


@dataclass
class ExperimentAConfig:
    """Experiment A configuration (profile selected via ngen/pop_size)."""

    # === METADATA ===
    name: str = "experiment-a-baseline"
    experiment_id: str = "A"
    environment: str = "test"  # Will be overridden by get_test/prod_config()
    notes: str = "Pure NSGA-II baseline"
    output_subdir: str = "a-baseline-nsga-only"

    # === GA PARAMETERS (profile-dependent) ===
    ngen: int = 30  # Test: 30, Prod: 2000
    pop_size: int = 10  # Test: 10, Prod: 200

    # === GA PARAMETERS (shared) ===
    cxpb: float = 0.70
    mutpb: float = 0.20
    elite_preservation: bool = True
    elite_size: float = 0.05
    tournament_size: int = 2

    # === POPULATION STRATEGY ===
    population_strategy: str = "hybrid"
    greedy_percentage: float = 0.25
    smart_percentage: float = 0.50
    random_percentage: float = 0.25

    # === KILLSWITCHES (ALL OFF - pure NSGA-II) ===
    repair_enabled: bool = False
    heuristics_master_enabled: bool = False
    heuristics_adaptive_priority_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # === PARALLEL PROCESSING ===
    use_multiprocessing: bool = True
    num_workers: int | None = None  # None = CPU count

    # === METRICS ===
    advanced_metrics_frequency: int = 5
    performance_profiling_enabled: bool = True

    # === PATHS ===
    data_dir: str = "data"
    output_dir: str = "output"

    # === CONSTRAINT WEIGHTS ===
    hard_weight: float = -1.0
    soft_weight: float = -0.01

    # === SEED ===
    seed: int | None = None


# Profile-specific factory functions
def get_test_config() -> ExperimentAConfig:
    """Get test profile config (30 gens, 10 pop, ~2-5 min)."""
    return ExperimentAConfig(
        ngen=30,
        pop_size=10,
        environment="test",
    )


def get_prod_config() -> ExperimentAConfig:
    """Get production profile config (2000 gens, 200 pop, ~3-5 hours)."""
    return ExperimentAConfig(
        ngen=2000,
        pop_size=200,
        environment="prod",
    )
