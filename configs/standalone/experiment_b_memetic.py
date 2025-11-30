"""
Experiment B: NSGA-II + Memetic Local Search

Standalone configuration with memetic search enabled.
Test vs Prod: Only ngen and pop_size differ.
"""

from dataclasses import dataclass


@dataclass
class ExperimentBConfig:
    """Experiment B configuration (NSGA-II + Memetic)."""

    # === METADATA ===
    name: str = "experiment-b-memetic"
    experiment_id: str = "B"
    environment: str = "test"
    notes: str = "NSGA-II + Memetic local search"
    output_subdir: str = "b-nsga-memetic"

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

    # === KILLSWITCHES (Memetic ON, others OFF) ===
    repair_enabled: bool = True  # Memetic uses repair
    heuristics_master_enabled: bool = False
    heuristics_adaptive_priority_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # === PARALLEL PROCESSING ===
    use_multiprocessing: bool = True
    num_workers: int | None = None

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


def get_test_config() -> ExperimentBConfig:
    """Get test profile config (30 gens, 10 pop, ~3-7 min)."""
    return ExperimentBConfig(ngen=30, pop_size=10, environment="test")


def get_prod_config() -> ExperimentBConfig:
    """Get production profile config (2000 gens, 200 pop, ~2-4 hours)."""
    return ExperimentBConfig(ngen=2000, pop_size=200, environment="prod")
