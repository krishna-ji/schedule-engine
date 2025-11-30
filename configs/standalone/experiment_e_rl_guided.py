"""
Experiment E: NSGA-II + RL-Guided Control

Standalone configuration with RL-guided heuristic selection.
Test vs Prod: Only ngen and pop_size differ.
"""

from dataclasses import dataclass


@dataclass
class ExperimentEConfig:
    """Experiment E configuration (RL-guided)."""

    # === METADATA ===
    name: str = "experiment-e-rl-guided"
    experiment_id: str = "E"
    environment: str = "test"
    notes: str = "NSGA-II + RL-guided control"
    output_subdir: str = "e-rl-guided"

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

    # === KILLSWITCHES (RL + All enhancements ON) ===
    repair_enabled: bool = True
    heuristics_master_enabled: bool = True
    heuristics_adaptive_priority_enabled: bool = False  # RL controls selection
    lns_enabled: bool = False
    rl_enabled: bool = True  # RL-guided
    enhancements_master_enabled: bool = True

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


def get_test_config() -> ExperimentEConfig:
    """Get test profile config (30 gens, 10 pop, ~10-20 min)."""
    return ExperimentEConfig(ngen=30, pop_size=10, environment="test")


def get_prod_config() -> ExperimentEConfig:
    """Get production profile config (2000 gens, 200 pop, ~4-6 hours)."""
    return ExperimentEConfig(ngen=2000, pop_size=200, environment="prod")
