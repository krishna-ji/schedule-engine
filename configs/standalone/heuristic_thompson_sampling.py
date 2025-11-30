"""
Heuristic Test: thompson_sampling (Meta-Heuristic)

Standalone configuration for isolated thompson_sampling testing.
Test vs Prod: Only ngen and pop_size differ.
"""

from dataclasses import dataclass


@dataclass
class ThompsonSamplingConfig:
    """Thompson sampling heuristic test config."""

    # === METADATA ===
    name: str = "heuristic-thompson-sampling"
    experiment_id: str = "F-META-02"
    environment: str = "test"
    notes: str = "Isolated test: thompson_sampling"
    output_subdir: str = "f-meta"

    # === GA PARAMETERS (profile-dependent) ===
    ngen: int = 30
    pop_size: int = 10

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

    # === KILLSWITCHES ===
    repair_enabled: bool = False
    heuristics_master_enabled: bool = True
    heuristics_adaptive_priority_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # === HEURISTIC TOGGLES (Only thompson_sampling ON) ===
    heuristic_greedy_sequential: bool = False
    heuristic_constraint_guided: bool = False
    heuristic_kempe_chain: bool = False
    heuristic_instructor_swap: bool = False
    heuristic_room_swap: bool = False
    heuristic_time_swap: bool = False
    heuristic_ejection_chain: bool = False
    heuristic_group_relocate: bool = False
    heuristic_hillclimb: bool = False
    heuristic_tabu_search: bool = False
    heuristic_simulated_annealing: bool = False
    heuristic_vnd: bool = False
    heuristic_crowding_distance: bool = False
    heuristic_rank_distance: bool = False
    heuristic_entropy_preserving: bool = False
    heuristic_ucb_selection: bool = False
    heuristic_thompson_sampling: bool = True
    heuristic_epsilon_greedy: bool = False
    heuristic_slot_repair: bool = False
    heuristic_feasibility_pump: bool = False
    heuristic_constraint_propagation: bool = False
    heuristic_resource_balancing: bool = False
    heuristic_workload_distribution: bool = False
    heuristic_schedule_compaction: bool = False
    heuristic_crossover_repair: bool = False

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


def get_test_config() -> ThompsonSamplingConfig:
    """Get test profile config (30 gens, 10 pop)."""
    return ThompsonSamplingConfig(ngen=30, pop_size=10, environment="test")


def get_prod_config() -> ThompsonSamplingConfig:
    """Get production profile config (2000 gens, 200 pop)."""
    return ThompsonSamplingConfig(ngen=2000, pop_size=200, environment="prod")
