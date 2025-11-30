"""
Heuristic Test: Kempe Chain (Improvement)

Standalone configuration testing only kempe_chain heuristic.
Everything else is OFF by default.

Test vs Prod: Only ngen and pop_size differ.
"""

from dataclasses import dataclass


@dataclass
class KempeChainConfig:
    """Kempe Chain heuristic test configuration."""

    # === METADATA ===
    name: str = "test-kempe-chain"
    experiment_id: str = "F"
    environment: str = "test"  # Will be overridden by get_test/prod_config()
    notes: str = "Testing kempe_chain heuristic only"
    output_subdir: str = "f-improvement"

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

    # === KILLSWITCHES ===
    repair_enabled: bool = False
    heuristics_master_enabled: bool = True  # Enable heuristics subsystem
    heuristics_adaptive_priority_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # === INDIVIDUAL HEURISTIC TOGGLES (ALL OFF except kempe_chain) ===
    # Construction (all OFF)
    heuristic_largest_degree_first: bool = False
    heuristic_most_constrained_first: bool = False
    heuristic_earliest_deadline_first: bool = False

    # Perturbation (all OFF)
    heuristic_random_swap: bool = False
    heuristic_temporal_shift: bool = False
    heuristic_room_shuffle: bool = False
    heuristic_instructor_reassign: bool = False
    heuristic_multi_perturbation: bool = False

    # Improvement (ONLY kempe_chain ON)
    heuristic_kempe_chain: bool = True  # ← TARGET HEURISTIC
    heuristic_ejection_chain: bool = False
    heuristic_variable_depth_search: bool = False

    # Diversity (all OFF)
    heuristic_distance_preserving_crossover: bool = False
    heuristic_crowding_mutation: bool = False
    heuristic_niching_selection: bool = False
    heuristic_adaptive_diversity_maintenance: bool = False

    # Meta (all OFF)
    heuristic_variable_neighborhood_descent: bool = False
    heuristic_iterated_local_search: bool = False
    heuristic_adaptive_large_neighborhood: bool = False
    heuristic_guided_local_search: bool = False

    # Repair (all OFF)
    heuristic_exhaustive_repair: bool = False
    heuristic_greedy_repair: bool = False
    heuristic_igls_repair: bool = False
    heuristic_lns_repair: bool = False
    heuristic_memetic_repair: bool = False
    heuristic_selective_repair: bool = False

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


# Profile-specific factory functions
def get_test_config() -> KempeChainConfig:
    """Get test profile config (30 gens, 10 pop, ~2-5 min)."""
    return KempeChainConfig(
        ngen=30,
        pop_size=10,
        environment="test",
    )


def get_prod_config() -> KempeChainConfig:
    """Get production profile config (2000 gens, 200 pop, ~1-3 hours)."""
    return KempeChainConfig(
        ngen=2000,
        pop_size=200,
        environment="prod",
    )
