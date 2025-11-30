"""Test: most_constrained_first (Construction)"""

from dataclasses import dataclass

from configs.profiles import TestConfig


@dataclass
class SingleHeuristicTestConfig(TestConfig):
    name: str = "test-most-constrained-first"
    experiment_id: str = "F"
    output_subdir: str = "f-construction"  # Category-specific output folder
    heuristics_master_enabled: bool = True
    repair_enabled: bool = False

    # ENABLED: most_constrained_first
    heuristic_most_constrained_first: bool = True

    # All others disabled
    heuristic_largest_degree_first: bool = False
    heuristic_earliest_deadline_first: bool = False
    heuristic_random_swap: bool = False
    heuristic_temporal_shift: bool = False
    heuristic_room_shuffle: bool = False
    heuristic_instructor_reassign: bool = False
    heuristic_multi_perturbation: bool = False
    heuristic_kempe_chain: bool = False
    heuristic_ejection_chain: bool = False
    heuristic_variable_depth_search: bool = False
    heuristic_distance_preserving_crossover: bool = False
    heuristic_crowding_mutation: bool = False
    heuristic_niching_selection: bool = False
    heuristic_adaptive_diversity_maintenance: bool = False
    heuristic_variable_neighborhood_descent: bool = False
    heuristic_iterated_local_search: bool = False
    heuristic_adaptive_large_neighborhood: bool = False
    heuristic_guided_local_search: bool = False
    heuristic_exhaustive_repair: bool = False
    heuristic_greedy_repair: bool = False
    heuristic_igls_repair: bool = False
    heuristic_lns_repair: bool = False
    heuristic_memetic_repair: bool = False
    heuristic_selective_repair: bool = False

    notes: str = "Testing most_constrained_first only"
