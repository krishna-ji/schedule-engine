"""
Experiment F: Individual Heuristic Testing

Mode F: Test individual heuristics in isolation.
Enable/disable specific heuristics via config flags.

Usage:
    # Enable only specific heuristics by setting them to True
    # All others default to False for isolated testing
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.profiles import ProdConfig, TestConfig

EXPERIMENT_NAME = "Experiment F: Heuristic Testing"
EXPERIMENT_DESCRIPTION = "Test individual heuristics in isolation"


@dataclass
class HeuristicTestingBaseConfig:
    """Shared knobs for Mode F heuristic testing profiles."""

    # === KILLSWITCHES: Enable heuristics subsystem ===
    repair_enabled: bool = False  # Keep repair disabled by default
    heuristics_master_enabled: bool = True  # Enable heuristics
    heuristics_adaptive_priority_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # === POPULATION STRATEGY: Random only (no greedy/smart) ===
    population_strategy: str = "random"

    # === INDIVIDUAL HEURISTIC TOGGLES ===
    # Construction Heuristics (3 total)
    heuristic_largest_degree_first: bool = False
    heuristic_most_constrained_first: bool = False
    heuristic_earliest_deadline_first: bool = False

    # Perturbation Heuristics (5 total)
    heuristic_random_swap: bool = False
    heuristic_temporal_shift: bool = False
    heuristic_room_shuffle: bool = False
    heuristic_instructor_reassign: bool = False
    heuristic_multi_perturbation: bool = False

    # Improvement Heuristics (3 total)
    heuristic_kempe_chain: bool = False
    heuristic_ejection_chain: bool = False
    heuristic_variable_depth_search: bool = False

    # Diversity Heuristics (4 total)
    heuristic_distance_preserving_crossover: bool = False
    heuristic_crowding_mutation: bool = False
    heuristic_niching_selection: bool = False
    heuristic_adaptive_diversity_maintenance: bool = False

    # Meta Heuristics (4 total)
    heuristic_variable_neighborhood_descent: bool = False
    heuristic_iterated_local_search: bool = False
    heuristic_adaptive_large_neighborhood: bool = False
    heuristic_guided_local_search: bool = False

    # Repair Heuristics (6 total)
    heuristic_exhaustive_repair: bool = False
    heuristic_greedy_repair: bool = False
    heuristic_igls_repair: bool = False
    heuristic_lns_repair: bool = False
    heuristic_memetic_repair: bool = False
    heuristic_selective_repair: bool = False

    # === NOTES ===
    notes: str = (
        "Heuristic testing mode - enable individual heuristics for isolated evaluation"
    )

    def get_enabled_heuristic_name(self) -> str | None:
        """
        Extract the name of the single enabled heuristic.

        Returns:
            Heuristic name (e.g., 'largest-degree-first') or None if multiple/none enabled.
        """
        heuristic_fields = [
            name
            for name in dir(self)
            if name.startswith("heuristic_") and isinstance(getattr(self, name), bool)
        ]

        enabled = [name for name in heuristic_fields if getattr(self, name)]

        if len(enabled) == 1:
            # Convert heuristic_largest_degree_first -> largest-degree-first
            return enabled[0].replace("heuristic_", "").replace("_", "-")
        return None


@dataclass
class HeuristicTestingTestConfig(HeuristicTestingBaseConfig, TestConfig):
    """Heuristic testing - test profile (30 gens, 10 pop)."""

    pass


@dataclass
class HeuristicTestingProdConfig(HeuristicTestingBaseConfig, ProdConfig):
    """Heuristic testing - production profile (2000 gens, 200 pop)."""

    pass


# Experiment instances
heuristic_testing_test = HeuristicTestingTestConfig()
heuristic_testing_prod = HeuristicTestingProdConfig()


# LEGACY COMPATIBILITY (Optional)
# ============================================
# For backward compatibility with existing launcher system

EXPERIMENT_ID = "F"

KILLSWITCHES = {
    "repair.enabled": False,
    "heuristics.master_enabled": True,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": False,
}


def get_test_config() -> HeuristicTestingTestConfig:
    """Get test profile config (30 gens, 10 pop)."""
    return HeuristicTestingTestConfig()


def get_prod_config() -> HeuristicTestingProdConfig:
    """Get production profile config (2000 gens, 200 pop)."""
    return HeuristicTestingProdConfig()


# Convenience config loader
def get_heuristic_testing_config(
    profile: str = "test",
) -> HeuristicTestingTestConfig | HeuristicTestingProdConfig:
    """Load heuristic testing config for given profile."""
    if profile == "test":
        return HeuristicTestingTestConfig()
    elif profile == "prod":
        return HeuristicTestingProdConfig()
    else:
        raise ValueError(f"Unknown profile: {profile}")
