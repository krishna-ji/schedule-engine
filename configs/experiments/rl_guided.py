"""
Experiment E: RL-Guided Hyper-Heuristic

Dataclass-based configuration for RL-guided heuristic selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.base import BaseConfig
from configs.profiles import ProdConfig, TestConfig


@dataclass
class RlGuidedBaseConfig(BaseConfig):
    """RL-guided heuristic selection base settings."""

    # Genetic operators
    use_constraint_guided_mutation: bool = True
    population_strategy: str = "hybrid"

    # Repair system
    repair_enabled: bool = True
    repair_memetic_mode: bool = True
    repair_apply_after_mutation: bool = True

    # GA enhancements
    ga_use_adaptive_probabilities: bool = True

    # Heuristics (RL takes over from adaptive)
    heuristics_master_enabled: bool = True
    heuristics_adaptive_priority_enabled: bool = False  # RL takes over
    heuristics_construction_largest_degree_first_enabled: bool = True
    heuristics_perturbation_random_swap_enabled: bool = True
    heuristics_improvement_kempe_chain_enabled: bool = True
    heuristics_meta_variable_neighborhood_descent_enabled: bool = True

    # LNS
    lns_enabled: bool = True

    # RL (KEY: Enabled)
    rl_enabled: bool = True
    rl_mode: str = "rl_primary"
    rl_hybrid_rl_probability: float = 0.8

    # Enhancements
    enhancements_master_enabled: bool = True
    enhancements_memetic_mode: bool = True
    enhancements_hypermutation_enabled: bool = True
    enhancements_population_restart_enabled: bool = True


@dataclass
class RlGuidedTestConfig(RlGuidedBaseConfig, TestConfig):
    """RL-guided - test profile (30 gens, 10 pop)."""


@dataclass
class RlGuidedProdConfig(RlGuidedBaseConfig, ProdConfig):
    """RL-guided - production profile (2000 gens, 200 pop)."""


# Experiment metadata
EXPERIMENT_ID = "E"
EXPERIMENT_NAME = "RL-Guided Hyper-Heuristic"
EXPERIMENT_DESCRIPTION = "Full NSGA-II + RL-guided heuristic selection"

# Killswitches (explicit documentation)
KILLSWITCHES = {
    "use_constraint_guided_mutation": True,
    "population_strategy": "hybrid",
    "repair.enabled": True,
    "repair.memetic_mode": True,
    "repair.apply_after_mutation": True,
    "ga.use_adaptive_probabilities": True,
    "heuristics.master_enabled": True,
    "heuristics.adaptive_priority.enabled": False,  # RL takes over
    "heuristics.construction.largest_degree_first.enabled": True,
    "heuristics.perturbation.random_swap.enabled": True,
    "heuristics.improvement.kempe_chain.enabled": True,
    "heuristics.meta.variable_neighborhood_descent.enabled": True,
    "lns.enabled": True,
    "rl.enabled": True,  # KEY: RL enabled
    "rl.mode": "rl_primary",
    "rl.hybrid.rl_probability": 0.8,
    "enhancements.master_enabled": True,
    "enhancements.memetic_mode": True,
    "enhancements.hypermutation.enabled": True,
    "enhancements.population_restart.enabled": True,
}


if __name__ == "__main__":
    # Test the dataclass configs
    test_cfg = RlGuidedTestConfig()
    prod_cfg = RlGuidedProdConfig()

    print(f"✓ {EXPERIMENT_NAME}")
    print("\nTest Config:")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  repair={test_cfg.repair_enabled}")
    print(f"  rl_enabled={test_cfg.rl_enabled}, mode={test_cfg.rl_mode}")
    print(f"  lns={test_cfg.lns_enabled}")
    print(f"  enhancements={test_cfg.enhancements_master_enabled}")

    print("\nProd Config:")
    print(f"  ngen={prod_cfg.ngen}, pop={prod_cfg.pop_size}")
    print(f"  total_evals={prod_cfg.total_evaluations:,}")
