"""
Experiment C: Round-Robin Heuristics

Dataclass-based configuration for round-robin heuristic selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.base import BaseConfig
from configs.profiles import ProdConfig, TestConfig


@dataclass
class RoundRobinBaseConfig(BaseConfig):
    """Round-robin heuristic selection base settings."""

    # Repair system
    repair_enabled: bool = True
    repair_memetic_mode: bool = True

    # Heuristics (round-robin = fixed rotation, NOT adaptive)
    heuristics_master_enabled: bool = True
    heuristics_adaptive_priority_enabled: bool = False  # Fixed rotation
    heuristics_construction_largest_degree_first_enabled: bool = True
    heuristics_perturbation_random_swap_enabled: bool = True

    # Enhancements
    enhancements_master_enabled: bool = True

    # Disabled features
    lns_enabled: bool = False
    rl_enabled: bool = False


@dataclass
class RoundRobinTestConfig(TestConfig, RoundRobinBaseConfig):
    """Round-robin heuristics - test profile (30 gens, 10 pop)."""


@dataclass
class RoundRobinProdConfig(ProdConfig, RoundRobinBaseConfig):
    """Round-robin heuristics - production profile (2000 gens, 200 pop)."""


# Experiment metadata
EXPERIMENT_ID = "C"
EXPERIMENT_NAME = "Round-Robin Heuristics"
EXPERIMENT_DESCRIPTION = "NSGA-II + round-robin heuristic selection"

# Killswitches (explicit documentation)
KILLSWITCHES = {
    "repair.enabled": True,
    "repair.memetic_mode": True,
    "heuristics.master_enabled": True,
    "heuristics.adaptive_priority.enabled": False,  # Fixed rotation, not adaptive
    "heuristics.construction.largest_degree_first.enabled": True,
    "heuristics.perturbation.random_swap.enabled": True,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": True,
}


if __name__ == "__main__":
    # Test the dataclass configs
    test_cfg = RoundRobinTestConfig()
    prod_cfg = RoundRobinProdConfig()

    print(f"✓ {EXPERIMENT_NAME}")
    print("\nTest Config:")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  repair={test_cfg.repair_enabled}, memetic={test_cfg.repair_memetic_mode}")
    print(f"  heuristics={test_cfg.heuristics_master_enabled}")
    print(f"  adaptive={test_cfg.heuristics_adaptive_priority_enabled}")

    print("\nProd Config:")
    print(f"  ngen={prod_cfg.ngen}, pop={prod_cfg.pop_size}")
    print(f"  total_evals={prod_cfg.total_evaluations:,}")
