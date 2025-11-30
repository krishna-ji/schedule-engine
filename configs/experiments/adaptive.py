"""
Experiment D: Adaptive Heuristic Selection

Dataclass-based configuration for adaptive performance-based heuristic selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.base import BaseConfig
from configs.profiles import ProdConfig, TestConfig


@dataclass
class AdaptiveBaseConfig(BaseConfig):
    """Adaptive heuristic selection base settings."""

    # Repair system
    repair_enabled: bool = True
    repair_memetic_mode: bool = True

    # GA enhancements
    ga_use_adaptive_probabilities: bool = True

    # Heuristics (adaptive = performance-based selection)
    heuristics_master_enabled: bool = True
    heuristics_adaptive_priority_enabled: bool = True  # KEY: Adaptive selection
    heuristics_adaptive_priority_evaluation_window: int = 10
    heuristics_adaptive_priority_reorder_interval: int = 10

    # Enhancements
    enhancements_master_enabled: bool = True
    enhancements_hypermutation_enabled: bool = True

    # Disabled features
    lns_enabled: bool = False
    rl_enabled: bool = False


@dataclass
class AdaptiveTestConfig(TestConfig, AdaptiveBaseConfig):
    """Adaptive heuristics - test profile (30 gens, 10 pop)."""


@dataclass
class AdaptiveProdConfig(ProdConfig, AdaptiveBaseConfig):
    """Adaptive heuristics - production profile (2000 gens, 200 pop)."""


# Experiment metadata
EXPERIMENT_ID = "D"
EXPERIMENT_NAME = "Adaptive Heuristic Selection"
EXPERIMENT_DESCRIPTION = "NSGA-II + adaptive performance-based heuristic selection"

# Killswitches (explicit documentation)
KILLSWITCHES = {
    "repair.enabled": True,
    "repair.memetic_mode": True,
    "ga.use_adaptive_probabilities": True,
    "heuristics.master_enabled": True,
    "heuristics.adaptive_priority.enabled": True,  # KEY: Adaptive selection
    "heuristics.adaptive_priority.evaluation_window": 10,
    "heuristics.adaptive_priority.reorder_interval": 10,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": True,
    "enhancements.hypermutation.enabled": True,
}


if __name__ == "__main__":
    # Test the dataclass configs
    test_cfg = AdaptiveTestConfig()
    prod_cfg = AdaptiveProdConfig()

    print(f"✓ {EXPERIMENT_NAME}")
    print("\nTest Config:")
    print(f"  ngen={test_cfg.ngen}, pop={test_cfg.pop_size}")
    print(f"  repair={test_cfg.repair_enabled}")
    print(f"  adaptive_probabilities={test_cfg.ga_use_adaptive_probabilities}")
    print(f"  adaptive_priority={test_cfg.heuristics_adaptive_priority_enabled}")
    print(f"  hypermutation={test_cfg.enhancements_hypermutation_enabled}")

    print("\nProd Config:")
    print(f"  ngen={prod_cfg.ngen}, pop={prod_cfg.pop_size}")
    print(f"  total_evals={prod_cfg.total_evaluations:,}")
