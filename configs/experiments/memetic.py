"""
Experiment B: NSGA-II + Memetic Local Search

Mode B: NSGA-II with memetic local search on elite solutions.

Killswitches:
- repair.enabled = True (memetic mode)
- heuristics.master_enabled = False
- lns.enabled = False
- rl.enabled = False
- enhancements.master_enabled = False
"""

from __future__ import annotations

from dataclasses import dataclass, field

from configs.profiles import ProdConfig, TestConfig


@dataclass
class MemeticBaseConfig:
    """Shared configuration for Mode B memetic profiles."""

    # === REPAIR: Enable memetic mode ===
    repair_enabled: bool = True
    repair_memetic_mode: bool = True
    repair_apply_after_mutation: bool = True
    repair_apply_after_crossover: bool = False
    repair_elite_percentage: float = 0.20
    repair_max_iterations: int = 100
    repair_heuristics_overrides: dict[str, dict[str, int | bool]] = field(
        default_factory=lambda: {
            "repair_room_overlap_reassign": {"enabled": True, "priority": 3}
        }
    )

    # === MUTATION: Enable constraint-guided ===
    use_constraint_guided_mutation: bool = True

    # === KILLSWITCHES: Disable other enhancements ===
    heuristics_master_enabled: bool = False
    heuristics_adaptive_priority_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # === POPULATION STRATEGY: Hybrid initialization ===
    population_strategy: str = "hybrid"
    greedy_percentage: float = 0.25
    smart_percentage: float = 0.50
    random_percentage: float = 0.25

    # === NOTES ===
    notes: str = "NSGA-II + memetic local search on elite solutions"


@dataclass
class MemeticTestConfig(MemeticBaseConfig, TestConfig):
    """Mode B memetic (test profile)."""

    name: str = "memetic-test"
    experiment_id: str = "B"


@dataclass
class MemeticProdConfig(MemeticBaseConfig, ProdConfig):
    """Mode B memetic (production profile)."""

    name: str = "memetic-prod"
    experiment_id: str = "B"
    notes: str = "NSGA-II + memetic local search for thesis (production)"


# ============================================
# LEGACY COMPATIBILITY
# ============================================

EXPERIMENT_ID = "B"
EXPERIMENT_NAME = "Memetic NSGA-II"
EXPERIMENT_DESCRIPTION = "NSGA-II with memetic local search on elite solutions"

KILLSWITCHES = {
    "repair.enabled": True,
    "repair.memetic_mode": True,
    "repair.heuristics.repair_room_overlap_reassign.enabled": True,
    "repair.heuristics.repair_room_overlap_reassign.priority": 3,
    "heuristics.master_enabled": False,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": False,
}


def get_test_config() -> MemeticTestConfig:
    """Get test profile config (30 gens, 10 pop)."""
    return MemeticTestConfig()


def get_prod_config(**overrides) -> MemeticProdConfig:
    """
    Get production profile config (2000 gens, 200 pop).

    Args:
        **overrides: Optional field overrides

    Example:
        >>> config = get_prod_config(ngen=2500, name="thesis-memetic-r01")
    """
    return MemeticProdConfig(**overrides)


if __name__ == "__main__":
    # Test instantiation
    test_cfg = get_test_config()
    print(f"✓ {EXPERIMENT_NAME} (TEST)")
    print(f"  Generations: {test_cfg.ngen}")
    print(f"  Population: {test_cfg.pop_size}")
    print(f"  Repair: {test_cfg.repair_enabled}")
    print(f"  Memetic mode: {test_cfg.repair_memetic_mode}")
    print(f"  Heuristics: {test_cfg.heuristics_master_enabled}")

    prod_cfg = get_prod_config()
    print(f"\n✓ {EXPERIMENT_NAME} (PROD)")
    print(f"  Generations: {prod_cfg.ngen}")
    print(f"  Population: {prod_cfg.pop_size}")
    print(f"  Elite percentage: {prod_cfg.repair_elite_percentage}")
