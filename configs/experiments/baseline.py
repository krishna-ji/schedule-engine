"""
Experiment A: Pure NSGA-II Baseline

Mode A: Minimal NSGA-II with all enhancements disabled.
Serves as baseline for comparing other experiments.

Killswitches:
- repair.enabled = False
- heuristics.master_enabled = False
- lns.enabled = False
- rl.enabled = False
- enhancements.master_enabled = False
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.profiles import ProdConfig, TestConfig


@dataclass
class BaselineBaseConfig:
    """Shared configuration for Mode A baseline profiles."""

    # === KILLSWITCHES: Disable ALL enhancements ===
    repair_enabled: bool = False
    heuristics_master_enabled: bool = False
    heuristics_adaptive_priority_enabled: bool = False
    lns_enabled: bool = False
    rl_enabled: bool = False
    enhancements_master_enabled: bool = False

    # === SOFT CONSTRAINTS: Disable break placement ===
    enforce_break_placement: bool = False  # Not used in baseline

    # === MUTATION: Pure random (no constraint guidance) ===
    use_constraint_guided_mutation: bool = False

    # === POPULATION STRATEGY: Random only ===
    population_strategy: str = "random"

    # === NOTES ===
    notes: str = "Pure NSGA-II baseline (no repairs, no heuristics, no enhancements)"


@dataclass
class BaselineTestConfig(BaselineBaseConfig, TestConfig):
    """Mode A baseline (test profile)."""

    name: str = "baseline-test"
    experiment_id: str = "A"


@dataclass
class BaselineProdConfig(BaselineBaseConfig, ProdConfig):
    """Mode A baseline (production profile)."""

    name: str = "baseline-prod"
    experiment_id: str = "A"
    notes: str = "Pure NSGA-II baseline for thesis (production)"


# ============================================
# LEGACY COMPATIBILITY (Optional)
# ============================================
# For backward compatibility with existing launcher system

EXPERIMENT_ID = "A"
EXPERIMENT_NAME = "Pure NSGA-II Baseline"
EXPERIMENT_DESCRIPTION = "Minimal NSGA-II (no repairs, no heuristics, no enhancements)"

KILLSWITCHES = {
    "repair.enabled": False,
    "heuristics.master_enabled": False,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": False,
}


def get_test_config() -> BaselineTestConfig:
    """Get test profile config (30 gens, 10 pop)."""
    return BaselineTestConfig()


def get_prod_config(**overrides) -> BaselineProdConfig:
    """
    Get production profile config (2000 gens, 200 pop).

    Args:
        **overrides: Optional field overrides (e.g., ngen=2500, name="thesis-r01")

    Example:
        >>> config = get_prod_config(ngen=2500, name="thesis-baseline-r01")
    """
    return BaselineProdConfig(**overrides)


if __name__ == "__main__":
    # Test instantiation
    test_cfg = get_test_config()
    print(f"✓ {EXPERIMENT_NAME} (TEST)")
    print(f"  Generations: {test_cfg.ngen}")
    print(f"  Population: {test_cfg.pop_size}")
    print(f"  Repair: {test_cfg.repair_enabled}")
    print(f"  Heuristics: {test_cfg.heuristics_master_enabled}")
    print(f"  Total evaluations: {test_cfg.total_evaluations}")

    print(f"\n✓ {EXPERIMENT_NAME} (PROD)")
    prod_cfg = get_prod_config()
    print(f"  Generations: {prod_cfg.ngen}")
    print(f"  Population: {prod_cfg.pop_size}")
    print(f"  Total evaluations: {prod_cfg.total_evaluations}")

    # Test custom overrides
    print("\n✓ Custom production run")
    custom_cfg = get_prod_config(ngen=2500, name="thesis-baseline-r01")
    print(f"  Name: {custom_cfg.name}")
    print(f"  Generations: {custom_cfg.ngen}")
