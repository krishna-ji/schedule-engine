"""
Experiment B: NSGA-II + Memetic Local Search

Mode B: NSGA-II with memetic local search on elite solutions.

Repair System:
- Uses ALL 7 base repair operators (HC1-HC5, HC8×2, HC4)
- Memetic mode: Deep repair on top 20% elite individuals
- Post-mutation repair: Quick 2-iteration fixes
- Selective mode: 3-4× faster (targets violated genes only)
- Max iterations: 100 for elite, 2 for post-mutation

Constraint Coverage:
  ✅ HC1 (student_group_exclusivity) → repair_group_overlaps
  ✅ HC2 (instructor_exclusivity) → repair_instructor_conflicts
  ✅ HC3 (instructor_qualifications) → repair_instructor_qualifications
  ✅ HC4 (room_suitability) → repair_room_type_mismatches
  ✅ HC5 (instructor_time_availability) → repair_instructor_availability
  ❌ HC6 (room_time_availability) - Not needed (rooms always available)
  ❌ HC7 (course_completeness) - Not needed (structural integrity)
  ✅ HC8 (room_exclusivity) → repair_room_conflicts + repair_room_overlap_reassign

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

    # === POPULATION STRATEGY: Random only (no greedy/smart) ===
    population_strategy: str = "random"
    greedy_percentage: float = 0.00
    smart_percentage: float = 0.00
    random_percentage: float = 1.00

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
