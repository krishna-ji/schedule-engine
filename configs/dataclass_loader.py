"""
Dataclass-based config loader.

Provides clean Python-native configs with IDE support and type safety.

Usage:
    # Direct instantiation
    >>> from configs.experiments import BaselineTestConfig
    >>> config = BaselineTestConfig()
    >>> pydantic_config = config.to_pydantic()

    # Loader helper
    >>> from configs.dataclass_loader import load_experiment_config
    >>> config = load_experiment_config("baseline", "test")

    # Custom overrides
    >>> config = load_experiment_config("baseline", "prod", ngen=2500, name="thesis-r01")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.models import Config

from configs.base import BaseConfig


def load_experiment_config(
    experiment: str,
    profile: str,
    **overrides,
) -> Config:
    """
    Load experiment config using dataclass hierarchy.

    Args:
        experiment: Experiment name (baseline, memetic, roundrobin, adaptive, rl_guided)
        profile: Profile name (test, prod)
        **overrides: Optional field overrides (e.g., ngen=2500, name="custom")

    Returns:
        Pydantic Config instance (validated)

    Example:
        >>> config = load_experiment_config("baseline", "test")
        >>> config = load_experiment_config("baseline", "prod", ngen=2500)
    """
    config_class = _get_config_class(experiment, profile)
    dataclass_config = config_class(**overrides)
    return dataclass_config.to_pydantic()


def _get_config_class(experiment: str, profile: str) -> type[BaseConfig]:
    """Get dataclass config class for experiment+profile combination."""

    # Registry mapping
    registry_map = {
        ("baseline", "test"): "configs.experiments.baseline:BaselineTestConfig",
        ("baseline", "prod"): "configs.experiments.baseline:BaselineProdConfig",
        ("memetic", "test"): "configs.experiments.memetic:MemeticTestConfig",
        ("memetic", "prod"): "configs.experiments.memetic:MemeticProdConfig",
        ("roundrobin", "test"): "configs.experiments.roundrobin:RoundRobinTestConfig",
        ("roundrobin", "prod"): "configs.experiments.roundrobin:RoundRobinProdConfig",
        ("adaptive", "test"): "configs.experiments.adaptive:AdaptiveTestConfig",
        ("adaptive", "prod"): "configs.experiments.adaptive:AdaptiveProdConfig",
        ("rl", "test"): "configs.experiments.rl_guided:RlGuidedTestConfig",
        ("rl", "prod"): "configs.experiments.rl_guided:RlGuidedProdConfig",
    }

    key = (experiment.lower(), profile.lower())
    class_path = registry_map.get(key)

    if not class_path:
        raise ValueError(
            f"Unknown experiment+profile: {experiment}/{profile}. "
            f"Available: {list(registry_map.keys())}"
        )

    # Import dynamically
    module_path, class_name = class_path.split(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ============================================
# CONVENIENCE HELPERS
# ============================================


def baseline_test(**overrides) -> Config:
    """Shortcut: BaselineTestConfig → Pydantic Config."""
    return load_experiment_config("baseline", "test", **overrides)


def baseline_prod(**overrides) -> Config:
    """Shortcut: BaselineProdConfig → Pydantic Config."""
    return load_experiment_config("baseline", "prod", **overrides)


def memetic_test(**overrides) -> Config:
    """Shortcut: MemeticTestConfig → Pydantic Config."""
    return load_experiment_config("memetic", "test", **overrides)


def memetic_prod(**overrides) -> Config:
    """Shortcut: MemeticProdConfig → Pydantic Config."""
    return load_experiment_config("memetic", "prod", **overrides)


if __name__ == "__main__":
    # Test loader
    print("Testing dataclass loader...\n")

    print("✓ Baseline Test")
    config = baseline_test()
    print(f"  Generations: {config.ga.ngen}")
    print(f"  Population: {config.ga.pop_size}")
    print(f"  Repair: {config.repair.enabled}")
    print(f"  Heuristics: {config.heuristics.master_enabled}")

    print("\n✓ Baseline Prod (custom)")
    config = baseline_prod(ngen=2500, name="thesis-baseline-r01")
    print(f"  Name: {config.name}")
    print(f"  Generations: {config.ga.ngen}")

    print("\n✓ Memetic Test")
    config = memetic_test()
    print(f"  Repair: {config.repair.enabled}")
    print(f"  Memetic: {config.repair.memetic_mode}")

    print("\n✓ All loaders working!")
