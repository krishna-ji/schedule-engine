"""
Heuristic Operator Registry

Decorator-based registration system for heuristic operators across six categories:
1. Construction: Build schedules greedily
2. Perturbation: Shake solutions
3. Improvement: Local search moves
4. Diversity: Population diversity maintenance
5. Meta: High-level search strategies
6. Repair: Fix constraint violations

Architecture inspired by src/constraints/registry.py and src/ga/operators/repair_wrappers.py
for consistency across the codebase.

Usage:
    from src.heuristics.registry import construction_heuristic

    @construction_heuristic(
        name="largest_degree_first",
        description="Schedule courses with most conflicts first",
        priority=1,
        enabled_by_default=True
    )
    def largest_degree_first(context):
        # implementation
        return individual
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class HeuristicCategory(str, Enum):
    """Categories of heuristic operators."""

    CONSTRUCTION = "construction"
    PERTURBATION = "perturbation"
    IMPROVEMENT = "improvement"
    DIVERSITY = "diversity"
    META = "meta"
    REPAIR = "repair"


@dataclass
class HeuristicMetadata:
    """
    Metadata for a registered heuristic operator.

    Attributes:
        name: Unique heuristic identifier (used in config)
        function: The heuristic function to call
        description: Human-readable explanation
        category: Heuristic category (construction/perturbation/etc)
        priority: Execution order (lower = higher priority, executed first)
        enabled_by_default: Whether enabled by default in config
        requires_population: Whether heuristic needs population access
        modifies_individual: Whether heuristic modifies individuals in-place
    """

    name: str
    function: Callable
    description: str
    category: HeuristicCategory
    priority: int
    enabled_by_default: bool = True
    requires_population: bool = False
    modifies_individual: bool = False


# ================
# GLOBAL HEURISTIC REGISTRIES (by category)
# ================

_CONSTRUCTION_HEURISTICS: dict[str, HeuristicMetadata] = {}
_PERTURBATION_HEURISTICS: dict[str, HeuristicMetadata] = {}
_IMPROVEMENT_HEURISTICS: dict[str, HeuristicMetadata] = {}
_DIVERSITY_HEURISTICS: dict[str, HeuristicMetadata] = {}
_META_HEURISTICS: dict[str, HeuristicMetadata] = {}
_REPAIR_HEURISTICS: dict[str, HeuristicMetadata] = {}

# Track global registration metadata to guard against duplicates
_GLOBAL_HEURISTIC_NAMES: dict[str, HeuristicCategory] = {}
_CATEGORY_PRIORITIES: dict[HeuristicCategory, dict[int, str]] = {
    category: {} for category in HeuristicCategory
}


def _get_registry(category: HeuristicCategory) -> dict[str, HeuristicMetadata]:
    """Get the registry for a specific category."""
    registries = {
        HeuristicCategory.CONSTRUCTION: _CONSTRUCTION_HEURISTICS,
        HeuristicCategory.PERTURBATION: _PERTURBATION_HEURISTICS,
        HeuristicCategory.IMPROVEMENT: _IMPROVEMENT_HEURISTICS,
        HeuristicCategory.DIVERSITY: _DIVERSITY_HEURISTICS,
        HeuristicCategory.META: _META_HEURISTICS,
        HeuristicCategory.REPAIR: _REPAIR_HEURISTICS,
    }
    return registries[category]


def _validate_registration(
    name: str, category: HeuristicCategory, priority: int
) -> None:
    """Ensure heuristic names and priorities remain unique."""

    if name in _GLOBAL_HEURISTIC_NAMES:
        existing_category = _GLOBAL_HEURISTIC_NAMES[name]
        raise ValueError(
            f"Heuristic '{name}' already registered under '{existing_category.value}'"
        )

    category_priorities = _CATEGORY_PRIORITIES[category]
    if priority in category_priorities:
        conflict = category_priorities[priority]
        raise ValueError(
            f"Priority {priority} already used by heuristic '{conflict}' in category '{category.value}'"
        )

    _GLOBAL_HEURISTIC_NAMES[name] = category
    category_priorities[priority] = name


# ================
# DECORATOR FUNCTIONS
# ================


def _heuristic_decorator(category: HeuristicCategory):
    """Factory function to create category-specific decorators."""

    def decorator(
        name: str,
        description: str,
        priority: int,
        enabled_by_default: bool = True,
        requires_population: bool = False,
        modifies_individual: bool = False,
    ):
        """
        Decorator to register a heuristic operator.

        Args:
            name: Heuristic identifier (must match config field name)
            description: Human-readable explanation
            priority: Execution order (1 = highest priority)
            enabled_by_default: Whether enabled by default
            requires_population: Whether heuristic needs population access
            modifies_individual: Whether heuristic modifies individuals in-place
        """

        def inner_decorator(func: Callable) -> Callable:
            _validate_registration(name, category, priority)
            metadata = HeuristicMetadata(
                name=name,
                function=func,
                description=description,
                category=category,
                priority=priority,
                enabled_by_default=enabled_by_default,
                requires_population=requires_population,
                modifies_individual=modifies_individual,
            )
            registry = _get_registry(category)
            registry[name] = metadata
            # Store metadata on function for introspection
            func._heuristic_metadata = metadata
            return func

        return inner_decorator

    return decorator


# Create category-specific decorators
construction_heuristic = _heuristic_decorator(HeuristicCategory.CONSTRUCTION)
perturbation_heuristic = _heuristic_decorator(HeuristicCategory.PERTURBATION)
improvement_heuristic = _heuristic_decorator(HeuristicCategory.IMPROVEMENT)
diversity_heuristic = _heuristic_decorator(HeuristicCategory.DIVERSITY)
meta_heuristic = _heuristic_decorator(HeuristicCategory.META)
repair_heuristic = _heuristic_decorator(HeuristicCategory.REPAIR)


# ================
# REGISTRY ACCESS FUNCTIONS
# ================


def get_all_heuristics() -> dict[str, HeuristicMetadata]:
    """Get all registered heuristics across all categories."""
    all_heuristics = {}
    for category in HeuristicCategory:
        registry = _get_registry(category)
        all_heuristics.update(registry)
    return all_heuristics


def get_heuristics_by_category(
    category: HeuristicCategory,
) -> dict[str, HeuristicMetadata]:
    """Get all heuristics in a specific category."""
    return _get_registry(category).copy()


def get_construction_heuristics() -> dict[str, HeuristicMetadata]:
    """Get all construction heuristics."""
    return get_heuristics_by_category(HeuristicCategory.CONSTRUCTION)


def get_perturbation_heuristics() -> dict[str, HeuristicMetadata]:
    """Get all perturbation heuristics."""
    return get_heuristics_by_category(HeuristicCategory.PERTURBATION)


def get_improvement_heuristics() -> dict[str, HeuristicMetadata]:
    """Get all improvement heuristics."""
    return get_heuristics_by_category(HeuristicCategory.IMPROVEMENT)


def get_diversity_heuristics() -> dict[str, HeuristicMetadata]:
    """Get all diversity heuristics."""
    return get_heuristics_by_category(HeuristicCategory.DIVERSITY)


def get_meta_heuristics() -> dict[str, HeuristicMetadata]:
    """Get all meta-heuristics."""
    return get_heuristics_by_category(HeuristicCategory.META)


def get_repair_heuristics() -> dict[str, HeuristicMetadata]:
    """Get all repair heuristics."""
    return get_heuristics_by_category(HeuristicCategory.REPAIR)


def get_heuristic_by_name(name: str) -> HeuristicMetadata | None:
    """Get heuristic metadata by name (searches all categories)."""
    all_heuristics = get_all_heuristics()
    return all_heuristics.get(name)


def get_enabled_heuristics(
    category: HeuristicCategory | None = None,
) -> dict[str, HeuristicMetadata]:
    """
    Get enabled heuristics from config, sorted by priority.

    Args:
        category: Optional category filter (if None, returns all categories)

    Returns:
        Dict mapping heuristic names to metadata (enabled only)
        Sorted by priority (lower priority number = executed first)
    """
    from src.config import get_config

    # Get heuristics to check
    if category:
        heuristics_to_check = get_heuristics_by_category(category)
    else:
        heuristics_to_check = get_all_heuristics()

    enabled_heuristics = {}

    # Get config (handle case where heuristics config doesn't exist yet)
    config = get_config()
    heuristics_config = getattr(config, "heuristics", None)

    if not heuristics_config:
        # No config section - use defaults
        for name, meta in heuristics_to_check.items():
            if meta.enabled_by_default:
                enabled_heuristics[name] = meta
    else:
        # Check each category's config
        for name, meta in heuristics_to_check.items():
            category_config = (
                getattr(heuristics_config, meta.category.value, None) or {}
            )
            heuristic_config = category_config.get(name, {})

            # Check if enabled
            is_enabled = heuristic_config.get("enabled", meta.enabled_by_default)

            if not is_enabled:
                continue

            # Get priority from config (or use default)
            priority = heuristic_config.get("priority", meta.priority)

            # Create metadata with config-overridden priority
            enabled_heuristics[name] = HeuristicMetadata(
                name=name,
                function=meta.function,
                description=meta.description,
                category=meta.category,
                priority=priority,
                enabled_by_default=meta.enabled_by_default,
                requires_population=meta.requires_population,
                modifies_individual=meta.modifies_individual,
            )

    # Sort by priority (lower = higher priority)
    enabled_heuristics = dict(
        sorted(enabled_heuristics.items(), key=lambda x: x[1].priority)
    )

    return enabled_heuristics


def list_all_heuristics() -> None:
    """Print all registered heuristics with their metadata."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    for category in HeuristicCategory:
        heuristics = get_heuristics_by_category(category)

        if not heuristics:
            continue

        console.print(f"\n[bold cyan]{category.value.upper()} HEURISTICS[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Priority", justify="center")
        table.add_column("Enabled", justify="center")
        table.add_column("Description")

        for name, meta in sorted(heuristics.items(), key=lambda x: x[1].priority):
            table.add_row(
                name,
                str(meta.priority),
                "✓" if meta.enabled_by_default else "✗",
                meta.description,
            )

        console.print(table)


# ================
# STATISTICS TRACKING
# ================


def get_heuristic_statistics_template() -> dict[str, int]:
    """Returns template for heuristic statistics tracking."""
    all_heuristics = get_all_heuristics()

    stats = {
        "total_applications": 0,
        "total_improvements": 0,
    }

    # Add counter for each heuristic
    for name in all_heuristics.keys():
        stats[f"{name}_applications"] = 0
        stats[f"{name}_improvements"] = 0

    return stats


if __name__ == "__main__":
    """Quick test of the heuristic registry."""
    from rich.console import Console

    console = Console()

    console.print("\n[bold cyan]Heuristic Operator Registry[/bold cyan]")
    console.print(
        "[dim]Use category-specific decorators to register heuristics[/dim]\n"
    )

    all_heuristics = get_all_heuristics()

    if all_heuristics:
        list_all_heuristics()
    else:
        console.print("[yellow]No heuristics registered yet.[/yellow]")
        console.print(
            "\n[dim]Import heuristic modules to register operators with decorators.[/dim]"
        )
