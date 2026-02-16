"""
Repair Operator Wrapper Registry

Provides decorator-based registration system for repair operators, following
the same clean architecture as constraints registry.

This wrapper layer:
- Decouples repair functions from registry logic
- Enables flexible enable/disable via config
- Supports priority ordering and metadata
- Allows pluggable repair strategies

Registered Operators (7 base + 1 soft constraint):
  Priority 1-7: Hard constraint repairs (HC1-HC5, HC8, HC4)
  Priority 8: Soft constraint repair (SC4 - session_continuity)

Default configuration (src.config):
  All hard constraint repairs enabled by default
  Soft constraint repair enabled in selective mode only

Architecture inspired by src.constraints.registry for consistency.

Usage:
    from src.ga.repair.wrappers import repair_operator

    @repair_operator(
        name="repair_group_overlaps",
        description="Fix group schedule overlaps",
        priority=2,
        modifies_length=False
    )
    def repair_group_overlaps(individual, context):
        # implementation
        return fixes_count

    # Get enabled repairs from config
    from src.ga.repair.wrappers import get_enabled_repair_operators
    enabled = get_enabled_repair_operators()  # Returns priority-sorted dict
"""

from collections.abc import Callable
from dataclasses import dataclass

RepairFunc = Callable[..., int]


@dataclass
class RepairOperatorMetadata:
    """
    Metadata for a registered repair operator.

    Attributes:
        name: Unique repair operator identifier (used in config)
        function: The repair function to call
        description: Human-readable explanation of what is repaired
        priority: Execution order (lower = higher priority, executed first)
        modifies_length: Whether repair can add/remove genes
        enabled_by_default: Whether operator is enabled by default in config
    """

    name: str
    function: Callable
    description: str
    priority: int
    modifies_length: bool = False
    enabled_by_default: bool = True


# GLOBAL REPAIR REGISTRY
_REPAIR_OPERATORS: dict[str, RepairOperatorMetadata] = {}


# DECORATOR FUNCTION
def repair_operator(
    name: str,
    description: str,
    priority: int,
    modifies_length: bool = False,
    enabled_by_default: bool = True,
) -> Callable[[RepairFunc], RepairFunc]:
    """
    Decorator to register a repair operator function.

    Repair operators fix constraint violations in GA individuals after
    mutation/crossover operations. They project infeasible solutions
    onto the feasible region.

    Args:
        name: Repair operator identifier (must match config field name)
        description: Human-readable explanation of what is repaired
        priority: Execution order (1 = highest priority, executed first)
        modifies_length: Whether repair can add/remove genes from individual
        enabled_by_default: Whether operator is enabled by default

    Example:
        @repair_operator(
            name="repair_group_overlaps",
            description="Fix group schedule overlaps (same group in multiple sessions)",
            priority=2,
            modifies_length=False
        )
        def repair_group_overlaps(
            individual: List[SessionGene],
            context: SchedulingContext
        ) -> int:
            fixes = 0
            # ... repair logic ...
            return fixes

    Function Signature Requirements:
        - Must accept 'individual' as first parameter (List[SessionGene])
        - Must accept 'context' as second parameter (SchedulingContext)
        - Must return int (number of fixes applied)
        - Must modify individual in-place (no return of modified individual)
    """

    def decorator(func: RepairFunc) -> RepairFunc:
        metadata = RepairOperatorMetadata(
            name=name,
            function=func,
            description=description,
            priority=priority,
            modifies_length=modifies_length,
            enabled_by_default=enabled_by_default,
        )
        _REPAIR_OPERATORS[name] = metadata
        # Store metadata on function for introspection
        func._repair_metadata = metadata  # type: ignore[attr-defined]
        return func

    return decorator


# REGISTRY ACCESS FUNCTIONS
def get_all_repair_operators() -> dict[str, RepairOperatorMetadata]:
    """
    Get all registered repair operators with their metadata.

    Returns:
        Dict mapping repair operator names to RepairOperatorMetadata objects
    """
    return _REPAIR_OPERATORS.copy()


def get_repair_operator_metadata(name: str) -> RepairOperatorMetadata | None:
    """
    Get repair operator metadata by name.

    Args:
        name: Repair operator identifier

    Returns:
        RepairOperatorMetadata if found, None otherwise
    """
    return _REPAIR_OPERATORS.get(name)


def get_repair_operator_function(name: str) -> Callable | None:
    """
    Get repair operator function by name.

    Args:
        name: Repair operator identifier

    Returns:
        Repair function if found, None otherwise
    """
    metadata = _REPAIR_OPERATORS.get(name)
    return metadata.function if metadata else None


def get_enabled_repair_operators() -> dict[str, RepairOperatorMetadata]:
    """
    Get enabled repair operators from config, sorted by priority.

    Filters operators according to config settings and applies
    configured priorities. Sorted by priority (ascending).

    Returns:
        Dict mapping repair operator names to metadata (enabled only)
        Sorted by priority (lower priority number = executed first)

    Example:
        >>> repairs = get_enabled_repair_operators()
        >>> for name, meta in repairs.items():
        ...     print(f"{name}: priority={meta.priority}")
        repair_instructor_availability: priority=1
        repair_group_overlaps: priority=2
    """
    from src.config import get_config

    all_repairs = get_all_repair_operators()
    enabled_repairs = {}

    heuristics_config = get_config().repair.heuristics or {}

    for name, repair_meta in all_repairs.items():
        # Check if enabled in config
        config_entry = heuristics_config.get(name, {})

        # Default to enabled_by_default if not in config
        is_enabled = config_entry.get("enabled", repair_meta.enabled_by_default)

        if not is_enabled:
            continue

        # Get priority from config (or use default from decorator)
        priority = config_entry.get("priority", repair_meta.priority)

        # Create new metadata with config-overridden priority
        enabled_repairs[name] = RepairOperatorMetadata(
            name=name,
            function=repair_meta.function,
            description=repair_meta.description,
            priority=priority,
            modifies_length=repair_meta.modifies_length,
            enabled_by_default=repair_meta.enabled_by_default,
        )

    # Sort by priority (lower = higher priority)
    enabled_repairs = dict(sorted(enabled_repairs.items(), key=lambda x: x[1].priority))

    return enabled_repairs


def list_all_repair_operators() -> None:
    """
    Print all registered repair operators with their metadata.

    Useful for debugging and documentation generation.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    all_repairs = get_all_repair_operators()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Repair Operator", style="cyan")
    table.add_column("Priority", justify="center")
    table.add_column("Modifies Length", justify="center")
    table.add_column("Description")

    for name, meta in sorted(all_repairs.items(), key=lambda x: x[1].priority):
        table.add_row(
            name,
            str(meta.priority),
            "✓" if meta.modifies_length else "✗",
            meta.description,
        )

    console.print(table)


# STATISTICS TRACKING
def get_repair_statistics_template() -> dict[str, int | float]:
    """
    Returns template for repair statistics tracking.

    Returns:
        Dict with all repair operator names initialized to 0
    """
    all_repairs = get_all_repair_operators()

    stats: dict[str, int | float] = {
        "iterations": 0,
        "total_fixes": 0,
    }

    # Add counter for each repair operator
    for name in all_repairs:
        stats[f"{name}_fixes"] = 0
        stats[f"{name}_calls"] = 0

    return stats


if __name__ == "__main__":
    """Quick test of the repair wrapper registry."""
    from rich.console import Console

    console = Console()

    console.print("\n[bold cyan]Repair Operator Wrapper Registry[/bold cyan]")
    console.print(
        "[dim]Use @repair_operator decorator to register repair functions[/dim]\n"
    )

    if _REPAIR_OPERATORS:
        list_all_repair_operators()
    else:
        console.print("[yellow]No repair operators registered yet.[/yellow]")
        console.print(
            "\n[dim]Import repair.py to register operators with decorators.[/dim]"
        )
