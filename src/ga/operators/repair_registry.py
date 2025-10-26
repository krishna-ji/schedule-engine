"""
Repair Heuristics Registry

Central registry for repair heuristics with configurable enable/disable
and priority settings. Follows the same pattern as constraints registry.

Usage:
    from src.ga.operators.repair_registry import get_enabled_repair_heuristics

    enabled_repairs = get_enabled_repair_heuristics()
    for name, info in enabled_repairs.items():
        repair_func = info["function"]
        priority = info["priority"]
        fixes = repair_func(individual, context)
"""

from typing import Dict, Callable
from config import get_config


def get_all_repair_heuristics() -> Dict[str, Dict]:
    """
    Returns registry of all available repair heuristics.

    Each heuristic has:
    - function: The repair function to call
    - priority: Execution order (lower = higher priority)
    - description: Human-readable explanation
    - modifies_length: Whether repair can add/remove genes

    Returns:
        Dict mapping heuristic name to heuristic metadata

    Note:
        Import repair functions here to avoid circular dependencies.
    """
    # Import here to avoid circular dependencies
    from src.ga.operators.repair import (
        repair_instructor_availability,
        repair_group_overlaps,
        repair_room_conflicts,
        repair_instructor_conflicts,
        repair_instructor_qualifications,
        repair_room_type_mismatches,
        repair_session_clustering,
        repair_incomplete_or_extra_sessions,
    )

    return {
        "repair_instructor_availability": {
            "function": repair_instructor_availability,
            "priority": 1,
            "description": "Fix instructor availability violations (shift sessions to instructor-available times)",
            "modifies_length": False,
        },
        "repair_group_overlaps": {
            "function": repair_group_overlaps,
            "priority": 2,
            "description": "Fix group schedule overlaps (same group in multiple sessions)",
            "modifies_length": False,
        },
        "repair_room_conflicts": {
            "function": repair_room_conflicts,
            "priority": 3,
            "description": "Fix room double-booking conflicts",
            "modifies_length": False,
        },
        "repair_instructor_conflicts": {
            "function": repair_instructor_conflicts,
            "priority": 4,
            "description": "Fix instructor double-booking conflicts",
            "modifies_length": False,
        },
        "repair_instructor_qualifications": {
            "function": repair_instructor_qualifications,
            "priority": 5,
            "description": "Reassign unqualified instructors to qualified ones",
            "modifies_length": False,
        },
        "repair_room_type_mismatches": {
            "function": repair_room_type_mismatches,
            "priority": 6,
            "description": "Fix room type mismatches (lab/lecture/seminar)",
            "modifies_length": False,
        },
        "repair_session_clustering": {
            "function": repair_session_clustering,
            "priority": 7,
            "description": "Improve session clustering (merge isolated 1-quantum sessions into 2-3 quantum blocks)",
            "modifies_length": False,
        },
        "repair_incomplete_or_extra_sessions": {
            "function": repair_incomplete_or_extra_sessions,
            "priority": 8,
            "description": "Add missing sessions or remove extra sessions",
            "modifies_length": True,  # WARNING: Can change individual length!
        },
    }


def get_enabled_repair_heuristics() -> Dict[str, Dict]:
    """
    Returns only the enabled repair heuristics based on configuration.

    Filters heuristics according to get_config().repair and applies
    configured priorities. Sorted by priority (ascending).

    Returns:
        Dict mapping heuristic name to heuristic metadata (enabled only)
        Sorted by priority (lower priority number = executed first)

    Example:
        >>> repairs = get_enabled_repair_heuristics()
        >>> for name, info in repairs.items():
        ...     print(f"{name}: priority={info['priority']}")
        repair_availability_violations: priority=1
        repair_group_overlaps: priority=2
    """
    all_repairs = get_all_repair_heuristics()
    enabled_repairs = {}

    heuristics_config = get_config().repair.heuristics or {}

    for name, repair_info in all_repairs.items():
        # Check if heuristic is enabled in config
        heuristic_config = heuristics_config.get(name, {})

        if heuristic_config.get("enabled", True):  # Default: enabled
            # Create copy to avoid modifying original
            enabled_repair = repair_info.copy()

            # Override priority if specified in config
            config_priority = heuristic_config.get("priority")
            if config_priority is not None:
                enabled_repair["priority"] = config_priority

            # Add any additional config params
            enabled_repair["config"] = heuristic_config

            enabled_repairs[name] = enabled_repair

    # Sort by priority (lower = higher priority)
    enabled_repairs = dict(
        sorted(enabled_repairs.items(), key=lambda x: x[1]["priority"])
    )

    return enabled_repairs


def get_repair_statistics_template() -> Dict[str, int]:
    """
    Returns template for repair statistics tracking.

    Returns:
        Dict with all repair heuristic names initialized to 0
    """
    all_repairs = get_all_repair_heuristics()

    stats = {
        "iterations": 0,
        "total_fixes": 0,
    }

    # Add counter for each repair heuristic
    for name in all_repairs.keys():
        # Convert function name to stat key (e.g., repair_availability_violations -> availability_fixes)
        stat_key = name.replace("repair_", "") + "_fixes"
        stats[stat_key] = 0

    return stats


if __name__ == "__main__":
    """Quick test of the registry."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]All Available Repair Heuristics:[/bold cyan]")
    all_repairs = get_all_repair_heuristics()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Heuristic", style="cyan")
    table.add_column("Priority", justify="center")
    table.add_column("Modifies Length", justify="center")
    table.add_column("Description")

    for name, info in all_repairs.items():
        modifies = "⚠️  YES" if info["modifies_length"] else "NO"
        table.add_row(
            name,
            str(info["priority"]),
            modifies,
            info["description"],
        )

    console.print(table)

    console.print("\n[bold green]Enabled Repair Heuristics:[/bold green]")
    enabled = get_enabled_repair_heuristics()

    table2 = Table(show_header=True, header_style="bold green")
    table2.add_column("Heuristic", style="green")
    table2.add_column("Priority", justify="center")
    table2.add_column("Status", justify="center")

    for name, info in enabled.items():
        table2.add_row(name, str(info["priority"]), "✓ ENABLED")

    console.print(table2)
    console.print(f"\n[dim]Total enabled: {len(enabled)}/{len(all_repairs)}[/dim]\n")
