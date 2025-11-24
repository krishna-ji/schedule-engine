"""
Heuristic Toolbox Registry Test

Quick validation that all heuristics are properly registered
and accessible via the decorator-based registry system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console


def test_heuristic_registry():
    """Test that all heuristics are properly registered."""
    console = Console()

    console.print("\n[bold cyan]Testing Heuristic Toolbox Registry[/bold cyan]\n")

    # Import to trigger registration
    from src.heuristics import (
        get_all_heuristics,
        get_construction_heuristics,
        get_perturbation_heuristics,
        get_improvement_heuristics,
        get_diversity_heuristics,
        get_meta_heuristics,
        get_repair_heuristics,
        list_all_heuristics,
    )

    # Get all heuristics
    all_heuristics = get_all_heuristics()

    console.print(
        f"[green]✓[/green] Total heuristics registered: {len(all_heuristics)}"
    )

    # Check each category
    categories = {
        "Construction": get_construction_heuristics(),
        "Perturbation": get_perturbation_heuristics(),
        "Improvement": get_improvement_heuristics(),
        "Diversity": get_diversity_heuristics(),
        "Meta": get_meta_heuristics(),
        "Repair": get_repair_heuristics(),
    }

    for category_name, heuristics in categories.items():
        console.print(f"[green]✓[/green] {category_name}: {len(heuristics)} heuristics")
        for name in heuristics.keys():
            console.print(f"  • {name}")

    # Expected counts
    expected_counts = {
        "Construction": 3,
        "Perturbation": 5,
        "Improvement": 3,
        "Diversity": 4,
        "Meta": 4,
        "Repair": 3,
    }

    console.print("\n[bold]Validation:[/bold]")
    all_valid = True

    for category_name, expected_count in expected_counts.items():
        actual_count = len(categories[category_name])
        if actual_count == expected_count:
            console.print(
                f"[green]✓[/green] {category_name}: {actual_count}/{expected_count}"
            )
        else:
            console.print(
                f"[red]✗[/red] {category_name}: {actual_count}/{expected_count} (MISMATCH)"
            )
            all_valid = False

    # Display full table
    console.print("\n[bold]All Registered Heuristics:[/bold]\n")
    list_all_heuristics()

    if all_valid:
        console.print(
            "\n[bold green]✓ All heuristics successfully registered![/bold green]"
        )
    else:
        console.print("\n[bold red]✗ Some heuristics missing![/bold red]")

    return all_valid


if __name__ == "__main__":
    success = test_heuristic_registry()
    exit(0 if success else 1)
