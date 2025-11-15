"""
Heuristic Toolbox Usage Examples

Demonstrates how to use the heuristic toolbox in various scenarios.
The heuristic system provides 19 operators across 5 categories for
schedule optimization, diversity maintenance, and search strategies.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from src.heuristics import (
    get_all_heuristics,
    get_enabled_heuristics,
    get_heuristic_by_name,
    list_all_heuristics,
    get_construction_heuristics,
    get_perturbation_heuristics,
    get_improvement_heuristics,
    get_diversity_heuristics,
    get_meta_heuristics,
)

console = Console()


def example_1_list_all_heuristics():
    """Example 1: List all registered heuristics."""
    console.print("\n[bold cyan]Example 1: List All Heuristics[/bold cyan]\n")

    all_heuristics = get_all_heuristics()
    console.print(f"Total heuristics: {len(all_heuristics)}\n")

    # Pretty table display
    list_all_heuristics()


def example_2_get_by_category():
    """Example 2: Get heuristics by category."""
    console.print("\n[bold cyan]Example 2: Get Heuristics by Category[/bold cyan]\n")

    # Get perturbation heuristics
    perturbation_ops = get_perturbation_heuristics()
    console.print(f"[green]Perturbation heuristics:[/green] {len(perturbation_ops)}")
    for name, meta in perturbation_ops.items():
        console.print(f"  • {name} (priority: {meta.priority})")

    # Get improvement heuristics
    improvement_ops = get_improvement_heuristics()
    console.print(f"\n[green]Improvement heuristics:[/green] {len(improvement_ops)}")
    for name, meta in improvement_ops.items():
        console.print(f"  • {name} (priority: {meta.priority})")


def example_3_get_enabled_only():
    """Example 3: Get only enabled heuristics from config."""
    console.print("\n[bold cyan]Example 3: Get Enabled Heuristics[/bold cyan]\n")

    # Get enabled perturbation heuristics (reads from config)
    try:
        from src.config import get_config

        config = get_config()

        enabled_perturbation = get_enabled_heuristics(category="perturbation")
        console.print(
            f"[green]Enabled perturbation heuristics:[/green] {len(enabled_perturbation)}"
        )
        for name in enabled_perturbation.keys():
            console.print(f"  • {name}")

        # Get all enabled heuristics
        all_enabled = get_enabled_heuristics()
        console.print(f"\n[green]Total enabled heuristics:[/green] {len(all_enabled)}")

    except Exception as e:
        console.print(f"[yellow]Note: Config not loaded: {e}[/yellow]")


def example_4_get_specific_heuristic():
    """Example 4: Get specific heuristic by name."""
    console.print("\n[bold cyan]Example 4: Get Specific Heuristic[/bold cyan]\n")

    # Get temporal_shift heuristic
    heuristic = get_heuristic_by_name("temporal_shift")

    if heuristic:
        console.print(f"[green]Found:[/green] {heuristic.name}")
        console.print(f"  Category: {heuristic.category.value}")
        console.print(f"  Description: {heuristic.description}")
        console.print(f"  Priority: {heuristic.priority}")
        console.print(f"  Requires population: {heuristic.requires_population}")
        console.print(f"  Modifies individual: {heuristic.modifies_individual}")


def example_5_apply_heuristic():
    """Example 5: Apply heuristic to individual (mock)."""
    console.print("\n[bold cyan]Example 5: Apply Heuristic (Mock)[/bold cyan]\n")

    # Get kempe_chain heuristic
    heuristic = get_heuristic_by_name("kempe_chain")

    if heuristic:
        console.print(f"[green]Applying:[/green] {heuristic.name}")
        console.print(f"  Description: {heuristic.description}")

        # In actual usage, you would call:
        # improvements = heuristic.function(individual, context, max_iterations=5)

        console.print("\n[dim]Mock call:[/dim]")
        console.print(
            "[dim]improvements = heuristic.function(individual, context)[/dim]"
        )


def example_6_construction_pipeline():
    """Example 6: Using construction heuristics for initialization."""
    console.print("\n[bold cyan]Example 6: Construction Pipeline[/bold cyan]\n")

    construction_ops = get_construction_heuristics()

    console.print(
        "[green]Construction heuristics for population initialization:[/green]"
    )
    console.print("Use these to create initial population with better quality:\n")

    for name, meta in sorted(construction_ops.items(), key=lambda x: x[1].priority):
        console.print(f"{meta.priority}. {name}")
        console.print(f"   {meta.description}")
        console.print(f"   [dim]Usage: individual = {name}(context)[/dim]\n")


def example_7_meta_heuristic_strategies():
    """Example 7: Meta-heuristic strategies."""
    console.print("\n[bold cyan]Example 7: Meta-Heuristic Strategies[/bold cyan]\n")

    meta_ops = get_meta_heuristics()

    console.print("[green]High-level search strategies:[/green]\n")

    for name, meta in sorted(meta_ops.items(), key=lambda x: x[1].priority):
        console.print(f"• [cyan]{name}[/cyan]")
        console.print(f"  {meta.description}")
        console.print()


def example_8_heuristic_metadata():
    """Example 8: Inspect heuristic metadata."""
    console.print("\n[bold cyan]Example 8: Heuristic Metadata[/bold cyan]\n")

    all_heuristics = get_all_heuristics()

    # Count by category
    category_counts = {}
    for meta in all_heuristics.values():
        category = meta.category.value
        category_counts[category] = category_counts.get(category, 0) + 1

    console.print("[green]Heuristics by category:[/green]")
    for category, count in sorted(category_counts.items()):
        console.print(f"  {category}: {count}")

    # Count by requirements
    console.print("\n[green]Heuristics requiring population:[/green]")
    pop_required = [
        name for name, meta in all_heuristics.items() if meta.requires_population
    ]
    console.print(f"  {len(pop_required)} heuristics")
    for name in pop_required:
        console.print(f"    • {name}")

    console.print("\n[green]Heuristics modifying individuals in-place:[/green]")
    modifying = [
        name for name, meta in all_heuristics.items() if meta.modifies_individual
    ]
    console.print(f"  {len(modifying)} heuristics")


def main():
    """Run all examples."""
    console.print("[bold magenta]Heuristic Toolbox Usage Examples[/bold magenta]")
    console.print("[dim]Demonstrating the decorator-based heuristic system[/dim]")

    example_1_list_all_heuristics()
    example_2_get_by_category()
    example_3_get_enabled_only()
    example_4_get_specific_heuristic()
    example_5_apply_heuristic()
    example_6_construction_pipeline()
    example_7_meta_heuristic_strategies()
    example_8_heuristic_metadata()

    console.print("\n[bold green]✓ All examples completed![/bold green]\n")


if __name__ == "__main__":
    main()
