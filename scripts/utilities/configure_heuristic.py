#!/usr/bin/env python3
"""
Heuristic Configuration Helper

Quick utility to enable individual heuristics for testing.
Modifies configs/experiments/heuristic_testing.py in-place.

Usage:
    python scripts/utilities/configure_heuristic.py largest-degree-first
    python scripts/utilities/configure_heuristic.py kempe-chain --prod
    python scripts/utilities/configure_heuristic.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table

console = Console()

# Map of user-friendly names to config field names
HEURISTIC_MAP = {
    # Construction
    "largest-degree-first": "heuristic_largest_degree_first",
    "most-constrained-first": "heuristic_most_constrained_first",
    "earliest-deadline-first": "heuristic_earliest_deadline_first",
    # Perturbation
    "random-swap": "heuristic_random_swap",
    "temporal-shift": "heuristic_temporal_shift",
    "room-shuffle": "heuristic_room_shuffle",
    "instructor-reassign": "heuristic_instructor_reassign",
    "multi-perturbation": "heuristic_multi_perturbation",
    # Improvement
    "kempe-chain": "heuristic_kempe_chain",
    "ejection-chain": "heuristic_ejection_chain",
    "variable-depth-search": "heuristic_variable_depth_search",
    # Diversity
    "distance-preserving-crossover": "heuristic_distance_preserving_crossover",
    "crowding-mutation": "heuristic_crowding_mutation",
    "niching-selection": "heuristic_niching_selection",
    "adaptive-diversity-maintenance": "heuristic_adaptive_diversity_maintenance",
    # Meta
    "variable-neighborhood-descent": "heuristic_variable_neighborhood_descent",
    "iterated-local-search": "heuristic_iterated_local_search",
    "adaptive-large-neighborhood": "heuristic_adaptive_large_neighborhood",
    "guided-local-search": "heuristic_guided_local_search",
    # Repair
    "exhaustive-repair": "heuristic_exhaustive_repair",
    "greedy-repair": "heuristic_greedy_repair",
    "igls-repair": "heuristic_igls_repair",
    "lns-repair": "heuristic_lns_repair",
    "memetic-repair": "heuristic_memetic_repair",
    "selective-repair": "heuristic_selective_repair",
}

CATEGORIES = {
    "Construction": [
        "largest-degree-first",
        "most-constrained-first",
        "earliest-deadline-first",
    ],
    "Perturbation": [
        "random-swap",
        "temporal-shift",
        "room-shuffle",
        "instructor-reassign",
        "multi-perturbation",
    ],
    "Improvement": ["kempe-chain", "ejection-chain", "variable-depth-search"],
    "Diversity": [
        "distance-preserving-crossover",
        "crowding-mutation",
        "niching-selection",
        "adaptive-diversity-maintenance",
    ],
    "Meta": [
        "variable-neighborhood-descent",
        "iterated-local-search",
        "adaptive-large-neighborhood",
        "guided-local-search",
    ],
    "Repair": [
        "exhaustive-repair",
        "greedy-repair",
        "igls-repair",
        "lns-repair",
        "memetic-repair",
        "selective-repair",
    ],
}


def list_heuristics() -> None:
    """Display all available heuristics in a table."""
    table = Table(title="Available Heuristics", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Heuristic Name", style="green")
    table.add_column("Total", justify="right", style="yellow")

    for category, heuristics in CATEGORIES.items():
        for idx, h in enumerate(heuristics):
            if idx == 0:
                table.add_row(category, h, str(len(heuristics)))
            else:
                table.add_row("", h, "")

    console.print(table)
    console.print(
        f"\n[dim]Total: {len(HEURISTIC_MAP)} heuristics across {len(CATEGORIES)} categories[/dim]"
    )


def enable_heuristic(name: str, profile: str = "test") -> None:
    """
    Enable a single heuristic in the config file.

    Args:
        name: Heuristic name (e.g., 'largest-degree-first')
        profile: Config profile ('test' or 'prod')
    """
    if name not in HEURISTIC_MAP:
        console.print(f"[red]Unknown heuristic: {name}[/red]")
        console.print("[yellow]Use --list to see available heuristics[/yellow]")
        sys.exit(1)

    field_name = HEURISTIC_MAP[name]
    config_path = project_root / "configs" / "experiments" / "heuristic_testing.py"

    # Read config
    with open(config_path) as f:
        lines = f.readlines()

    # Find the config class to modify
    target_class = (
        "HeuristicTestingTestConfig"
        if profile == "test"
        else "HeuristicTestingProdConfig"
    )
    in_target_class = False
    modified = False

    for i, line in enumerate(lines):
        # Track which class we're in
        if f"class {target_class}" in line:
            in_target_class = True
        elif "class " in line and in_target_class:
            in_target_class = False

        # Modify heuristic fields
        if in_target_class and "heuristic_" in line and ": bool = " in line:
            # Extract field name
            parts = line.split(":")
            current_field = parts[0].strip()

            # Disable all, enable only target
            if current_field == field_name:
                lines[i] = line.replace(": bool = False", ": bool = True")
                if "False" in line:
                    modified = True
            else:
                lines[i] = line.replace(": bool = True", ": bool = False")

    # Write back
    with open(config_path, "w") as f:
        f.writelines(lines)

    status_message = (
        "[green]✓ Enabled: {name}[/green]"
        if modified
        else f"[yellow]Already enabled: {name}[/yellow]"
    )
    console.print(status_message)
    console.print(f"[dim]  Profile: {profile}[/dim]")
    console.print(f"[dim]  Config: {config_path.relative_to(project_root)}[/dim]")
    console.print("\n[yellow]Next step:[/yellow]")
    console.print(f"  uv run heuristic-testing --{profile}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Configure individual heuristics for testing"
    )
    parser.add_argument("heuristic", nargs="?", help="Heuristic name to enable")
    parser.add_argument(
        "--list", action="store_true", help="List all available heuristics"
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Configure production profile (default: test)",
    )

    args = parser.parse_args()

    if args.list:
        list_heuristics()
        return

    if not args.heuristic:
        parser.print_help()
        console.print("\n[yellow]Tip: Use --list to see available heuristics[/yellow]")
        sys.exit(1)

    profile = "prod" if args.prod else "test"
    enable_heuristic(args.heuristic, profile)


if __name__ == "__main__":
    main()
