"""
Example: Testing Repair Heuristics Configuration

This script demonstrates how to experiment with different repair configurations
for ablation studies, performance tuning, and debugging.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from config import get_config
from src.ga.operators.repair_registry import get_enabled_repair_heuristics

console = Console()


def show_current_config():
    """Display current repair configuration."""
    console.print("\n[bold cyan]Current Repair Configuration:[/bold cyan]")

    enabled = get_enabled_repair_heuristics()
    console.print(
        f"  Master switch: {'✓ ON' if get_config().repair['enabled'] else '✗ OFF'}"
    )
    console.print(f"  Enabled repairs: {len(enabled)}/7")
    console.print(f"  Max iterations: {get_config().repair['max_iterations']}")
    console.print()


def example_1_disable_expensive_repair():
    """Example 1: Disable expensive repair for faster experimentation."""
    console.rule("[bold]Example 1: Disable Expensive Repair[/bold]")
    console.print()

    console.print("[-] [dim]Use case: Speed up GA runs during experimentation[/dim]")
    console.print()
    console.print("[yellow]Disabling: repair_incomplete_or_extra_sessions[/yellow]")
    console.print(
        "[dim]This repair modifies individual length and can be expensive[/dim]"
    )
    console.print()

    # Show code
    console.print("```python")
    console.print(
        'get_config().repair["heuristics"]["repair_incomplete_or_extra_sessions"]["enabled"] = False'
    )
    console.print("```")
    console.print()


def example_2_ablation_study():
    """Example 2: Run ablation study to measure repair impact."""
    console.rule("[bold]Example 2: Ablation Study[/bold]")
    console.print()

    console.print(
        "[-] [dim]Use case: Measure the individual impact of each repair[/dim]"
    )
    console.print()
    console.print("Strategy: Enable repairs one at a time, measure fitness improvement")
    console.print()

    console.print("```python")
    console.print("# Disable all repairs first")
    console.print("for name in get_config().repair['heuristics']:")
    console.print("    get_config().repair['heuristics'][name]['enabled'] = False")
    console.print()
    console.print("# Enable one repair at a time")
    console.print(
        "for repair_name in ['repair_availability_violations', 'repair_group_overlaps', ...]:"
    )
    console.print(
        "    get_config().repair['heuristics'][repair_name]['enabled'] = True"
    )
    console.print("    run_ga()  # Run and record results")
    console.print(
        "    get_config().repair['heuristics'][repair_name]['enabled'] = False"
    )
    console.print("```")
    console.print()


def example_3_priority_reordering():
    """Example 3: Test different repair order."""
    console.rule("[bold]Example 3: Priority Reordering[/bold]")
    console.print()

    console.print("[-] [dim]Use case: Find optimal repair execution order[/dim]")
    console.print()
    console.print(
        "Hypothesis: Fixing group overlaps first may reduce cascading violations"
    )
    console.print()

    console.print("```python")
    console.print("# Prioritize group overlaps")
    console.print(
        'get_config().repair["heuristics"]["repair_group_overlaps"]["priority"] = 1'
    )
    console.print(
        'get_config().repair["heuristics"]["repair_availability_violations"]["priority"] = 2'
    )
    console.print("```")
    console.print()


def example_4_debugging_constraint():
    """Example 4: Debug specific constraint by disabling its repair."""
    console.rule("[bold]Example 4: Debug Specific Constraint[/bold]")
    console.print()

    console.print(
        "[-] [dim]Use case: Isolate violations to understand their root cause[/dim]"
    )
    console.print()
    console.print(
        "Want to see raw availability violations without repairs masking them?"
    )
    console.print()

    console.print("```python")
    console.print("# Disable availability repair to see violations clearly")
    console.print(
        'get_config().repair["heuristics"]["repair_availability_violations"]["enabled"] = False'
    )
    console.print()
    console.print("# Run GA and examine violation report")
    console.print("# This helps identify data quality issues or seeding problems")
    console.print("```")
    console.print()


def example_5_thesis_experiment():
    """Example 5: Thesis experiment configurations."""
    console.rule("[bold]Example 5: Thesis/Research Experiment[/bold]")
    console.print()

    console.print(
        "[-] [dim]Use case: Compare GA with vs without repair heuristics[/dim]"
    )
    console.print()
    console.print("Experimental conditions:")
    console.print("  - Baseline: No repairs")
    console.print("  - Treatment 1: Basic repairs only (availability, overlaps)")
    console.print("  - Treatment 2: All repairs")
    console.print()

    console.print("```python")
    console.print("# Baseline: No repairs")
    console.print('get_config().repair["enabled"] = False')
    console.print()
    console.print("# Treatment 1: Basic repairs only")
    console.print('get_config().repair["enabled"] = True')
    console.print("for name in get_config().repair['heuristics']:")
    console.print(
        "    if name in ['repair_availability_violations', 'repair_group_overlaps']:"
    )
    console.print(
        "        get_config().repair['heuristics'][name]['enabled'] = True"
    )
    console.print("    else:")
    console.print(
        "        get_config().repair['heuristics'][name]['enabled'] = False"
    )
    console.print()
    console.print("# Treatment 2: All repairs (default)")
    console.print("# All repairs enabled in config")
    console.print("```")
    console.print()


def main():
    console.print()
    console.print(
        "[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]"
    )
    console.print("[bold cyan]  REPAIR HEURISTICS CONFIGURATION EXAMPLES[/bold cyan]")
    console.print(
        "[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]"
    )

    show_current_config()

    example_1_disable_expensive_repair()
    example_2_ablation_study()
    example_3_priority_reordering()
    example_4_debugging_constraint()
    example_5_thesis_experiment()

    console.rule()
    console.print()
    console.print(
        "[dim][-] Tip: Copy these examples to config/ga_params.py to experiment[/dim]"
    )
    console.print(
        "[dim] See docs/REPAIR_REGISTRY_SYSTEM.md for complete documentation[/dim]"
    )
    console.print()


if __name__ == "__main__":
    main()
