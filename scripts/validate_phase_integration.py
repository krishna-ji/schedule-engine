#!/usr/bin/env python3
"""
Validation script for Phase 1 & 2 integration.

Checks that all core components are working correctly:
- Configuration system (hierarchical loading)
- Heuristics registry (19 operators)
- GA core components
- Environment profiles (test/prod/med)

Usage:
    python scripts/validate_phase_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def validate_config_system():
    """Validate hierarchical configuration system."""
    from src.config.loader import load_config, _check_hierarchical_structure
    from src.config import init_config, get_config
    
    console.print("\n[bold cyan]1. Configuration System[/bold cyan]")
    
    # Check structure
    if not _check_hierarchical_structure():
        console.print("[red]✗ Hierarchical structure not found[/red]")
        return False
    
    console.print("[green]✓ Hierarchical structure exists[/green]")
    
    # Load config
    init_config()
    config = get_config()
    
    # Validate domains
    domains = {
        'common': ['time', 'io', 'parallel', 'feasibility'],
        'ga': ['ga', 'hard_constraints', 'soft_constraints', 'repair', 'heuristics'],
        'rl': ['rl'],
    }
    
    for domain, attrs in domains.items():
        for attr in attrs:
            if not hasattr(config, attr):
                console.print(f"[red]✗ Missing {domain} domain: {attr}[/red]")
                return False
    
    console.print(f"[green]✓ All config domains present[/green]")
    console.print(f"[dim]  Loaded: {config.name} ({config.environment})[/dim]")
    
    return True


def validate_heuristics():
    """Validate heuristics registry."""
    from src.heuristics.registry import get_registry, HeuristicCategory
    
    console.print("\n[bold cyan]2. Heuristics Registry[/bold cyan]")
    
    # Import all heuristics
    import src.heuristics.construction
    import src.heuristics.perturbation
    import src.heuristics.improvement
    import src.heuristics.diversity
    import src.heuristics.meta
    
    registry = get_registry()
    
    if len(registry) == 0:
        console.print("[red]✗ No heuristics registered[/red]")
        return False
    
    # Count by category
    category_counts = {}
    for h in registry.values():
        category_counts[h.category.value] = category_counts.get(h.category.value, 0) + 1
    
    console.print(f"[green]✓ {len(registry)} heuristics registered[/green]")
    
    # Show breakdown
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    
    for category in HeuristicCategory:
        count = category_counts.get(category.value, 0)
        table.add_row(category.value, str(count))
    
    console.print(table)
    
    return len(registry) == 19  # Expected count


def validate_environment_profiles():
    """Validate environment profiles load correctly."""
    import os
    from src.config.loader import load_config
    
    console.print("\n[bold cyan]3. Environment Profiles[/bold cyan]")
    
    profiles = {
        'test': {'ngen': 30, 'pop_size': 10, 'runtime': '5-10 min'},
        'prod': {'ngen': 1000, 'pop_size': 100, 'runtime': '24-48h'},
        'med': {'ngen': 200, 'pop_size': 50, 'runtime': '2-4h'},
    }
    
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Environment", style="cyan")
    table.add_column("Generations", justify="right")
    table.add_column("Population", justify="right")
    table.add_column("Runtime", style="dim")
    
    all_ok = True
    for env, expected in profiles.items():
        os.environ['ENVIRONMENT'] = env
        config = load_config()
        
        if config.ga.ngen != expected['ngen'] or config.ga.pop_size != expected['pop_size']:
            all_ok = False
            table.add_row(
                env,
                f"[red]{config.ga.ngen}[/red]",
                f"[red]{config.ga.pop_size}[/red]",
                expected['runtime']
            )
        else:
            table.add_row(
                env,
                f"[green]{config.ga.ngen}[/green]",
                f"[green]{config.ga.pop_size}[/green]",
                expected['runtime']
            )
    
    console.print(table)
    
    # Reset
    os.environ['ENVIRONMENT'] = 'test'
    
    if all_ok:
        console.print("[green]✓ All profiles valid[/green]")
    else:
        console.print("[red]✗ Some profiles have incorrect values[/red]")
    
    return all_ok


def validate_ga_components():
    """Validate GA core components."""
    console.print("\n[bold cyan]4. GA Core Components[/bold cyan]")
    
    try:
        from src.config import init_config
        from src.core.ga_scheduler import GAScheduler
        from src.ga.population import generate_course_group_aware_population
        from src.ga.operators.crossover import crossover_course_group_aware
        from src.ga.operators.mutation import mutate_individual
        
        init_config()
        
        console.print("[green]✓ GA scheduler[/green]")
        console.print("[green]✓ Population generator[/green]")
        console.print("[green]✓ Crossover operator[/green]")
        console.print("[green]✓ Mutation operator[/green]")
        
        return True
    except Exception as e:
        console.print(f"[red]✗ Import error: {e}[/red]")
        return False


def main():
    """Run all validations."""
    console.print(Panel.fit(
        "[bold cyan]Phase 1 & 2 Integration Validation[/bold cyan]\n"
        "Checking configuration, heuristics, and core components",
        border_style="cyan"
    ))
    
    validations = [
        ("Configuration System", validate_config_system),
        ("Heuristics Registry", validate_heuristics),
        ("Environment Profiles", validate_environment_profiles),
        ("GA Components", validate_ga_components),
    ]
    
    results = []
    for name, validator in validations:
        try:
            success = validator()
            results.append((name, success))
        except Exception as e:
            console.print(f"\n[red]✗ {name} validation failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    console.print("\n" + "="*60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    if passed == total:
        console.print(f"[bold green]✓ All validations passed ({passed}/{total})[/bold green]")
        console.print("\n[green]Phase 1 & 2 integration is ready for production[/green]")
        return 0
    else:
        failed = total - passed
        console.print(f"[bold red]✗ {failed} validation(s) failed ({passed}/{total} passed)[/bold red]")
        console.print("\n[red]Please fix issues before proceeding[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
