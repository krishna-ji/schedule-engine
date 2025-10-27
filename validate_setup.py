"""
Pre-Run Validation Script
Checks that everything is ready for a successful dev run
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def check_data_files():
    """Check that all required data files exist"""
    required_files = [
        "data/Course.json",
        "data/Groups.json",
        "data/Instructors.json",
        "data/Rooms.json",
    ]

    missing = []
    found = []

    for file in required_files:
        if Path(file).exists():
            found.append(file)
        else:
            missing.append(file)

    return found, missing


def check_config_files():
    """Check that config files exist"""
    required = ["configs/common.yaml", "configs/dev.yaml"]

    missing = []
    found = []

    for file in required:
        if Path(file).exists():
            found.append(file)
        else:
            missing.append(file)

    return found, missing


def check_python_packages():
    """Check that required packages are installed"""
    required = ["deap", "pydantic", "rich", "matplotlib", "pyyaml"]

    installed = []
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            installed.append(pkg)
        except ImportError:
            missing.append(pkg)

    return installed, missing


def main():
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]Pre-Run Validation Check[/bold cyan]", border_style="cyan"
        )
    )
    console.print()

    all_good = True

    # Check data files
    console.print("[bold]1. Data Files[/bold]")
    found_data, missing_data = check_data_files()

    if missing_data:
        console.print(f"  [red]✗ Missing:[/red] {', '.join(missing_data)}")
        all_good = False
    else:
        console.print(f"  [green]✓ All data files present[/green]")

    # Check config files
    console.print("\n[bold]2. Configuration Files[/bold]")
    found_config, missing_config = check_config_files()

    if missing_config:
        console.print(f"  [red]✗ Missing:[/red] {', '.join(missing_config)}")
        all_good = False
    else:
        console.print(f"  [green]✓ All config files present[/green]")

    # Check Python packages
    console.print("\n[bold]3. Python Packages[/bold]")
    installed_pkgs, missing_pkgs = check_python_packages()

    if missing_pkgs:
        console.print(f"  [red]✗ Missing:[/red] {', '.join(missing_pkgs)}")
        console.print(f"  [yellow]Run:[/yellow] pip install {' '.join(missing_pkgs)}")
        all_good = False
    else:
        console.print(f"  [green]✓ All packages installed[/green]")

    # Try loading dev config
    console.print("\n[bold]4. Configuration Loading[/bold]")
    try:
        import os

        os.environ["ENVIRONMENT"] = "dev"
        from src.config import init_config

        config = init_config()

        console.print(f"  [green]✓ Config loaded successfully[/green]")
        console.print(f"    - Environment: {config.environment}")
        console.print(f"    - Generations: {config.ga.ngen}")
        console.print(f"    - Population: {config.ga.pop_size}")
        console.print(f"    - Multiprocessing: {config.parallel.use_multiprocessing}")

    except Exception as e:
        console.print(f"  [red]✗ Config failed to load:[/red] {e}")
        all_good = False

    # Final verdict
    console.print()
    console.rule()

    if all_good:
        console.print("\n[bold green]✓ All checks passed! Ready to run:[/bold green]")
        console.print("[cyan]python main.py --env dev[/cyan]\n")
        return 0
    else:
        console.print(
            "\n[bold red]✗ Some checks failed. Fix issues above before running.[/bold red]\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
