#!/usr/bin/env python3
"""
Minimal CLI utilities for Schedule Engine.

The primary workflow is through Jupyter notebooks (notebooks/).
This CLI provides utility commands only.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def main_diagnose() -> int:
    """System diagnostics."""
    try:
        from scripts.diagnostics.diagnose_gpu import main as diagnose_gpu
        diagnose_gpu()
        return 0
    except Exception as e:
        console.print(f"[red]Diagnostics failed: {e}[/red]")
        return 1


def main_clean() -> int:
    """Clean output directory (keeps manifest)."""
    output_dir = Path("output")
    if not output_dir.exists():
        console.print("[yellow]No output directory to clean[/yellow]")
        return 0

    console.print(f"[yellow]Cleaning: {output_dir}...[/yellow]")
    for item in output_dir.iterdir():
        if item.name == "experiment_manifest.json":
            continue  # Keep manifest
        if item.name == "README.md":
            continue  # Keep readme
        try:
            if item.is_dir():
                shutil.rmtree(item)
                console.print(f"  Removed: {item.name}/")
            else:
                item.unlink()
                console.print(f"  Removed: {item.name}")
        except Exception as e:
            console.print(f"[red]  Failed: {item.name} ({e})[/red]")

    console.print("[green]✓ Cleaned output directory[/green]")
    return 0


def main_list() -> int:
    """List experiment history from manifest."""
    manifest_path = Path("output/experiment_manifest.json")

    if not manifest_path.exists():
        console.print("[yellow]No experiments found[/yellow]")
        return 0

    with open(manifest_path) as f:
        manifest = json.load(f)

    experiments = manifest.get("experiments", [])
    if not experiments:
        console.print("[yellow]No experiments in manifest[/yellow]")
        return 0

    table = Table(title="Experiment History")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Mode", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Best Fitness", style="white")

    for exp in experiments[-20:]:  # Last 20 entries
        fitness = exp.get("best_fitness", "")
        if isinstance(fitness, (list, tuple)):
            fitness_str = f"({fitness[0]:.0f}, {fitness[1]:.2f})" if len(fitness) == 2 else str(fitness)
        else:
            fitness_str = str(fitness) if fitness else ""

        table.add_row(
            exp.get("timestamp", "")[:19],  # Trim to datetime
            exp.get("name", "")[:30],  # Truncate long names
            exp.get("mode", ""),
            exp.get("status", ""),
            fitness_str,
        )

    console.print(table)
    return 0


def main_stats() -> int:
    """Show manifest statistics."""
    manifest_path = Path("output/experiment_manifest.json")

    if not manifest_path.exists():
        console.print("[yellow]No manifest found[/yellow]")
        return 0

    with open(manifest_path) as f:
        manifest = json.load(f)

    experiments = manifest.get("experiments", [])

    # Count by status
    status_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}

    for exp in experiments:
        status = exp.get("status", "unknown")
        mode = exp.get("mode", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    console.print("[cyan]Manifest Statistics[/cyan]")
    console.print(f"  Total experiments: {len(experiments)}")
    console.print()
    console.print("[cyan]By Status:[/cyan]")
    for status, count in sorted(status_counts.items()):
        console.print(f"  {status}: {count}")
    console.print()
    console.print("[cyan]By Mode:[/cyan]")
    for mode, count in sorted(mode_counts.items()):
        console.print(f"  {mode}: {count}")

    return 0


def main_archive() -> int:
    """Archive incomplete runs."""
    try:
        from src.workflows.experiment_manager import ExperimentManager
        manager = ExperimentManager()
        console.print("[cyan]Before archiving:[/cyan]")
        manager.print_manifest_stats()
        console.print()
        archived = manager.archive_incomplete_runs()
        if archived > 0:
            console.print()
            console.print("[cyan]After archiving:[/cyan]")
            manager.print_manifest_stats()
        return 0
    except Exception as e:
        console.print(f"[red]Archive failed: {e}[/red]")
        return 1


def main_interactive() -> int:
    """Show help message directing users to notebooks."""
    console.print("[cyan]Schedule Engine - Notebook-Based Workflow[/cyan]")
    console.print()
    console.print("Primary workflow is through Jupyter notebooks:")
    console.print("  notebooks/mode_a_baseline.ipynb    - Pure NSGA-II")
    console.print("  notebooks/mode_b_memetic.ipynb     - + Memetic search")
    console.print("  notebooks/mode_c_roundrobin.ipynb  - + Round-robin heuristics")
    console.print("  notebooks/mode_d_adaptive.ipynb    - + Adaptive selection")
    console.print("  notebooks/mode_e_rl_guided.ipynb   - + RL-guided selection")
    console.print()
    console.print("[cyan]Available CLI utilities:[/cyan]")
    console.print("  uv run diagnose        - System diagnostics")
    console.print("  uv run clean           - Clean output directory")
    console.print("  uv run list-experiments - List experiment history")
    console.print("  uv run stats           - Show manifest statistics")
    console.print("  uv run archive         - Archive incomplete runs")
    console.print("  uv run lint            - Run code linter")
    console.print("  uv run typecheck       - Run type checker")
    return 0


if __name__ == "__main__":
    sys.exit(main_interactive())
