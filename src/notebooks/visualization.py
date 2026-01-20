"""Visualization utilities for experiment notebooks.

Provides plotting functions for results analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.notebooks.evolution import EvolutionStats


def _is_interactive() -> bool:
    """Check if matplotlib backend is interactive (can show plots)."""
    backend = plt.get_backend()
    # Non-interactive backends that shouldn't use plt.show()
    non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    return backend.lower() not in non_interactive


def plot_convergence(
    stats: EvolutionStats | dict[str, Any],
    output_path: str | Path | None = None,
    title_prefix: str = "",
    show: bool = True,
) -> plt.Figure:
    """Plot convergence curves (hard violations + feasible count).

    Args:
        stats: EvolutionStats or dict with keys: gen, min_hard, avg_hard, feasible
        output_path: Optional path to save figure
        title_prefix: Prefix for plot titles (e.g., "Mode A: ")
        show: Whether to display plot

    Returns:
        Matplotlib figure
    """
    # Handle both EvolutionStats and dict
    if hasattr(stats, "to_dict"):
        data = stats.to_dict()
    else:
        data = stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Hard constraint convergence
    ax1 = axes[0]
    ax1.plot(data["gen"], data["min_hard"], "b-", linewidth=2, label="Min")
    ax1.plot(data["gen"], data["avg_hard"], "b--", alpha=0.5, label="Avg")
    if "max_hard" in data:
        ax1.fill_between(
            data["gen"], data["min_hard"], data["max_hard"], alpha=0.1, color="blue"
        )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Hard Violations")
    ax1.set_title(f"{title_prefix}Hard Constraint Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Feasible count
    ax2 = axes[1]
    pop_size = max(data["feasible"]) if max(data["feasible"]) > 0 else 50
    ax2.fill_between(data["gen"], data["feasible"], alpha=0.3, color="green")
    ax2.plot(data["gen"], data["feasible"], "g-", linewidth=2)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Feasible Count")
    ax2.set_title(f"{title_prefix}Feasible Solutions (/{pop_size})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f" Saved: {output_path}")

    if show and _is_interactive():
        plt.show()
    elif output_path:
        # Close figure to free memory when saving and not showing
        plt.close(fig)

    return fig


def plot_constraint_breakdown(
    breakdown: dict[str, int],
    output_path: str | Path | None = None,
    title: str = "Constraint Breakdown",
    show: bool = True,
) -> plt.Figure:
    """Plot bar chart of constraint violations.

    Args:
        breakdown: Dict mapping constraint names to violation counts
        output_path: Optional path to save figure
        title: Plot title
        show: Whether to display plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(breakdown.keys())
    values = list(breakdown.values())
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(names)))

    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel("Violations")
    ax.set_title(title)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f" Saved: {output_path}")

    if show and _is_interactive():
        plt.show()
    elif output_path:
        # Close figure to free memory when saving and not showing
        plt.close(fig)

    return fig


def plot_comparison(
    results: dict[str, EvolutionStats | dict[str, Any]],
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot comparison of multiple experiment modes.

    Args:
        results: Dict mapping mode names to stats
        output_path: Optional path to save figure
        show: Whether to display plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Hard violations comparison
    ax1 = axes[0]
    for (name, stats), color in zip(results.items(), colors):
        data = stats.to_dict() if hasattr(stats, "to_dict") else stats
        ax1.plot(data["gen"], data["min_hard"], label=name, color=color, linewidth=2)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Min Hard Violations")
    ax1.set_title("Mode Comparison: Hard Constraints")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final performance bar chart
    ax2 = axes[1]
    names = list(results.keys())
    final_hard = []
    for stats in results.values():
        data = stats.to_dict() if hasattr(stats, "to_dict") else stats
        final_hard.append(data["min_hard"][-1] if data["min_hard"] else 0)

    bars = ax2.bar(names, final_hard, color=colors)
    ax2.set_ylabel("Final Hard Violations")
    ax2.set_title("Final Performance Comparison")

    for bar, val in zip(bars, final_hard):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f" Saved: {output_path}")

    if show and _is_interactive():
        plt.show()
    elif output_path:
        plt.close(fig)

    return fig


def print_summary(
    population: list[Any],
    stats: EvolutionStats | dict[str, Any],
    breakdown: dict[str, int] | None = None,
) -> None:
    """Print summary of evolution results.

    Args:
        population: Final population
        stats: Evolution statistics
        breakdown: Optional constraint breakdown
    """
    data = stats.to_dict() if hasattr(stats, "to_dict") else stats
    elapsed = getattr(stats, "elapsed_time", 0)

    best = min(population, key=lambda x: (x.fitness.values[0], x.fitness.values[1]))

    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"Best Solution: hard={best.fitness.values[0]}, soft={best.fitness.values[1]}"
    )
    print(
        f"Final Generation: min_hard={data['min_hard'][-1]:.0f}, "
        f"min_soft={data['min_soft'][-1]:.0f}, avg_hard={data['avg_hard'][-1]:.1f}"
    )
    print(f"Feasible Solutions: {data['feasible'][-1]}")
    print(f"Elapsed Time: {elapsed:.1f}s")

    if breakdown:
        # Separate hard and soft constraints
        hard_constraints = {
            k: v for k, v in breakdown.items() if not k.startswith("soft_")
        }
        soft_constraints = {k: v for k, v in breakdown.items() if k.startswith("soft_")}

        if hard_constraints:
            print("\nHard Constraint Violations:")
            for name, val in hard_constraints.items():
                print(f"  {name}: {val}")

        if soft_constraints:
            print("\nSoft Constraint Penalties:")
            for name, val in soft_constraints.items():
                # Remove 'soft_' prefix for cleaner display
                display_name = name.replace("soft_", "")
                print(f"  {display_name}: {val}")

    print("=" * 60)
