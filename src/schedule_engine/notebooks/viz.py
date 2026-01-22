"""
Visualization Utilities for Notebooks.

Provides plotting and summary functions for notebook experiments:
- Convergence plots (hard/soft constraints over generations)
- Constraint breakdown visualizations
- Summary printing utilities

DRY Principle: All notebooks import visualization functions from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from schedule_engine.notebooks.core import EvolutionStats

__all__ = [
    "plot_convergence",
    "plot_constraint_breakdown",
    "print_summary",
]


def plot_convergence(
    stats: EvolutionStats,
    save_path: Path | str | None = None,
    title_prefix: str = "",
    show: bool = True,
) -> None:
    """
    Plot evolution convergence curves.

    Creates a 2-panel figure:
    - Left: Hard constraint violations over generations
    - Right: Soft constraint penalty over generations

    Args:
        stats: EvolutionStats object with recorded metrics
        save_path: Optional path to save the plot
        title_prefix: Prefix for plot titles (e.g., "Mode A: ")
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Hard constraints
    ax1 = axes[0]
    ax1.plot(stats.generations, stats.min_hard, "b-", label="Min", linewidth=2)
    ax1.plot(stats.generations, stats.avg_hard, "g--", label="Avg", linewidth=1)
    ax1.fill_between(
        stats.generations,
        stats.min_hard,
        stats.max_hard,
        alpha=0.2,
        color="blue",
    )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Hard Violations")
    ax1.set_title(f"{title_prefix}Hard Constraint Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Soft constraints
    ax2 = axes[1]
    ax2.plot(stats.generations, stats.min_soft, "r-", label="Min", linewidth=2)
    ax2.plot(
        stats.generations,
        stats.avg_soft,
        "orange",
        linestyle="--",
        label="Avg",
        linewidth=1,
    )
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Soft Penalty")
    ax2.set_title(f"{title_prefix}Soft Constraint Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_constraint_breakdown(
    breakdown: dict[str, int | float],
    save_path: Path | str | None = None,
    title: str = "Constraint Violations",
    show: bool = True,
) -> None:
    """
    Plot constraint violation breakdown as a horizontal bar chart.

    Args:
        breakdown: Dict mapping constraint names to violation counts
        save_path: Optional path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    # Sort by value (descending)
    sorted_items = sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Color by constraint type (hard = red, soft = blue)
    hard_constraints = {
        "student_group_exclusivity",
        "instructor_exclusivity",
        "room_exclusivity",
        "instructor_qualifications",
        "room_suitability",
        "course_completeness",
    }
    colors = ["#ff6b6b" if name in hard_constraints else "#4dabf7" for name in names]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Violations / Penalty")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for i, (name, val) in enumerate(zip(names, values)):
        ax.text(
            val + 0.5 if val >= 0 else val - 0.5,
            i,
            (
                f"{val:.0f}"
                if isinstance(val, float) and val == int(val)
                else f"{val:.1f}"
            ),
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#ff6b6b", edgecolor="black", label="Hard Constraints"),
        Patch(facecolor="#4dabf7", edgecolor="black", label="Soft Constraints"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_summary(
    population: list[Any],
    stats: EvolutionStats,
    breakdown: dict[str, int | float] | None = None,
) -> None:
    """
    Print a formatted summary of evolution results.

    Args:
        population: Final population
        stats: Evolution statistics
        breakdown: Optional constraint breakdown for best individual
    """
    print("\n" + "=" * 60)
    print(" EVOLUTION SUMMARY")
    print("=" * 60)

    # Best fitness
    best = min(
        population, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
    )
    hard, soft = best.fitness.values

    print(f"\n Best Solution:")
    print(f"   Hard Violations: {hard:.0f}")
    print(f"   Soft Penalty:    {soft:.1f}")
    print(f"   Feasible:        {' Yes' if hard == 0 else ' No'}")

    # Population stats
    hard_vals = [ind.fitness.values[0] for ind in population]
    soft_vals = [ind.fitness.values[1] for ind in population]
    feasible_count = sum(1 for h in hard_vals if h == 0)

    print(f"\n Final Population (n={len(population)}):")
    print(
        f"   Feasible:     {feasible_count}/{len(population)} ({100*feasible_count/len(population):.1f}%)"
    )
    print(f"   Min Hard:     {min(hard_vals):.0f}")
    print(f"   Avg Hard:     {np.mean(hard_vals):.1f}")
    print(f"   Min Soft:     {min(soft_vals):.1f}")
    print(f"   Avg Soft:     {np.mean(soft_vals):.1f}")

    # Timing
    print(f"\n️ Execution Time: {stats.elapsed_time:.1f}s")

    # Constraint breakdown
    if breakdown:
        print(f"\n Best Solution Constraint Breakdown:")
        hard_total = 0
        soft_total = 0.0
        hard_constraints = {
            "student_group_exclusivity",
            "instructor_exclusivity",
            "room_exclusivity",
            "instructor_qualifications",
            "room_suitability",
            "course_completeness",
        }

        for name, value in sorted(breakdown.items()):
            is_hard = name in hard_constraints
            marker = "" if is_hard and value > 0 else "" if is_hard else ""
            print(
                f"   {marker} {name}: {value:.0f}"
                if value == int(value)
                else f"   {marker} {name}: {value:.1f}"
            )
            if is_hard:
                hard_total += int(value)
            else:
                soft_total += float(value)

        print(f"\n   Total Hard: {hard_total}, Total Soft: {soft_total:.1f}")

    print("\n" + "=" * 60)


def plot_feasibility_progress(
    stats: EvolutionStats,
    save_path: Path | str | None = None,
    title_prefix: str = "",
    show: bool = True,
) -> None:
    """
    Plot feasibility progress over generations.

    Args:
        stats: EvolutionStats object
        save_path: Optional path to save the plot
        title_prefix: Prefix for plot title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(stats.generations, stats.feasible_count, "g-", linewidth=2)
    ax.fill_between(
        stats.generations, 0, stats.feasible_count, alpha=0.3, color="green"
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Feasible Individuals")
    ax.set_title(f"{title_prefix}Feasibility Progress")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
