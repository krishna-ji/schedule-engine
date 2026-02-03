"""
Visualization Utilities for Notebooks.

Provides plotting and summary functions for notebook experiments:
- Convergence plots (hard/soft constraints over generations)
- Constraint breakdown visualizations
- Summary printing utilities

DRY Principle: All notebooks import visualization functions from here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from schedule_engine.notebooks.core import EvolutionStats

__all__ = [
    "plot_convergence",
    "plot_constraint_breakdown",
    "plot_pareto_front",
    "plot_diversity_metrics",
    "plot_constraint_trends",
    "plot_feasibility_progress",
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
    logger: logging.Logger | None = None,
) -> None:
    """
    Print a formatted summary of evolution results.

    Args:
        population: Final population
        stats: Evolution statistics
        breakdown: Optional constraint breakdown for best individual
        logger: Optional logger to write summary (defaults to print)
    """
    lines: list[str] = []
    lines.append("\n" + "=" * 60)
    lines.append(" EVOLUTION SUMMARY")

    # Best fitness
    best = min(
        population, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
    )
    hard, soft = best.fitness.values

    lines.append(f"\n Best Solution:")
    lines.append(f"   Hard Violations: {hard:.0f}")
    lines.append(f"   Soft Penalty:    {soft:.1f}")
    lines.append(f"   Feasible:        {' Yes' if hard == 0 else ' No'}")

    # Population stats
    hard_vals = [ind.fitness.values[0] for ind in population]
    soft_vals = [ind.fitness.values[1] for ind in population]
    feasible_count = sum(1 for h in hard_vals if h == 0)

    lines.append(f"\n Final Population (n={len(population)}):")
    lines.append(
        f"   Feasible:     {feasible_count}/{len(population)} ({100*feasible_count/len(population):.1f}%)"
    )
    lines.append(f"   Min Hard:     {min(hard_vals):.0f}")
    lines.append(f"   Avg Hard:     {np.mean(hard_vals):.1f}")
    lines.append(f"\n Min Soft:     {min(soft_vals):.1f}")
    lines.append(f"   Avg Soft:     {np.mean(soft_vals):.1f}")

    # Timing
    lines.append(f"\nExecution Time: {stats.elapsed_time:.1f}s")

    # Constraint breakdown
    if breakdown:
        lines.append(f"\n Best Solution Constraint Breakdown:")
        from schedule_engine.constraints.registry import (
            get_all_hard_constraints,
            get_all_soft_constraints,
        )

        hard_registry = get_all_hard_constraints()
        soft_registry = get_all_soft_constraints()
        hard_names = set(hard_registry.keys())
        soft_names = set(soft_registry.keys())

        hard_total_raw = 0.0
        soft_total_raw = 0.0
        hard_total_weighted = 0.0
        soft_total_weighted = 0.0

        for name, value in sorted(breakdown.items()):
            is_hard = name in hard_names
            marker = "" if is_hard else ""
            lines.append(
                f"   {marker} {name}: {value:.0f}"
                if value == int(value)
                else f"   {marker} {name}: {value:.1f}"
            )
            if is_hard:
                hard_total_raw += float(value)
                weight = hard_registry.get(name).default_weight if hard_registry.get(name) else 1.0
                hard_total_weighted += float(value) * float(weight)
            elif name in soft_names:
                soft_total_raw += float(value)
                weight = soft_registry.get(name).default_weight if soft_registry.get(name) else 1.0
                soft_total_weighted += float(value) * float(weight)

        lines.append(
            "\n   Total Hard (raw): "
            f"{hard_total_raw:.1f}, Total Hard (weighted): {hard_total_weighted:.1f}"
        )
        lines.append(
            "   Total Soft (raw): "
            f"{soft_total_raw:.1f}, Total Soft (weighted): {soft_total_weighted:.1f}"
        )

    lines.append("\n" + "=" * 60)

    if logger is None:
        for line in lines:
            print(line)
    else:
        for line in lines:
            logger.info(line)


def plot_pareto_front(
    population: list[Any],
    save_path: Path | str | None = None,
    title: str = "Pareto Front (Objective Space)",
    show: bool = True,
) -> None:
    """
    Plot the Pareto front in objective space (Hard vs Soft violations).

    Highlights non-dominated solutions and shows population distribution.

    Args:
        population: Final population with fitness values
        save_path: Optional path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    from deap import tools

    # Extract all fitness values
    all_hard = [ind.fitness.values[0] for ind in population]
    all_soft = [ind.fitness.values[1] for ind in population]

    # Get Pareto front (non-dominated solutions)
    pareto_front = tools.sortNondominated(
        population, len(population), first_front_only=True
    )[0]
    pf_hard = [ind.fitness.values[0] for ind in pareto_front]
    pf_soft = [ind.fitness.values[1] for ind in pareto_front]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all solutions (faded)
    ax.scatter(all_hard, all_soft, c="lightgray", alpha=0.5, s=30, label="Dominated")

    # Plot Pareto front (highlighted)
    ax.scatter(
        pf_hard,
        pf_soft,
        c="red",
        s=80,
        marker="*",
        edgecolors="black",
        linewidths=0.5,
        label=f"Pareto Front (n={len(pareto_front)})",
        zorder=5,
    )

    # Connect Pareto front points
    sorted_pf = sorted(zip(pf_hard, pf_soft), key=lambda x: x[0])
    pf_hard_sorted = [p[0] for p in sorted_pf]
    pf_soft_sorted = [p[1] for p in sorted_pf]
    ax.plot(pf_hard_sorted, pf_soft_sorted, "r--", alpha=0.5, linewidth=1)

    # Mark feasible region
    ax.axvline(
        x=0, color="green", linestyle=":", alpha=0.7, label="Feasibility Boundary"
    )

    # Best solution
    best = min(
        population, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
    )
    ax.scatter(
        [best.fitness.values[0]],
        [best.fitness.values[1]],
        c="blue",
        s=200,
        marker="D",
        edgecolors="black",
        linewidths=2,
        label=f"Best: ({best.fitness.values[0]:.0f}, {best.fitness.values[1]:.0f})",
        zorder=10,
    )

    ax.set_xlabel("Hard Constraint Violations", fontsize=12)
    ax.set_ylabel("Soft Constraint Penalty", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")
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


def plot_diversity_metrics(
    diversity_history: list[float],
    generations: list[int] | None = None,
    save_path: Path | str | None = None,
    title: str = "Population Diversity Over Generations",
    show: bool = True,
) -> None:
    """
    Plot diversity metrics over generations.

    Args:
        diversity_history: List of diversity values per generation
        generations: List of generation numbers (defaults to 0..n-1)
        save_path: Optional path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    if generations is None:
        generations = list(range(len(diversity_history)))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(generations, diversity_history, "b-", linewidth=2, label="Diversity")
    ax.fill_between(generations, 0, diversity_history, alpha=0.2, color="blue")

    # Add trend line
    if len(diversity_history) > 10:
        z = np.polyfit(generations, diversity_history, 1)
        p = np.poly1d(z)
        ax.plot(
            generations,
            p(generations),
            "r--",
            linewidth=1,
            alpha=0.7,
            label=f"Trend (slope={z[0]:.4f})",
        )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Population Diversity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotations
    if diversity_history:
        ax.annotate(
            f"Start: {diversity_history[0]:.3f}",
            xy=(generations[0], diversity_history[0]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
        )
        ax.annotate(
            f"End: {diversity_history[-1]:.3f}",
            xy=(generations[-1], diversity_history[-1]),
            xytext=(-50, 10),
            textcoords="offset points",
            fontsize=9,
        )

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


def plot_constraint_trends(
    constraint_history: dict[str, list[float]],
    generations: list[int] | None = None,
    save_path: Path | str | None = None,
    title: str = "Constraint Violations Over Generations",
    show: bool = True,
) -> None:
    """
    Plot individual constraint trends over generations.

    Args:
        constraint_history: Dict mapping constraint names to list of values per gen
        generations: List of generation numbers
        save_path: Optional path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    if not constraint_history:
        print("[!] No constraint history to plot")
        return

    # Determine generations
    first_key = next(iter(constraint_history))
    if generations is None:
        generations = list(range(len(constraint_history[first_key])))

    # Separate hard and soft constraints
    hard_constraints = {
        "student_group_exclusivity",
        "instructor_exclusivity",
        "room_exclusivity",
        "instructor_qualifications",
        "room_suitability",
        "course_completeness",
        "instructor_time_availability",
        "room_time_availability",
    }

    hard_history = {
        k: v for k, v in constraint_history.items() if k in hard_constraints
    }
    soft_history = {
        k: v for k, v in constraint_history.items() if k not in hard_constraints
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Hard constraints
    ax1 = axes[0]
    for name, values in hard_history.items():
        short_name = name.replace("_", " ").title()[:20]
        ax1.plot(generations, values, linewidth=1.5, label=short_name)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Violations")
    ax1.set_title("Hard Constraints")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Soft constraints
    ax2 = axes[1]
    for name, values in soft_history.items():
        short_name = name.replace("_", " ").title()[:20]
        ax2.plot(generations, values, linewidth=1.5, label=short_name)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Penalty")
    ax2.set_title("Soft Constraints")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
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
