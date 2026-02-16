"""
Multi-Metric Convergence Visualization

Generates comprehensive convergence plots combining multiple metrics:
- Hypervolume, Spacing, IGD, Spread on normalized scales
- Convergence rate analysis
- Constraint satisfaction rate evolution
- Combined metric dashboard

These plots provide holistic view of algorithm performance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.output_paths import get_nsga_plot_dir

from .thesis_style import (
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

# Apply thesis styling
apply_thesis_style()


def plot_multi_metric_convergence(metrics_dict: dict, output_dir: str) -> None:
    """
    Plot multiple normalized metrics on single graph for comparison.

    Shows hypervolume, spacing, IGD, spread, etc. on normalized [0, 1] scale
    for easy comparison of convergence patterns.

    Args:
        metrics_dict: Dictionary with metric names as keys and histories as values
                     E.g., {"hypervolume": [...], "spacing": [...], "igd": [...]}
        output_dir: Directory to save plots

    Saves:
        - plots/nsga/convergence_multi_metric.pdf: Combined metric plot

    Note:
        CSV data available in csv/constraint_metrics.csv (individual metric columns)
    """
    if not metrics_dict:
        return

    plot_dir = get_nsga_plot_dir(output_dir)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Normalize all metrics to [0, 1] range
    normalized_metrics = {}
    max_length = 0

    for name, values in metrics_dict.items():
        if not values:
            continue
        max_length = max(max_length, len(values))
        values_array = np.array(values)

        # Normalize to [0, 1]
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        if max_val > min_val:
            normalized = (values_array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values_array)

        normalized_metrics[name] = normalized

    if not normalized_metrics:
        return

    # Create plot
    fig, ax = create_thesis_figure(1, 1, figsize=(12, 7))

    colors = [
        get_color("blue"),
        get_color("green"),
        get_color("red"),
        get_color("orange"),
        get_color("purple"),
    ]
    markers = ["o", "s", "^", "D", "v"]

    for idx, (name, values) in enumerate(normalized_metrics.items()):
        gens = list(range(len(values)))
        ax.plot(
            gens,
            values,
            color=colors[idx % len(colors)],
            linewidth=2.5,
            marker=markers[idx % len(markers)],
            markersize=4,
            markevery=max(1, len(gens) // 15),
            label=name.replace("_", " ").title(),
            alpha=0.8,
        )

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Normalized Metric Value",
        title="Multi-Metric Convergence Analysis\n(All metrics normalized to [0, 1] scale)",
        legend=True,
    )

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)

    output_path = str(Path(plot_dir) / "convergence_multi_metric.pdf")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_convergence_dashboard(
    hard_violations: list,
    soft_penalties: list,
    diversity: list,
    hypervolume: list,
    spacing: list,
    feasibility_rate: list,
    output_dir: str,
) -> None:
    """
    Create comprehensive 2x3 dashboard of all key metrics.

    Provides complete overview of algorithm performance in single figure.

    Args:
        hard_violations: Hard constraint violations per generation
        soft_penalties: Soft penalties per generation
        diversity: Population diversity per generation
        hypervolume: Hypervolume per generation
        spacing: Spacing per generation
        feasibility_rate: Constraint satisfaction rate per generation
        output_dir: Directory to save plots

    Saves:
        - plots/nsga/convergence_dashboard.pdf: Complete dashboard
    """
    plot_dir = get_nsga_plot_dir(output_dir)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Create 2x3 subplot grid
    fig, axes = create_thesis_figure(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    generations = list(range(len(hard_violations)))

    # Plot 1: Hard Violations
    axes[0].plot(generations, hard_violations, color=get_color("red"), linewidth=2.5)
    axes[0].fill_between(
        generations, hard_violations, alpha=0.2, color=get_color("red")
    )
    format_axis(
        axes[0],
        xlabel="Generation",
        ylabel="Hard Violations",
        title="Hard Constraint Convergence",
        legend=False,
    )
    axes[0].grid(True, alpha=0.3, linestyle="--")

    # Plot 2: Soft Penalties
    if soft_penalties:
        axes[1].plot(
            generations[: len(soft_penalties)],
            soft_penalties,
            color=get_color("orange"),
            linewidth=2.5,
        )
        axes[1].fill_between(
            generations[: len(soft_penalties)],
            soft_penalties,
            alpha=0.2,
            color=get_color("orange"),
        )
    format_axis(
        axes[1],
        xlabel="Generation",
        ylabel="Soft Penalty",
        title="Soft Constraint Optimization",
        legend=False,
    )
    axes[1].grid(True, alpha=0.3, linestyle="--")

    # Plot 3: Diversity
    if diversity:
        axes[2].plot(
            generations[: len(diversity)],
            diversity,
            color=get_color("purple"),
            linewidth=2.5,
        )
        axes[2].fill_between(
            generations[: len(diversity)],
            diversity,
            alpha=0.2,
            color=get_color("purple"),
        )
    format_axis(
        axes[2],
        xlabel="Generation",
        ylabel="Diversity",
        title="Population Diversity",
        legend=False,
    )
    axes[2].grid(True, alpha=0.3, linestyle="--")

    # Plot 4: Hypervolume
    if hypervolume:
        axes[3].plot(
            generations[: len(hypervolume)],
            hypervolume,
            color=get_color("blue"),
            linewidth=2.5,
        )
        axes[3].fill_between(
            generations[: len(hypervolume)],
            hypervolume,
            alpha=0.2,
            color=get_color("blue"),
        )
    format_axis(
        axes[3],
        xlabel="Generation",
        ylabel="Hypervolume",
        title="Hypervolume Indicator",
        legend=False,
    )
    axes[3].grid(True, alpha=0.3, linestyle="--")

    # Plot 5: Spacing
    if spacing:
        axes[4].plot(
            generations[: len(spacing)],
            spacing,
            color=get_color("green"),
            linewidth=2.5,
        )
        axes[4].fill_between(
            generations[: len(spacing)], spacing, alpha=0.2, color=get_color("green")
        )
    format_axis(
        axes[4],
        xlabel="Generation",
        ylabel="Spacing",
        title="Solution Spacing",
        legend=False,
    )
    axes[4].grid(True, alpha=0.3, linestyle="--")

    # Plot 6: Feasibility Rate
    if feasibility_rate:
        axes[5].plot(
            generations[: len(feasibility_rate)],
            feasibility_rate,
            color=get_color("cyan"),
            linewidth=2.5,
            marker="o",
            markersize=3,
            markevery=max(1, len(feasibility_rate) // 20),
        )
        axes[5].fill_between(
            generations[: len(feasibility_rate)],
            feasibility_rate,
            alpha=0.2,
            color=get_color("cyan"),
        )
        axes[5].set_ylim([0, 105])  # 0-100% scale
    format_axis(
        axes[5],
        xlabel="Generation",
        ylabel="Feasibility Rate (%)",
        title="Constraint Satisfaction Rate",
        legend=False,
    )
    axes[5].grid(True, alpha=0.3, linestyle="--")

    plt.suptitle(
        "Comprehensive Convergence Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_path = str(Path(plot_dir) / "convergence_dashboard.pdf")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_convergence_rate(
    metric_history: list, output_dir: str, metric_name: str = "Hard Violations"
) -> None:
    """
    Plot convergence rate (improvement per generation) over time.

    Shows optimization dynamics: positive rate = improvement, near-zero = stagnation.

    Args:
        metric_history: Metric values per generation
        output_dir: Directory to save plots
        metric_name: Name of metric being analyzed

    Saves:
        - plots/nsga/convergence_rate_{metric_name}.pdf: Rate analysis plot
    """
    if len(metric_history) < 11:  # Need at least 11 generations for window=10
        return

    plot_dir = get_nsga_plot_dir(output_dir)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Calculate convergence rate with window=10
    window = 10
    rates = []
    for i in range(len(metric_history) - window):
        improvement = metric_history[i] - metric_history[i + window]
        rate = improvement / window
        rates.append(rate)

    if not rates:
        return

    rate_generations = list(range(window, len(metric_history)))

    # Create plot
    fig, (ax1, ax2) = create_thesis_figure(2, 1, figsize=(12, 10))

    # Top plot: Original metric
    ax1.plot(
        range(len(metric_history)),
        metric_history,
        color=get_color("blue"),
        linewidth=2.5,
        label=metric_name,
    )
    ax1.fill_between(
        range(len(metric_history)),
        metric_history,
        alpha=0.2,
        color=get_color("blue"),
    )
    format_axis(
        ax1,
        xlabel="Generation",
        ylabel=metric_name,
        title=f"{metric_name} Evolution",
        legend=True,
    )
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Bottom plot: Convergence rate
    # Color-code: green=improving, red=degrading, yellow=stagnant
    colors = [
        (
            get_color("green")
            if r > 0.01
            else (get_color("red") if r < -0.01 else get_color("yellow"))
        )
        for r in rates
    ]

    ax2.bar(rate_generations, rates, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # Add annotations for stagnation periods
    stagnant_count = sum(1 for r in rates if abs(r) <= 0.01)
    if stagnant_count > 0:
        stagnant_pct = (stagnant_count / len(rates)) * 100
        ax2.text(
            0.98,
            0.98,
            f"Stagnation: {stagnant_pct:.1f}% of generations",
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
        )

    format_axis(
        ax2,
        xlabel="Generation",
        ylabel=f"Improvement Rate\n(Δ{metric_name}/gen)",
        title="Convergence Rate Analysis\n(Green=Improving, Yellow=Stagnant, Red=Degrading)",
        legend=False,
    )
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()

    safe_name = metric_name.lower().replace(" ", "_")
    output_path = str(Path(plot_dir) / f"convergence_rate_{safe_name}.pdf")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_constraint_satisfaction_evolution(
    feasibility_rates: list, output_dir: str
) -> None:
    """
    Plot evolution of constraint satisfaction rate over generations.

    Shows percentage of feasible solutions in population.

    Args:
        feasibility_rates: List of feasibility percentages per generation
        output_dir: Directory to save plots

    Saves:
        - plots/nsga/feasibility_rate_over_generations.pdf: Feasibility rate plot
    """
    if not feasibility_rates:
        return

    plot_dir = get_nsga_plot_dir(output_dir)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    generations = list(range(len(feasibility_rates)))

    ax.plot(
        generations,
        feasibility_rates,
        color=get_color("cyan"),
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=max(1, len(generations) // 20),
    )

    ax.fill_between(
        generations,
        feasibility_rates,
        alpha=0.3,
        color=get_color("cyan"),
    )

    # Add 100% reference line
    ax.axhline(
        y=100,
        color="green",
        linestyle="--",
        linewidth=2,
        label="100% Feasible",
        alpha=0.5,
    )

    # Statistics
    final_rate = feasibility_rates[-1]
    max_rate = max(feasibility_rates)
    avg_rate = np.mean(feasibility_rates)

    textstr = (
        f"Final: {final_rate:.1f}%\nMax: {max_rate:.1f}%\nAverage: {avg_rate:.1f}%"
    )
    ax.text(
        0.02,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Feasibility Rate (%)",
        title="Constraint Satisfaction Rate Evolution\n(Percentage of Feasible Solutions)",
        legend=True,
    )

    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, linestyle="--")

    output_path = str(Path(plot_dir) / "feasibility_rate_over_generations.pdf")
    save_figure(fig, output_path)
    plt.close(fig)
