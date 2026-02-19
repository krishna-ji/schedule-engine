"""
Spacing Metric Visualization

Generates plots for spacing metric which measures uniformity of Pareto front distribution.
Lower spacing values indicate more uniform solution distribution, providing better
trade-off options for decision makers.

Visualizations include:
- Spacing trend over generations
- Distribution histogram of nearest-neighbor distances
- Spacing + Pareto front combined view
"""

import matplotlib.pyplot as plt
import numpy as np

from src.ga.metrics._nds import get_pareto_front
from src.utils.output_paths import get_nsga_plot_dir

from .thesis_style import (
    PALETTE,
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

# Apply thesis styling
apply_thesis_style()


def plot_spacing_trend(spacing_history: list, output_dir: str) -> None:
    """
    Plot spacing evolution over generations.

    Spacing measures uniformity of Pareto front distribution. Lower values
    indicate more evenly distributed solutions.

    Args:
        spacing_history: List of spacing values per generation
        output_dir: Directory to save plots

    Saves:
        - plots/nsga/spacing_metric_over_generations.pdf: Main trend plot

    Note:
        CSV data available in csv/constraint_metrics.csv (spacing column)
    """
    if not spacing_history:
        return

    # Create plot directory
    plot_dir = get_nsga_plot_dir(output_dir)

    # Create plot
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    generations = list(range(len(spacing_history)))

    # Plot line with markers
    ax.plot(
        generations,
        spacing_history,
        color=get_color("green"),
        linewidth=2.5,
        marker="s",
        markersize=4,
        markevery=max(1, len(generations) // 20),
        label="Spacing",
    )

    # Fill area to emphasize trend
    ax.fill_between(
        generations,
        spacing_history,
        alpha=0.2,
        color=get_color("green"),
    )

    # Add annotations for best (minimum) spacing
    min_spacing = min(spacing_history)
    min_gen = spacing_history.index(min_spacing)
    ax.annotate(
        f"Best: {min_spacing:.4f}",
        xy=(min_gen, min_spacing),
        xytext=(10, 10),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "yellow", "alpha": 0.7},
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0"},
        fontsize=10,
    )

    # Calculate improvement
    if len(spacing_history) > 1:
        initial_spacing = spacing_history[0]
        final_spacing = spacing_history[-1]
        improvement = (
            ((initial_spacing - final_spacing) / initial_spacing * 100)
            if initial_spacing > 0
            else 0
        )

        textstr = f"Initial: {initial_spacing:.4f}\nFinal: {final_spacing:.4f}\nImprovement: {improvement:.1f}%"
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Spacing Metric",
        title="Spacing Evolution\n(Lower = More Uniform Distribution)",
        legend=True,
    )

    ax.grid(True, alpha=0.3, linestyle="--")

    output_path = plot_dir / "spacing_metric_over_generations.pdf"
    save_figure(fig, output_path)
    plt.close(fig)


def plot_spacing_distribution(population: list, output_dir: str) -> None:
    """
    Plot histogram of nearest-neighbor distances in final Pareto front.

    This shows the distribution of spacing between solutions. More uniform
    histogram indicates better spacing.

    Args:
        population: Final population with fitness values
        output_dir: Directory to save plots

    Saves:
        - plots/nsga/spacing_distribution_final_pareto_front.pdf: Histogram of distances
    """
    if not population:
        return

    plot_dir = get_nsga_plot_dir(output_dir)

    # Extract Pareto front
    pareto_front = get_pareto_front(population)

    if len(pareto_front) < 2:
        return

    # Calculate nearest-neighbor distances
    fitnesses = np.array([ind.fitness.values for ind in pareto_front])
    distances = []

    for i, point in enumerate(fitnesses):
        other_points = np.delete(fitnesses, i, axis=0)
        dists = np.linalg.norm(other_points - point, axis=1)
        min_dist = np.min(dists)
        distances.append(min_dist)

    # Create histogram
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    ax.hist(
        distances,
        bins=min(20, len(distances)),
        color=get_color("green"),
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add statistics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    spacing = np.sqrt(
        np.sum((np.array(distances) - mean_dist) ** 2) / (len(distances) - 1)
    )

    ax.axvline(
        mean_dist,
        color=get_color("red"),
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_dist:.4f}",
    )

    textstr = f"Spacing: {spacing:.4f}\nMean: {mean_dist:.4f}\nStd: {std_dist:.4f}"
    ax.text(
        0.98,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    format_axis(
        ax,
        xlabel="Nearest-Neighbor Distance",
        ylabel="Frequency",
        title="Distribution of Solution Spacing in Final Pareto Front",
        legend=True,
    )

    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    output_path = plot_dir / "spacing_distribution_final_pareto_front.pdf"
    save_figure(fig, output_path)
    plt.close(fig)


def plot_spacing_with_pareto(
    population: list, spacing_history: list, output_dir: str
) -> None:
    """
    Combined visualization: Pareto front + spacing information.

    Shows Pareto front with spacing metric annotated, providing visual
    context for spacing quality.

    Args:
        population: Final population
        spacing_history: Spacing values per generation
        output_dir: Directory to save plots

    Saves:
        - plots/nsga/spacing_pareto_combined.pdf: Combined visualization
    """
    if not population or not spacing_history:
        return

    plot_dir = get_nsga_plot_dir(output_dir)

    # Extract Pareto front
    pareto_front = get_pareto_front(population)

    if len(pareto_front) < 2:
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = create_thesis_figure(1, 2, figsize=(14, 6))

    # Left plot: Pareto front with nearest-neighbor lines
    pareto_fitnesses = np.array([ind.fitness.values for ind in pareto_front])

    # Sort by first objective for connecting lines
    sorted_indices = np.argsort(pareto_fitnesses[:, 0])
    sorted_fitnesses = pareto_fitnesses[sorted_indices]

    # Plot Pareto front points
    ax1.scatter(
        sorted_fitnesses[:, 0],
        sorted_fitnesses[:, 1],
        color=get_color("red"),
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
        label="Pareto Front",
    )

    # Draw lines connecting consecutive points
    ax1.plot(
        sorted_fitnesses[:, 0],
        sorted_fitnesses[:, 1],
        color=get_color("green"),
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Connections",
    )

    # Add spacing annotation
    final_spacing = spacing_history[-1]
    ax1.text(
        0.02,
        0.98,
        f"Final Spacing: {final_spacing:.4f}\n(Lower = More Uniform)",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    format_axis(
        ax1,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title="Final Pareto Front with Spacing",
        legend=True,
    )
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Right plot: Spacing trend
    generations = list(range(len(spacing_history)))
    ax2.plot(
        generations,
        spacing_history,
        color=get_color("green"),
        linewidth=2.5,
        marker="s",
        markersize=3,
        markevery=max(1, len(generations) // 15),
    )
    ax2.fill_between(generations, spacing_history, alpha=0.2, color=get_color("green"))

    format_axis(
        ax2,
        xlabel="Generation",
        ylabel="Spacing Metric",
        title="Spacing Evolution",
        legend=False,
    )
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    output_path = plot_dir / "spacing_pareto_combined.pdf"
    save_figure(fig, output_path)
    plt.close(fig)


def plot_spacing_multi_run(
    runs_spacing: list, output_dir: str, run_labels: list | None = None
) -> None:
    """
    Plot spacing trends for multiple runs with confidence intervals.

    Args:
        runs_spacing: List of spacing histories, one per run
        output_dir: Directory to save plots
        run_labels: Optional labels for each run

    Saves:
        - plots/nsga/spacing_multi_run.pdf: Multi-run comparison
    """
    if not runs_spacing:
        return

    plot_dir = get_nsga_plot_dir(output_dir)

    # Pad runs to same length
    max_gens = max(len(run) for run in runs_spacing)
    padded_runs = []
    for run in runs_spacing:
        if len(run) < max_gens:
            padded = list(run) + [run[-1]] * (max_gens - len(run))
            padded_runs.append(padded)
        else:
            padded_runs.append(run)

    runs_array = np.array(padded_runs)
    generations = list(range(max_gens))

    # Calculate statistics
    mean_spacing = np.mean(runs_array, axis=0)
    std_spacing = np.std(runs_array, axis=0)
    n_runs = len(runs_spacing)
    confidence = 1.96 * (std_spacing / np.sqrt(n_runs))

    # Create plot
    fig, ax = create_thesis_figure(1, 1, figsize=(12, 7))

    # Plot individual runs
    for i, run in enumerate(runs_spacing):
        label = run_labels[i] if run_labels and i < len(run_labels) else f"Run {i + 1}"
        ax.plot(
            range(len(run)),
            run,
            color=PALETTE[i % len(PALETTE)],
            alpha=0.3,
            linewidth=1,
            label=label if n_runs <= 5 else None,
        )

    # Plot mean with CI
    ax.plot(
        generations,
        mean_spacing,
        color=get_color("green"),
        linewidth=3,
        marker="s",
        markersize=5,
        markevery=max(1, max_gens // 20),
        label=f"Mean (n={n_runs})",
        zorder=10,
    )

    ax.fill_between(
        generations,
        mean_spacing - confidence,
        mean_spacing + confidence,
        alpha=0.3,
        color=get_color("green"),
        label="95% CI",
    )

    # Statistics box
    final_mean = mean_spacing[-1]
    final_std = std_spacing[-1]
    textstr = f"Final Spacing:\nMean: {final_mean:.4f}\nStd: {final_std:.4f}"
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Spacing Metric",
        title=f"Spacing Evolution Across {n_runs} Independent Runs",
        legend=True,
    )

    ax.grid(True, alpha=0.3, linestyle="--")

    output_path = plot_dir / "spacing_multi_run.pdf"
    save_figure(fig, output_path)
    plt.close(fig)
