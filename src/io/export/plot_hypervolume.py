"""
Hypervolume Indicator Visualization

Generates publication-quality plots for hypervolume metric tracking:
- Trend plot showing hypervolume evolution over generations
- Comparison plots for multiple runs
- Statistical visualizations

Hypervolume is the gold standard metric for multi-objective optimization,
combining convergence and diversity into a single value.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.output_paths import get_csv_dir, get_nsga_plot_dir

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


def plot_hypervolume_trend(hypervolume_history: list, output_dir: str) -> None:
    """
    Plot hypervolume evolution over generations.

    Shows how hypervolume increases as algorithm improves Pareto front.
    Higher values indicate better convergence and diversity.

    Args:
        hypervolume_history: List of hypervolume values per generation
        output_dir: Directory to save plots

    Saves:
        - plots/nsga/hypervolume_indicator_over_generations.pdf: Main trend plot

    Note:
        CSV data available in csv/constraint_metrics.csv (hypervolume column)
    """
    if not hypervolume_history:
        return

    # Create plot directory
    plot_dir = get_nsga_plot_dir(output_dir)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Create plot
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    generations = list(range(len(hypervolume_history)))

    # Plot line with markers
    ax.plot(
        generations,
        hypervolume_history,
        color=get_color("blue"),
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=max(1, len(generations) // 20),  # Show ~20 markers
        label="Hypervolume",
    )

    # Fill area under curve for visual emphasis
    ax.fill_between(
        generations,
        hypervolume_history,
        alpha=0.2,
        color=get_color("blue"),
    )

    # Add annotations for key points
    max_hv = max(hypervolume_history)
    max_gen = hypervolume_history.index(max_hv)
    ax.annotate(
        f"Best: {max_hv:.2f}",
        xy=(max_gen, max_hv),
        xytext=(10, 10),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "yellow", "alpha": 0.7},
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0"},
        fontsize=10,
    )

    # Calculate improvement
    if len(hypervolume_history) > 1:
        initial_hv = hypervolume_history[0]
        final_hv = hypervolume_history[-1]
        improvement = (
            ((final_hv - initial_hv) / initial_hv * 100) if initial_hv > 0 else 0
        )

        # Add improvement text
        textstr = f"Initial: {initial_hv:.2f}\nFinal: {final_hv:.2f}\nImprovement: {improvement:.1f}%"
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
        ylabel="Hypervolume Indicator",
        title="Hypervolume Evolution\n(Higher = Better Convergence + Diversity)",
        legend=True,
    )

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Save figure
    output_path = str(Path(plot_dir) / "hypervolume_indicator_over_generations.pdf")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_hypervolume_with_confidence(
    runs_hypervolumes: list, output_dir: str, run_labels: list | None = None
) -> None:
    """
    Plot hypervolume trends with confidence intervals for multiple runs.

    Shows mean hypervolume across runs with confidence bands, useful for
    statistical analysis and algorithm comparison.

    Args:
        runs_hypervolumes: List of hypervolume histories, one per run
        output_dir: Directory to save plots
        run_labels: Optional labels for each run (e.g., ["Run 1", "Run 2"])

    Saves:
        - plots/hypervolume_multi_run.pdf: Multi-run comparison
        - CSVs/hypervolume_statistics.csv: Statistical data
    """
    if not runs_hypervolumes:
        return

    plot_dir = get_nsga_plot_dir(output_dir)
    csv_dir = get_csv_dir(output_dir)

    # Ensure all runs have same length (pad with last value if needed)
    max_gens = max(len(run) for run in runs_hypervolumes)
    padded_runs = []
    for run in runs_hypervolumes:
        if len(run) < max_gens:
            padded = list(run) + [run[-1]] * (max_gens - len(run))
            padded_runs.append(padded)
        else:
            padded_runs.append(run)

    runs_array = np.array(padded_runs)
    generations = list(range(max_gens))

    # Calculate statistics
    mean_hv = np.mean(runs_array, axis=0)
    std_hv = np.std(runs_array, axis=0)
    min_hv = np.min(runs_array, axis=0)
    max_hv = np.max(runs_array, axis=0)

    # 95% confidence interval
    n_runs = len(runs_hypervolumes)
    confidence = 1.96 * (std_hv / np.sqrt(n_runs))  # 95% CI

    # Save statistics to CSV
    csv_path = Path(csv_dir) / "hypervolume_statistics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Generation", "Mean", "Std", "Min", "Max", "CI_Lower", "CI_Upper"]
        )
        for gen in range(max_gens):
            writer.writerow(
                [
                    gen,
                    mean_hv[gen],
                    std_hv[gen],
                    min_hv[gen],
                    max_hv[gen],
                    mean_hv[gen] - confidence[gen],
                    mean_hv[gen] + confidence[gen],
                ]
            )

    # Create plot
    fig, ax = create_thesis_figure(1, 1, figsize=(12, 7))

    # Plot individual runs (light lines)
    for i, run in enumerate(runs_hypervolumes):
        label = run_labels[i] if run_labels and i < len(run_labels) else f"Run {i + 1}"
        ax.plot(
            range(len(run)),
            run,
            color=PALETTE[i % len(PALETTE)],
            alpha=0.3,
            linewidth=1,
            label=(
                label if len(runs_hypervolumes) <= 5 else None
            ),  # Show legend only for ≤5 runs
        )

    # Plot mean with confidence interval
    ax.plot(
        generations,
        mean_hv,
        color=get_color("blue"),
        linewidth=3,
        marker="o",
        markersize=5,
        markevery=max(1, max_gens // 20),
        label=f"Mean (n={n_runs})",
        zorder=10,
    )

    ax.fill_between(
        generations,
        mean_hv - confidence,
        mean_hv + confidence,
        alpha=0.3,
        color=get_color("blue"),
        label="95% Confidence Interval",
    )

    # Add statistics box
    final_mean = mean_hv[-1]
    final_std = std_hv[-1]
    textstr = f"Final HV:\nMean: {final_mean:.2f}\nStd: {final_std:.2f}\nRuns: {n_runs}"
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
        ylabel="Hypervolume Indicator",
        title=f"Hypervolume Evolution Across {n_runs} Independent Runs",
        legend=True,
    )

    ax.grid(True, alpha=0.3, linestyle="--")

    output_path = str(Path(plot_dir) / "hypervolume_multi_run.pdf")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_hypervolume_comparison(
    algo1_hv: list,
    algo2_hv: list,
    output_dir: str,
    algo1_name: str = "Algorithm 1",
    algo2_name: str = "Algorithm 2",
) -> None:
    """
    Compare hypervolume evolution between two algorithms.

    Side-by-side comparison useful for evaluating algorithm improvements.

    Args:
        algo1_hv: Hypervolume history for first algorithm
        algo2_hv: Hypervolume history for second algorithm
        output_dir: Directory to save plots
        algo1_name: Label for first algorithm
        algo2_name: Label for second algorithm

    Saves:
        - plots/hypervolume_comparison.pdf: Comparison plot
    """
    plot_dir = get_nsga_plot_dir(output_dir)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = create_thesis_figure(1, 1, figsize=(12, 7))

    # Plot both algorithms
    gens1 = list(range(len(algo1_hv)))
    gens2 = list(range(len(algo2_hv)))

    ax.plot(
        gens1,
        algo1_hv,
        color=get_color("blue"),
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=max(1, len(gens1) // 15),
        label=algo1_name,
    )

    ax.plot(
        gens2,
        algo2_hv,
        color=get_color("red"),
        linewidth=2.5,
        marker="s",
        markersize=4,
        markevery=max(1, len(gens2) // 15),
        label=algo2_name,
    )

    # Add statistics
    final1 = algo1_hv[-1] if algo1_hv else 0
    final2 = algo2_hv[-1] if algo2_hv else 0
    diff = final1 - final2
    pct_diff = (diff / final2 * 100) if final2 > 0 else 0

    textstr = (
        f"{algo1_name} final: {final1:.2f}\n"
        f"{algo2_name} final: {final2:.2f}\n"
        f"Difference: {diff:+.2f} ({pct_diff:+.1f}%)"
    )

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
        ylabel="Hypervolume Indicator",
        title="Algorithm Comparison: Hypervolume Evolution",
        legend=True,
    )

    ax.grid(True, alpha=0.3, linestyle="--")

    output_path = str(Path(plot_dir) / "hypervolume_comparison.pdf")
    save_figure(fig, output_path)
    plt.close(fig)
