"""
Statistical Metrics Comparison and Analysis

Generates statistical comparison plots for multiple runs:
- Box plots with confidence intervals
- Algorithm performance comparison
- Success rate analysis
- Statistical significance testing visualization

Essential for thesis/paper reporting to demonstrate algorithm reliability.
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .thesis_style import (
    apply_thesis_style,
    get_color,
    PALETTE,
    save_figure,
    create_thesis_figure,
    format_axis,
)

# Apply thesis styling
apply_thesis_style()


def plot_metrics_boxplot(runs_data: dict, output_dir: str, generation: int = -1):
    """
    Create box plot comparing final metric values across multiple runs.

    Shows distribution, outliers, and statistical spread for each metric.

    Args:
        runs_data: Dict with metric names as keys and list of run histories as values
                  E.g., {"hypervolume": [run1_hv, run2_hv, ...], "spacing": [...]}
        output_dir: Directory to save plots
        generation: Which generation to analyze (-1 = final)

    Saves:
        - plots/metrics_boxplot.pdf: Box plot comparison
        - CSVs/metrics_statistics.csv: Statistical summary
    """
    if not runs_data:
        return

    plot_dir = os.path.join(output_dir, "plots")
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Extract final values for each metric
    metrics_values = {}
    for metric_name, runs in runs_data.items():
        values = []
        for run in runs:
            if len(run) > 0:
                idx = generation if generation >= 0 else len(run) - 1
                if idx < len(run):
                    values.append(run[idx])
        if values:
            metrics_values[metric_name] = values

    if not metrics_values:
        return

    # Create box plot
    fig, ax = create_thesis_figure(1, 1, figsize=(12, 7))

    positions = list(range(len(metrics_values)))
    box_data = list(metrics_values.values())
    labels = [name.replace("_", " ").title() for name in metrics_values.keys()]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
    )

    # Color boxes
    colors = [
        get_color("blue"),
        get_color("green"),
        get_color("red"),
        get_color("orange"),
        get_color("purple"),
    ]
    for patch, color in zip(bp["boxes"], colors * len(bp["boxes"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Save statistics to CSV
    csv_path = os.path.join(csv_dir, "metrics_statistics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Metric", "Mean", "Median", "Std", "Min", "Max", "Q1", "Q3", "N_Runs"]
        )

        for metric_name, values in metrics_values.items():
            values_array = np.array(values)
            writer.writerow(
                [
                    metric_name,
                    np.mean(values_array),
                    np.median(values_array),
                    np.std(values_array, ddof=1),
                    np.min(values_array),
                    np.max(values_array),
                    np.percentile(values_array, 25),
                    np.percentile(values_array, 75),
                    len(values),
                ]
            )

    format_axis(
        ax,
        xlabel="Metric",
        ylabel="Value",
        title=f"Statistical Comparison Across Multiple Runs\n(Generation {generation if generation >= 0 else 'Final'})",
        legend=False,
    )

    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(plot_dir, "metrics_boxplot.pdf")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_algorithm_comparison(
    algo1_data: dict,
    algo2_data: dict,
    output_dir: str,
    algo1_name: str = "Algorithm 1",
    algo2_name: str = "Algorithm 2",
):
    """
    Compare two algorithm configurations with statistical significance.

    Creates side-by-side bar plot with error bars and significance markers.

    Args:
        algo1_data: Dict of metric histories for algorithm 1
                   E.g., {"hypervolume": [run1, run2, ...], ...}
        algo2_data: Dict of metric histories for algorithm 2
        output_dir: Directory to save plots
        algo1_name: Name/label for algorithm 1
        algo2_name: Name/label for algorithm 2

    Saves:
        - plots/algorithm_comparison.pdf: Comparison bar plot
        - CSVs/algorithm_comparison_stats.csv: Statistical test results
    """
    plot_dir = os.path.join(output_dir, "plots")
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Extract final values
    metrics = list(algo1_data.keys())
    algo1_means = []
    algo1_stds = []
    algo2_means = []
    algo2_stds = []
    p_values = []

    for metric in metrics:
        # Extract final values
        values1 = [run[-1] for run in algo1_data[metric] if len(run) > 0]
        values2 = [run[-1] for run in algo2_data[metric] if len(run) > 0]

        if values1 and values2:
            algo1_means.append(np.mean(values1))
            algo1_stds.append(np.std(values1, ddof=1))
            algo2_means.append(np.mean(values2))
            algo2_stds.append(np.std(values2, ddof=1))

            # t-test
            _, p_val = stats.ttest_ind(values1, values2)
            p_values.append(p_val)
        else:
            algo1_means.append(0)
            algo1_stds.append(0)
            algo2_means.append(0)
            algo2_stds.append(0)
            p_values.append(1.0)

    # Create bar plot
    fig, ax = create_thesis_figure(1, 1, figsize=(12, 7))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        algo1_means,
        width,
        yerr=algo1_stds,
        label=algo1_name,
        color=get_color("blue"),
        alpha=0.7,
        capsize=5,
    )

    bars2 = ax.bar(
        x + width / 2,
        algo2_means,
        width,
        yerr=algo2_stds,
        label=algo2_name,
        color=get_color("red"),
        alpha=0.7,
        capsize=5,
    )

    # Add significance markers
    for i, p_val in enumerate(p_values):
        if p_val < 0.001:
            marker = "***"
        elif p_val < 0.01:
            marker = "**"
        elif p_val < 0.05:
            marker = "*"
        else:
            marker = "ns"

        # Place marker above higher bar
        y_pos = max(algo1_means[i] + algo1_stds[i], algo2_means[i] + algo2_stds[i])
        ax.text(
            i,
            y_pos * 1.05,
            marker,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value (Mean ± Std)")
    ax.set_title(
        f"Algorithm Comparison: {algo1_name} vs {algo2_name}\n"
        f"(*** p<0.001, ** p<0.01, * p<0.05, ns=not significant)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metrics], rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()

    output_path = os.path.join(plot_dir, "algorithm_comparison.pdf")
    save_figure(fig, output_path)
    plt.close(fig)

    # Save statistics
    csv_path = os.path.join(csv_dir, "algorithm_comparison_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Metric",
                f"{algo1_name}_Mean",
                f"{algo1_name}_Std",
                f"{algo2_name}_Mean",
                f"{algo2_name}_Std",
                "P_Value",
                "Significant",
            ]
        )

        for i, metric in enumerate(metrics):
            writer.writerow(
                [
                    metric,
                    algo1_means[i],
                    algo1_stds[i],
                    algo2_means[i],
                    algo2_stds[i],
                    p_values[i],
                    "Yes" if p_values[i] < 0.05 else "No",
                ]
            )


def plot_success_rate_comparison(
    runs_hard_violations: list,
    output_dir: str,
    thresholds: list = [0, 10, 50, 100],
):
    """
    Plot success rates at different feasibility thresholds.

    Shows percentage of runs that achieved hard violations below each threshold.

    Args:
        runs_hard_violations: List of hard violation histories (one per run)
        output_dir: Directory to save plots
        thresholds: List of threshold values to check

    Saves:
        - plots/success_rate.pdf: Success rate bar plot
    """
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Calculate success rates
    success_rates = []
    for threshold in thresholds:
        successful = sum(1 for run in runs_hard_violations if min(run) <= threshold)
        rate = (successful / len(runs_hard_violations)) * 100
        success_rates.append(rate)

    # Create bar plot
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    bars = ax.bar(
        range(len(thresholds)),
        success_rates,
        color=get_color("green"),
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add percentage labels on bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Hard Constraint Threshold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(
        f"Algorithm Success Rate Across {len(runs_hard_violations)} Independent Runs\n"
        f"(Success = achieving violations ≤ threshold)"
    )
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"≤{t}" for t in thresholds])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()

    output_path = os.path.join(plot_dir, "success_rate.pdf")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_convergence_speed_comparison(
    runs_hard_violations: list,
    output_dir: str,
    target_value: float = 0,
):
    """
    Compare convergence speed: generations needed to reach target.

    Shows histogram of generations-to-target across runs.

    Args:
        runs_hard_violations: List of hard violation histories
        output_dir: Directory to save plots
        target_value: Target threshold (default 0 = feasible)

    Saves:
        - plots/convergence_speed.pdf: Histogram of convergence generations
    """
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Calculate generations to target for each run
    generations_to_target = []
    for run in runs_hard_violations:
        for gen, value in enumerate(run):
            if value <= target_value:
                generations_to_target.append(gen)
                break

    if not generations_to_target:
        print("Warning: No runs reached target value")
        return

    # Create histogram
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    ax.hist(
        generations_to_target,
        bins=min(20, len(generations_to_target)),
        color=get_color("blue"),
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add statistics
    mean_gens = np.mean(generations_to_target)
    median_gens = np.median(generations_to_target)
    min_gens = np.min(generations_to_target)
    max_gens = np.max(generations_to_target)

    ax.axvline(
        mean_gens,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_gens:.1f}",
    )
    ax.axvline(
        median_gens,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_gens:.1f}",
    )

    textstr = (
        f"Mean: {mean_gens:.1f} gens\n"
        f"Median: {median_gens:.1f} gens\n"
        f"Range: [{min_gens}, {max_gens}] gens\n"
        f"Success: {len(generations_to_target)}/{len(runs_hard_violations)} runs"
    )

    ax.text(
        0.98,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Generations to Reach Target")
    ax.set_ylabel("Frequency (Number of Runs)")
    ax.set_title(
        f"Convergence Speed Distribution\n(Target: Hard Violations ≤ {target_value})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()

    output_path = os.path.join(plot_dir, "convergence_speed.pdf")
    save_figure(fig, output_path)
    plt.close(fig)


def generate_statistical_summary_table(runs_data: dict, output_dir: str):
    """
    Generate comprehensive statistical summary table (CSV).

    Args:
        runs_data: Dict with metric names as keys, list of run histories as values
        output_dir: Directory to save CSV

    Saves:
        - CSVs/statistical_summary.csv: Complete statistical table
    """
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, "statistical_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Metric",
                "Mean",
                "Median",
                "Std",
                "Min",
                "Max",
                "Q1",
                "Q3",
                "95% CI Lower",
                "95% CI Upper",
                "N Runs",
            ]
        )

        for metric_name, runs in runs_data.items():
            # Extract final values
            values = [run[-1] for run in runs if len(run) > 0]

            if not values:
                continue

            values_array = np.array(values)
            mean_val = np.mean(values_array)
            median_val = np.median(values_array)
            std_val = np.std(values_array, ddof=1)
            min_val = np.min(values_array)
            max_val = np.max(values_array)
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)

            # 95% CI
            if len(values) > 1:
                ci = stats.t.interval(
                    0.95, len(values) - 1, loc=mean_val, scale=stats.sem(values_array)
                )
            else:
                ci = (mean_val, mean_val)

            writer.writerow(
                [
                    metric_name,
                    mean_val,
                    median_val,
                    std_val,
                    min_val,
                    max_val,
                    q1,
                    q3,
                    ci[0],
                    ci[1],
                    len(values),
                ]
            )

    print(f"Statistical summary saved to: {csv_path}")
