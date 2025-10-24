import matplotlib.pyplot as plt
import os
import csv
from typing import Dict, List
from .thesis_style import (
    apply_thesis_style,
    get_color,
    PALETTE,
    LINE_STYLES,
    MARKERS,
    save_figure,
    create_thesis_figure,
    format_axis,
)

# Apply thesis styling
apply_thesis_style()


def plot_individual_hard_constraints(
    hard_trends: Dict[str, List[int]], output_dir: str
):
    """
    Plots each hard constraint trend separately and saves them in hard/ subdirectory.

    Args:
        hard_trends: Dictionary mapping constraint names to their trends over generations
        output_dir: Base output directory
    """
    hard_dir = os.path.join(output_dir, "hard")
    os.makedirs(hard_dir, exist_ok=True)

    # Create CSVs subdirectory
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(csv_dir, exist_ok=True)

    # Individual plots for each hard constraint
    for constraint_name, trend in hard_trends.items():
        # Save individual constraint data to CSV
        csv_path = os.path.join(csv_dir, f"hard_{constraint_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", constraint_name])
            for gen, value in enumerate(trend):
                writer.writerow([gen, value])

        fig, ax = create_thesis_figure(1, 1, figsize=(10, 5.5))

        # Main trend line
        ax.plot(
            trend,
            color=get_color("red"),
            linewidth=2.5,
            marker="o",
            markersize=5,
            markevery=max(1, len(trend) // 15),
            label=constraint_name.replace("_", " ").title(),
        )

        # Add statistics
        final_value = trend[-1]
        max_value = max(trend)
        min_value = min(trend)
        avg_value = sum(trend) / len(trend)

        # Add horizontal lines for statistics
        ax.axhline(
            y=final_value,
            color=get_color("red"),
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label=f"Final: {final_value}",
        )
        ax.axhline(
            y=max_value,
            color=get_color("orange"),
            linestyle=":",
            alpha=0.5,
            linewidth=1.5,
            label=f"Max: {max_value}",
        )
        ax.axhline(
            y=avg_value,
            color=get_color("gray"),
            linestyle="-.",
            alpha=0.5,
            linewidth=1.5,
            label=f"Avg: {avg_value:.1f}",
        )

        format_axis(
            ax,
            xlabel="Generation (0 = Initial Population)",
            ylabel="Violations",
            title=f"Hard Constraint Trend: {constraint_name.replace('_', ' ').title()}",
            legend=True,
        )

        plt.tight_layout()

        # Save individual plot
        filename = f"{constraint_name}_trend.pdf"
        save_figure(fig, os.path.join(hard_dir, filename))

    # Save combined hard constraints data to CSV
    csv_path = os.path.join(csv_dir, "hard_constraints_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Generation"] + [name for name in hard_trends.keys()]
        writer.writerow(header)
        num_generations = len(next(iter(hard_trends.values())))
        for gen in range(num_generations):
            row = [gen] + [hard_trends[name][gen] for name in hard_trends.keys()]
            writer.writerow(row)

    # Combined plot with all hard constraints
    fig, ax = create_thesis_figure(1, 1, figsize=(12, 7))

    for i, (constraint_name, trend) in enumerate(hard_trends.items()):
        color = PALETTE[i % len(PALETTE)]
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(
            trend,
            label=constraint_name.replace("_", " ").title(),
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=0.85,
            marker=marker,
            markersize=5,
            markevery=max(1, len(trend) // 10),  # Show markers at intervals
        )

    format_axis(
        ax,
        xlabel="Generation (0 = Initial Population)",
        ylabel="Violations",
        title="All Hard Constraints Trends",
        legend=True,
    )

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    save_figure(fig, os.path.join(hard_dir, "all_hard_constraints.pdf"))

    # Create a summary statistics table plot
    fig, ax = create_thesis_figure(1, 1, figsize=(11, 6.5))
    constraint_names = list(hard_trends.keys())
    final_values = [trend[-1] for trend in hard_trends.values()]
    max_values = [max(trend) for trend in hard_trends.values()]
    avg_values = [sum(trend) / len(trend) for trend in hard_trends.values()]

    x = range(len(constraint_names))
    width = 0.25

    ax.bar(
        [i - width for i in x],
        final_values,
        width,
        label="Final",
        color=get_color("red"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        x,
        max_values,
        width,
        label="Maximum",
        color=get_color("orange"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        [i + width for i in x],
        avg_values,
        width,
        label="Average",
        color=get_color("gray"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )

    format_axis(
        ax,
        xlabel="Constraints",
        ylabel="Violations",
        title="Hard Constraints Statistics Summary",
        legend=True,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [name.replace("_", "\n") for name in constraint_names],
        rotation=45,
        ha="right",
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(fig, os.path.join(hard_dir, "hard_constraints_summary.pdf"))


def plot_individual_soft_constraints(
    soft_trends: Dict[str, List[int]], output_dir: str
):
    """
    Plots each soft constraint trend separately and saves them in soft/ subdirectory.

    Args:
        soft_trends: Dictionary mapping constraint names to their trends over generations
        output_dir: Base output directory
    """
    soft_dir = os.path.join(output_dir, "soft")
    os.makedirs(soft_dir, exist_ok=True)

    # Create CSVs subdirectory
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(csv_dir, exist_ok=True)

    # Individual plots for each soft constraint
    for constraint_name, trend in soft_trends.items():
        # Save individual constraint data to CSV
        csv_path = os.path.join(csv_dir, f"soft_{constraint_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", constraint_name])
            for gen, value in enumerate(trend):
                writer.writerow([gen, value])

        fig, ax = create_thesis_figure(1, 1, figsize=(10, 5.5))

        # Main trend line
        ax.plot(
            trend,
            color=get_color("green"),
            linewidth=2.5,
            marker="s",
            markersize=5,
            markevery=max(1, len(trend) // 15),
            label=constraint_name.replace("_", " ").title(),
        )

        # Add statistics
        final_value = trend[-1]
        max_value = max(trend)
        min_value = min(trend)
        avg_value = sum(trend) / len(trend)

        # Add horizontal lines for statistics
        ax.axhline(
            y=final_value,
            color=get_color("green"),
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label=f"Final: {final_value}",
        )
        ax.axhline(
            y=max_value,
            color=get_color("orange"),
            linestyle=":",
            alpha=0.5,
            linewidth=1.5,
            label=f"Max: {max_value}",
        )
        ax.axhline(
            y=avg_value,
            color=get_color("gray"),
            linestyle="-.",
            alpha=0.5,
            linewidth=1.5,
            label=f"Avg: {avg_value:.1f}",
        )

        format_axis(
            ax,
            xlabel="Generation (0 = Initial Population)",
            ylabel="Penalty",
            title=f"Soft Constraint Trend: {constraint_name.replace('_', ' ').title()}",
            legend=True,
        )

        plt.tight_layout()

        # Save individual plot
        filename = f"{constraint_name}_trend.pdf"
        save_figure(fig, os.path.join(soft_dir, filename))

    # Save combined soft constraints data to CSV
    csv_path = os.path.join(csv_dir, "soft_constraints_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Generation"] + [name for name in soft_trends.keys()]
        writer.writerow(header)
        num_generations = len(next(iter(soft_trends.values())))
        for gen in range(num_generations):
            row = [gen] + [soft_trends[name][gen] for name in soft_trends.keys()]
            writer.writerow(row)

    # Combined plot with all soft constraints
    fig, ax = create_thesis_figure(1, 1, figsize=(12, 7))

    for i, (constraint_name, trend) in enumerate(soft_trends.items()):
        color = PALETTE[i % len(PALETTE)]
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(
            trend,
            label=constraint_name.replace("_", " ").title(),
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=0.85,
            marker=marker,
            markersize=5,
            markevery=max(1, len(trend) // 10),  # Show markers at intervals
        )

    format_axis(
        ax,
        xlabel="Generation (0 = Initial Population)",
        ylabel="Penalty",
        title="All Soft Constraints Trends",
        legend=True,
    )

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    save_figure(fig, os.path.join(soft_dir, "all_soft_constraints.pdf"))

    # Create a summary statistics table plot
    fig, ax = create_thesis_figure(1, 1, figsize=(11, 6.5))
    constraint_names = list(soft_trends.keys())
    final_values = [trend[-1] for trend in soft_trends.values()]
    max_values = [max(trend) for trend in soft_trends.values()]
    avg_values = [sum(trend) / len(trend) for trend in soft_trends.values()]

    x = range(len(constraint_names))
    width = 0.25

    ax.bar(
        [i - width for i in x],
        final_values,
        width,
        label="Final",
        color=get_color("green"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        x,
        max_values,
        width,
        label="Maximum",
        color=get_color("orange"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        [i + width for i in x],
        avg_values,
        width,
        label="Average",
        color=get_color("gray"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )

    format_axis(
        ax,
        xlabel="Constraints",
        ylabel="Penalty",
        title="Soft Constraints Statistics Summary",
        legend=True,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [name.replace("_", "\n") for name in constraint_names],
        rotation=45,
        ha="right",
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(fig, os.path.join(soft_dir, "soft_constraints_summary.pdf"))


def plot_constraint_summary(
    hard_trends: Dict[str, List[int]],
    soft_trends: Dict[str, List[int]],
    output_dir: str,
):
    """
    Creates a summary dashboard showing total trends and final constraint values.
    """
    # Create CSVs subdirectory
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(csv_dir, exist_ok=True)

    # Calculate totals
    total_hard = [sum(values) for values in zip(*hard_trends.values())]
    total_soft = [sum(values) for values in zip(*soft_trends.values())]

    # Save summary data to CSV
    csv_path = os.path.join(csv_dir, "constraint_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Total_Hard_Violations", "Total_Soft_Penalties"])
        for gen, (h, s) in enumerate(zip(total_hard, total_soft)):
            writer.writerow([gen, h, s])

    fig, ((ax1, ax2), (ax3, ax4)) = create_thesis_figure(2, 2, figsize=(14, 10))

    # Total hard constraints trend
    ax1.plot(
        total_hard,
        color=get_color("red"),
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=max(1, len(total_hard) // 15),
    )
    format_axis(
        ax1,
        xlabel="Generation (0 = Initial Population)",
        ylabel="Total Violations",
        title="Total Hard Constraint Violations",
        legend=False,
    )

    # Total soft constraints trend
    ax2.plot(
        total_soft,
        color=get_color("green"),
        linewidth=2.5,
        marker="s",
        markersize=4,
        markevery=max(1, len(total_soft) // 15),
    )
    format_axis(
        ax2,
        xlabel="Generation (0 = Initial Population)",
        ylabel="Total Penalty",
        title="Total Soft Constraint Penalties",
        legend=False,
    )

    # Final hard constraint values (bar chart)
    final_hard = {name: trend[-1] for name, trend in hard_trends.items()}
    ax3.bar(
        range(len(final_hard)),
        list(final_hard.values()),
        color=get_color("red"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    format_axis(
        ax3,
        xlabel="",
        ylabel="Violations",
        title="Final Hard Constraint Violations",
        legend=False,
    )
    ax3.set_xticks(range(len(final_hard)))
    ax3.set_xticklabels(
        [name.replace("_", "\n") for name in final_hard.keys()], rotation=45, ha="right"
    )

    # Final soft constraint values (bar chart)
    final_soft = {name: trend[-1] for name, trend in soft_trends.items()}
    ax4.bar(
        range(len(final_soft)),
        list(final_soft.values()),
        color=get_color("green"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    format_axis(
        ax4,
        xlabel="",
        ylabel="Penalty",
        title="Final Soft Constraint Penalties",
        legend=False,
    )
    ax4.set_xticks(range(len(final_soft)))
    ax4.set_xticklabels(
        [name.replace("_", "\n") for name in final_soft.keys()], rotation=45, ha="right"
    )

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "constraint_summary.pdf"))
