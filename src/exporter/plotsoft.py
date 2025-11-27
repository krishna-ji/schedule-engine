import os

import matplotlib.pyplot as plt

from .thesis_style import (
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

# Apply thesis styling
apply_thesis_style()


def plot_soft_constraint_violation_over_generation(soft_trend, output_dir):
    """
    Plots the trend of soft constraint penalties over generations.

    Args:
        soft_trend (List[int]): List of soft constraint penalty counts per generation.
                                Index 0 = initial population, Index 1+ = evolved generations
        output_dir (str): Directory to save the plot.

    Note:
        CSV data available in data/metrics.csv (soft_total column)
    """
    fig, ax = create_thesis_figure(1, 1, figsize=(9, 5))
    ax.plot(
        soft_trend,
        color=get_color("green"),
        linewidth=2.5,
        label="Soft Constraint Penalties",
        marker="s",
        markersize=4,
        markevery=max(1, len(soft_trend) // 15),
    )
    # ax.set_yscale("log")  # Uncomment if needed

    format_axis(
        ax,
        xlabel="Generation (0 = Initial Population)",
        ylabel="Penalty",
        title="Soft Constraint Penalties Over Generations",
        legend=True,
    )

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "soft_constraint_trend.pdf"))
