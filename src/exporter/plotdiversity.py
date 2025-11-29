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


def plot_diversity_trend(diversity_trend: list, output_dir: str) -> None:
    """
    Plot population diversity over generations.

    Note:
        CSV data available in data/metrics.csv (diversity column)
    """
    fig, ax = create_thesis_figure(1, 1, figsize=(9, 5))
    ax.plot(
        diversity_trend,
        color=get_color("orange"),
        linewidth=2.5,
        label="Population Diversity",
        marker="^",
        markersize=4,
        markevery=max(1, len(diversity_trend) // 15),
    )

    format_axis(
        ax,
        xlabel="Generation (0 = Initial Population)",
        ylabel="Average Chromosome Distance",
        title="Population Diversity Over Generations",
        legend=False,
    )

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "diversity.pdf"))
