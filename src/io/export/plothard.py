import matplotlib.pyplot as plt

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


def plot_hard_constraint_violation_over_generation(
    hard_trend: list[int], output_dir: str
) -> None:
    """
    Plots the trend of total hard constraint violations over generations.

    Args:
        hard_trend (List[int]): List of hard constraint violation counts per generation.
                                 Index 0 = initial population, Index 1+ = evolved generations
        output_dir (str): Directory to save the plot.

    Note:
        CSV data available in csv/constraint_metrics.csv (hard_total column)
    """
    fig, ax = create_thesis_figure(1, 1, figsize=(9, 5))
    ax.plot(
        hard_trend,
        color=get_color("red"),
        linewidth=2.5,
        label="Hard Constraint Violations",
        marker="o",
        markersize=4,
        markevery=max(1, len(hard_trend) // 15),
    )
    # ax.set_yscale("log")  # Uncomment if needed

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Violations",
        title="Total Hard Constraint Violations Over Generations",
        legend=True,
    )

    plt.tight_layout()
    plot_dir = get_nsga_plot_dir(output_dir)
    save_figure(fig, plot_dir / "total_hard_constraint_violations_over_generations.pdf")
