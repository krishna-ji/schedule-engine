import matplotlib.pyplot as plt
import os
import csv
from .thesis_style import (
    apply_thesis_style,
    get_color,
    save_figure,
    create_thesis_figure,
    format_axis,
)

# Apply thesis styling
apply_thesis_style()


def plot_hard_constraint_violation_over_generation(hard_trend, output_dir):
    """
    Plots the trend of hard constraint violations over generations.

    Args:
        hard_trend (List[int]): List of hard constraint violation counts per generation.
                                 Index 0 = initial population, Index 1+ = evolved generations
        output_dir (str): Directory to save the plot.
    """
    # Create CSVs subdirectory
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(csv_dir, exist_ok=True)

    # Save data to CSV
    csv_path = os.path.join(csv_dir, "hard_constraint_trend.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Hard_Constraint_Violations"])
        for gen, value in enumerate(hard_trend):
            writer.writerow([gen, value])

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
        xlabel="Generation (0 = Initial Population)",
        ylabel="Violations",
        title="Hard Constraint Violations Over Generations",
        legend=True,
    )

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "hard_constraint_trend.pdf"))
