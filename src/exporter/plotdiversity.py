import os
import csv
import matplotlib.pyplot as plt
from .thesis_style import (
    apply_thesis_style,
    get_color,
    save_figure,
    create_thesis_figure,
    format_axis,
)

# Apply thesis styling
apply_thesis_style()


def plot_diversity_trend(diversity_trend, output_dir):
    # Create CSVs subdirectory
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(csv_dir, exist_ok=True)

    # Save data to CSV
    csv_path = os.path.join(csv_dir, "diversity_trend.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Average_Chromosome_Distance"])
        for gen, value in enumerate(diversity_trend):
            writer.writerow([gen, value])

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
