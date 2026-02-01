import csv
import os

import matplotlib.pyplot as plt
from deap import tools

from schedule_engine.utils.output_paths import get_csv_dir, get_nsga_plot_dir

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


def plot_pareto_front(population: list, output_dir: str) -> None:
    """
    Enhanced Pareto front visualization showing all points with better visibility.
    """
    hard_vals, soft_vals = zip(
        *[ind.fitness.values for ind in population], strict=False
    )

    # Route CSVs to consolidated csv/ directory
    csv_dir = get_csv_dir(output_dir)

    # Save population data to CSV
    csv_path = os.path.join(csv_dir, "population_fitness.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Individual_Index",
                "Hard_Constraint_Violations",
                "Soft_Constraint_Penalties",
            ]
        )
        for idx, (h, s) in enumerate(zip(hard_vals, soft_vals, strict=False)):
            writer.writerow([idx, h, s])

    # Save Pareto front data to CSV
    pareto_front = tools.sortNondominated(
        population, len(population), first_front_only=True
    )[0]
    pareto_hard = [ind.fitness.values[0] for ind in pareto_front]
    pareto_soft = [ind.fitness.values[1] for ind in pareto_front]

    csv_path = os.path.join(csv_dir, "pareto_front.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Pareto_Index", "Hard_Constraint_Violations", "Soft_Constraint_Penalties"]
        )
        for idx, (h, s) in enumerate(zip(pareto_hard, pareto_soft, strict=False)):
            writer.writerow([idx, h, s])

    plot_dir = get_nsga_plot_dir(output_dir)

    # Create the single Pareto front plot
    fig, ax = create_thesis_figure(1, 1, figsize=(9, 7))
    ax.scatter(
        hard_vals,
        soft_vals,
        color=PALETTE[1],
        alpha=0.35,
        s=30,
        label="Population",
        edgecolors="none",
    )
    ax.scatter(
        pareto_hard,
        pareto_soft,
        color=get_color("red"),
        alpha=0.9,
        s=90,
        label=f"Pareto Front ({len(pareto_front)} solutions)",
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )

    format_axis(
        ax,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title=f"Final Population Fitness Distribution\n({len(population)} individuals, "
        f"{len({(h, s) for h, s in zip(hard_vals, soft_vals, strict=False)})} unique solutions)",
        legend=True,
    )

    plt.tight_layout()
    save_figure(fig, plot_dir / "pareto_front_population_and_nondominated.pdf")
