import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
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

    # Annotate knee point (max distance from line between extremes)
    knee_point: tuple[float, float] | None = None
    if len(pareto_front) >= 3:
        sorted_points = sorted(
            zip(pareto_hard, pareto_soft, strict=False), key=lambda p: (p[0], p[1])
        )
        h_vals = np.array([p[0] for p in sorted_points], dtype=float)
        s_vals = np.array([p[1] for p in sorted_points], dtype=float)
        h_min, h_max = float(np.min(h_vals)), float(np.max(h_vals))
        s_min, s_max = float(np.min(s_vals)), float(np.max(s_vals))
        h_range = h_max - h_min if h_max > h_min else 1.0
        s_range = s_max - s_min if s_max > s_min else 1.0
        h_norm = (h_vals - h_min) / h_range
        s_norm = (s_vals - s_min) / s_range
        x1, y1 = h_norm[0], s_norm[0]
        x2, y2 = h_norm[-1], s_norm[-1]
        denom = math.hypot(y2 - y1, x2 - x1)
        if denom > 0:
            distances = (
                np.abs((y2 - y1) * h_norm - (x2 - x1) * s_norm + x2 * y1 - y2 * x1)
                / denom
            )
            knee_idx = int(np.argmax(distances))
            knee_point = (h_vals[knee_idx], s_vals[knee_idx])

    if knee_point is not None:
        ax.scatter(
            [knee_point[0]],
            [knee_point[1]],
            color=get_color("purple"),
            s=160,
            marker="*",
            edgecolors="black",
            linewidth=1.0,
            label="Knee Point",
            zorder=6,
        )

    # Annotate best feasible tradeoff (hard == 0, lowest soft)
    feasible_points = [
        (h, s) for h, s in zip(pareto_hard, pareto_soft, strict=False) if h == 0
    ]
    if feasible_points:
        best_feasible = min(feasible_points, key=lambda p: p[1])
        ax.scatter(
            [best_feasible[0]],
            [best_feasible[1]],
            color=get_color("green"),
            s=120,
            marker="D",
            edgecolors="black",
            linewidth=1.0,
            label="Best Feasible Tradeoff",
            zorder=6,
        )

    format_axis(
        ax,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title=f"Final Population Fitness Distribution\n({len(population)} individuals, "
        f"{len({(h, s) for h, s in zip(hard_vals, soft_vals, strict=False)})} unique solutions)",
        legend=True,
    )
    ax.set_xlim(left=0)

    plt.tight_layout()
    save_figure(fig, plot_dir / "pareto_front_population_and_nondominated.pdf")
