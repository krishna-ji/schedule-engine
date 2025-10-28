import os
import csv
import matplotlib.pyplot as plt
from deap import tools
import numpy as np
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


def plot_pareto_front(population, output_dir):
    """
    Enhanced Pareto front visualization showing all points with better visibility.
    """
    hard_vals, soft_vals = zip(*[ind.fitness.values for ind in population])

    # Create CSVs subdirectory
    csv_dir = os.path.join(output_dir, "CSVs")
    os.makedirs(csv_dir, exist_ok=True)

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
        for idx, (h, s) in enumerate(zip(hard_vals, soft_vals)):
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
        for idx, (h, s) in enumerate(zip(pareto_hard, pareto_soft)):
            writer.writerow([idx, h, s])

    # Create comprehensive plots showing all data
    fig, ((ax1, ax2), (ax3, ax4)) = create_thesis_figure(2, 2, figsize=(14, 11))

    # Plot 1: All population points with jitter to show overlapping points
    unique_points = {}
    jittered_hard = []
    jittered_soft = []

    for h, s in zip(hard_vals, soft_vals):
        key = (h, s)
        if key in unique_points:
            unique_points[key] += 1
            # Add small jitter to overlapping points
            jitter_strength = 0.1 * unique_points[key]
            jittered_hard.append(h + np.random.normal(0, jitter_strength))
            jittered_soft.append(s + np.random.normal(0, jitter_strength))
        else:
            unique_points[key] = 1
            jittered_hard.append(h)
            jittered_soft.append(s)

    ax1.scatter(
        jittered_hard,
        jittered_soft,
        color=PALETTE[1],
        alpha=0.5,
        s=30,
        edgecolors="white",
        linewidth=0.5,
    )
    format_axis(
        ax1,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title=f"All Population Points with Jitter\n({len(population)} individuals)",
        legend=False,
    )

    # Plot 2: Original view with population and Pareto front
    ax2.scatter(
        hard_vals,
        soft_vals,
        color=PALETTE[1],
        alpha=0.35,
        s=25,
        label="Population",
        edgecolors="none",
    )

    # Use the Pareto front data already calculated above
    ax2.scatter(
        pareto_hard,
        pareto_soft,
        color=get_color("red"),
        alpha=0.85,
        s=80,
        label=f"Pareto Front ({len(pareto_front)} solutions)",
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )

    # Show count for overlapping points
    for (h, s), count in unique_points.items():
        if count > 1:
            ax2.annotate(
                f"{count}",
                (h, s),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    format_axis(
        ax2,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title=f"Population with Pareto Front\n({len(unique_points)} unique solutions)",
        legend=True,
    )

    # Plot 3: Heatmap of point density
    try:
        from scipy.stats import gaussian_kde

        # Check if data has sufficient variance for KDE
        hard_std = np.std(hard_vals)
        soft_std = np.std(soft_vals)
        has_sufficient_variance = (
            len(np.unique(hard_vals)) > 1
            and len(np.unique(soft_vals)) > 1
            and hard_std > 1e-6
            and soft_std > 1e-6
        )

        if has_sufficient_variance:
            # Create density heatmap
            hard_range = np.linspace(min(hard_vals), max(hard_vals), 50)
            soft_range = np.linspace(min(soft_vals), max(soft_vals), 50)
            H, S = np.meshgrid(hard_range, soft_range)
            positions = np.vstack([H.ravel(), S.ravel()])
            values = np.vstack([hard_vals, soft_vals])

            try:
                kernel = gaussian_kde(values)
                density = np.reshape(kernel(positions).T, H.shape)
            except np.linalg.LinAlgError:
                # Data is too degenerate for KDE, fall back to histogram
                has_sufficient_variance = False

        if has_sufficient_variance:
            # Successfully created KDE, plot it
            im = ax3.contourf(H, S, density, levels=20, cmap="Blues", alpha=0.6)
            ax3.scatter(
                hard_vals,
                soft_vals,
                color=get_color("blue"),
                alpha=0.4,
                s=12,
                edgecolors="none",
            )
            ax3.scatter(
                pareto_hard,
                pareto_soft,
                color=get_color("red"),
                s=50,
                alpha=0.9,
                edgecolors="black",
                linewidth=1.5,
                zorder=5,
            )
            fig.colorbar(im, ax=ax3, label="Density")
        else:
            # Fallback: use 2D histogram for low-variance data
            # Create histogram
            try:
                hist, xedges, yedges = np.histogram2d(
                    hard_vals,
                    soft_vals,
                    bins=[
                        min(20, len(np.unique(hard_vals))),
                        min(20, len(np.unique(soft_vals))),
                    ],
                )
                # Plot histogram as heatmap
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax3.imshow(
                    hist.T,
                    extent=extent,
                    origin="lower",
                    cmap="Blues",
                    alpha=0.6,
                    aspect="auto",
                )
                plt.colorbar(im, ax=ax3, label="Count")
            except:
                # Ultimate fallback: just scatter
                pass

            ax3.scatter(
                hard_vals,
                soft_vals,
                color=PALETTE[1],
                alpha=0.5,
                s=30,
                edgecolors="white",
                linewidth=0.5,
            )
            ax3.scatter(
                pareto_hard,
                pareto_soft,
                color=get_color("red"),
                s=70,
                alpha=0.9,
                edgecolors="black",
                linewidth=1.5,
                zorder=5,
            )
    except (ImportError, Exception) as e:
        # Fallback without scipy or on any other error
        ax3.scatter(
            hard_vals,
            soft_vals,
            color=PALETTE[1],
            alpha=0.5,
            s=30,
            edgecolors="white",
            linewidth=0.5,
        )
        ax3.scatter(
            pareto_hard,
            pareto_soft,
            color=get_color("red"),
            s=70,
            alpha=0.9,
            edgecolors="black",
            linewidth=1.5,
            zorder=5,
        )

    format_axis(
        ax3,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title="Population Density with Pareto Front",
        legend=False,
    )

    # Plot 4: Size-coded points showing frequency
    sizes = [unique_points.get((h, s), 1) * 25 for h, s in zip(hard_vals, soft_vals)]
    scatter = ax4.scatter(
        hard_vals,
        soft_vals,
        c=sizes,
        s=sizes,
        alpha=0.55,
        cmap="YlOrRd",
        edgecolors="black",
        linewidth=0.5,
    )
    ax4.scatter(
        pareto_hard,
        pareto_soft,
        color=get_color("red"),
        s=120,
        alpha=0.95,
        edgecolors="white",
        linewidth=2.5,
        label="Pareto Front",
        zorder=5,
    )

    fig.colorbar(scatter, ax=ax4, label="Point Frequency")
    format_axis(
        ax4,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title="Frequency-Coded Population",
        legend=True,
    )

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "pareto_front_comprehensive.pdf"))

    # Create the original single plot for backward compatibility
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

    # Show count for overlapping points
    for (h, s), count in unique_points.items():
        if count > 1:
            ax.annotate(
                f"{count}",
                (h, s),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    format_axis(
        ax,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title=f"Final Population Fitness Distribution\n({len(population)} individuals, "
        f"{len(unique_points)} unique solutions)",
        legend=True,
    )

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "pareto_front.pdf"))

    # Create a separate plot focusing only on the Pareto front
    if len(pareto_front) > 1:
        fig, ax = create_thesis_figure(1, 1, figsize=(8, 6))
        ax.scatter(
            pareto_hard,
            pareto_soft,
            color=get_color("red"),
            s=120,
            alpha=0.9,
            edgecolors="black",
            linewidth=1.5,
            zorder=5,
        )
        ax.plot(
            pareto_hard,
            pareto_soft,
            color=get_color("red"),
            linestyle="--",
            alpha=0.4,
            linewidth=1.5,
        )

        # Annotate each point with its index
        for i, (h, s) in enumerate(zip(pareto_hard, pareto_soft)):
            ax.annotate(
                f"{i+1}",
                (h, s),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        format_axis(
            ax,
            xlabel="Hard Constraint Violations",
            ylabel="Soft Constraint Penalty",
            title=f"Pareto Front Detail ({len(pareto_front)} non-dominated solutions)",
            legend=False,
        )

        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, "pareto_front_detail.pdf"))
