import matplotlib.pyplot as plt
import seaborn as sns

from schedule_engine.utils.output_paths import get_nsga_plot_dir

from .thesis_style import (
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

# Apply thesis styling
apply_thesis_style()


def plot_fitness_histograms_and_feasibility(
    population: list, feasibility_rate: list[float], output_dir: str
) -> None:
    """
    Plot final population fitness histograms (hard + soft) alongside feasibility rate.
    """
    if not population or not feasibility_rate:
        return

    hard_vals = [ind.fitness.values[0] for ind in population]
    soft_vals = [ind.fitness.values[1] for ind in population]

    plot_dir = get_nsga_plot_dir(output_dir)

    fig, (ax1, ax2) = create_thesis_figure(1, 2, figsize=(14, 6))

    sns.histplot(
        hard_vals,
        bins=min(30, max(5, len(set(hard_vals)))),
        stat="density",
        color=get_color("red"),
        alpha=0.55,
        edgecolor="black",
        linewidth=0.5,
        label="Hard Violations",
        ax=ax1,
    )
    sns.histplot(
        soft_vals,
        bins=min(30, max(5, len(set(soft_vals)))),
        stat="density",
        color=get_color("green"),
        alpha=0.45,
        edgecolor="black",
        linewidth=0.5,
        label="Soft Penalties",
        ax=ax1,
    )

    format_axis(
        ax1,
        xlabel="Fitness Value",
        ylabel="Density",
        title="Final Population Fitness Distributions",
        legend=True,
    )

    generations = list(range(len(feasibility_rate)))
    ax2.plot(
        generations,
        feasibility_rate,
        color=get_color("blue"),
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=max(1, len(generations) // 20),
    )
    format_axis(
        ax2,
        xlabel="Generation",
        ylabel="Feasibility Rate (%)",
        title="Feasibility Rate Over Generations",
        legend=False,
    )
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    save_figure(
        fig,
        plot_dir / "final_population_fitness_histograms_and_feasibility_rate.pdf",
    )
