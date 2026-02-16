import matplotlib.pyplot as plt

from src.utils.output_paths import get_constraint_plot_dir

from .thesis_style import (
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

# Apply thesis styling
apply_thesis_style()


def plot_individual_hard_constraints(
    hard_trends: dict[str, list[int]], output_dir: str
) -> None:
    """
    Plot each hard constraint trend and store it under plots/constraints/.

    Args:
        hard_trends: Dictionary mapping constraint names to their trends over generations
        output_dir: Base output directory

    Note:
        CSV data available in csv/constraint_metrics.csv (hard_<constraint> columns)
    """
    constraint_dir = get_constraint_plot_dir(output_dir)

    # Individual plots for each hard constraint
    for constraint_name, trend in hard_trends.items():
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
            xlabel="Generation",
            ylabel="Violations",
            title=f"Hard Constraint Trend: {constraint_name.replace('_', ' ').title()}",
            legend=True,
        )

        plt.tight_layout()

        # Save individual plot
        filename = f"hard_constraint_{constraint_name}_violations_over_generations.pdf"
        save_figure(fig, constraint_dir / filename)


def plot_individual_soft_constraints(
    soft_trends: dict[str, list[int]], output_dir: str
) -> None:
    """
    Plot each soft constraint trend and store it under plots/constraints/.

    Args:
        soft_trends: Dictionary mapping constraint names to their trends over generations
        output_dir: Base output directory

    Note:
        CSV data available in csv/constraint_metrics.csv (soft_<constraint> columns)
    """
    constraint_dir = get_constraint_plot_dir(output_dir)

    # Individual plots for each soft constraint
    for constraint_name, trend in soft_trends.items():
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
            xlabel="Generation",
            ylabel="Penalty",
            title=f"Soft Constraint Trend: {constraint_name.replace('_', ' ').title()}",
            legend=True,
        )

        plt.tight_layout()

        # Save individual plot
        filename = f"soft_constraint_{constraint_name}_penalty_over_generations.pdf"
        save_figure(fig, constraint_dir / filename)


def plot_constraint_summary(
    hard_trends: dict[str, list[int]],
    soft_trends: dict[str, list[int]],
    output_dir: str,
) -> None:
    """
    Deprecated: constraint dashboard intentionally removed to avoid multi-plot files.

    Note:
        CSV data available in csv/constraint_metrics.csv (hard_total, soft_total columns)
    """
    return
