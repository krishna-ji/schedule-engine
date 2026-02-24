import matplotlib.pyplot as plt
import numpy as np

from src.utils.output_paths import get_constraint_plot_dir

from .thesis_style import (
    PALETTE,
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    save_figure,
)

# Apply thesis styling
apply_thesis_style()


def plot_individual_hard_constraints(
    hard_trends: dict[str, list[int]], output_dir: str
) -> None:
    """
    Plot a **stacked area chart** of all hard constraints over generations.

    Each constraint type is a coloured band so the reader can see the
    volume reduction of each constraint over time.

    Args:
        hard_trends: ``{constraint_name: [violation_per_gen, ...]}``
        output_dir: Base output directory
    """
    if not hard_trends:
        return

    constraint_dir = get_constraint_plot_dir(output_dir)
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    # Build arrays — keys ordered by total violations (largest band at bottom)
    sorted_keys = sorted(
        hard_trends.keys(), key=lambda k: sum(hard_trends[k]), reverse=True
    )
    labels = [k.replace("_", " ").title() for k in sorted_keys]
    data = np.array([hard_trends[k] for k in sorted_keys], dtype=float)
    generations = np.arange(data.shape[1])

    # Stacked area chart with colorblind-safe palette
    ax.stackplot(
        generations,
        data,
        labels=labels,
        colors=PALETTE[: len(sorted_keys)],
        alpha=0.85,
    )

    # Add a thin total line on top for clarity
    total = data.sum(axis=0)
    ax.plot(
        generations,
        total,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=f"Total (final={int(total[-1])})",
    )

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Violations",
        title="Hard Constraint Violations (Academic Nomenclature) \u2014 Stacked Area",
        legend=True,
        y_from_zero=True,
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, constraint_dir / "hard_constraints_stacked_area.pdf")


def plot_individual_soft_constraints(
    soft_trends: dict[str, list[int]], output_dir: str
) -> None:
    """
    Plot a **stacked area chart** of all soft constraint penalties over generations.

    Args:
        soft_trends: ``{constraint_name: [penalty_per_gen, ...]}``
        output_dir: Base output directory
    """
    if not soft_trends:
        return

    constraint_dir = get_constraint_plot_dir(output_dir)
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    sorted_keys = sorted(
        soft_trends.keys(), key=lambda k: sum(soft_trends[k]), reverse=True
    )
    labels = [k.replace("_", " ").title() for k in sorted_keys]
    data = np.array([soft_trends[k] for k in sorted_keys], dtype=float)
    generations = np.arange(data.shape[1])

    ax.stackplot(
        generations,
        data,
        labels=labels,
        colors=PALETTE[: len(sorted_keys)],
        alpha=0.85,
    )

    total = data.sum(axis=0)
    ax.plot(
        generations,
        total,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=f"Total (final={int(total[-1])})",
    )

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Penalty",
        title="Soft Constraint Penalties (Academic Nomenclature) \u2014 Stacked Area",
        legend=True,
        y_from_zero=True,
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, constraint_dir / "soft_constraints_stacked_area.pdf")


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
