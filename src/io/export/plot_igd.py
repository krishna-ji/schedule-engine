"""
Inverted Generational Distance (IGD) trend plot.

Generates a publication-quality plot showing IGD evolution over
generations.  Lower IGD indicates the approximation set is closer
to the reference Pareto front.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.output_paths import get_nsga_plot_dir

from .thesis_style import (
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

apply_thesis_style()


def plot_igd_trend(igd_history: list[float], output_dir: str) -> None:
    """Plot IGD over generations (lower is better).

    NaN entries (no feasible solutions that generation) are shown as
    gaps in the line.

    Args:
        igd_history: IGD values per recorded generation (may contain NaN).
        output_dir: Directory to save the plot.
    """
    if not igd_history:
        return

    plot_dir = get_nsga_plot_dir(output_dir)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))

    gens = list(range(len(igd_history)))
    vals = np.array(igd_history, dtype=float)

    # Mask NaN for line continuity
    finite = np.isfinite(vals)
    if not finite.any():
        plt.close(fig)
        return

    ax.plot(
        np.array(gens)[finite],
        vals[finite],
        color=get_color("orange"),
        linewidth=2.5,
        marker="s",
        markersize=4,
        markevery=max(1, int(finite.sum()) // 20),
        label="IGD",
    )

    ax.fill_between(
        np.array(gens)[finite],
        vals[finite],
        alpha=0.15,
        color=get_color("orange"),
    )

    # Annotate best
    best_val = float(vals[finite].min())
    best_idx = int(np.nanargmin(vals))
    ax.annotate(
        f"Best: {best_val:.4f}",
        xy=(best_idx, best_val),
        xytext=(10, -15),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "alpha": 0.7},
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0"},
        fontsize=10,
    )

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="IGD (Inverted Generational Distance)",
        title="IGD Evolution\n(Lower = Better Proximity to Reference Front)",
        legend=True,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    output_path = str(Path(plot_dir) / "igd_over_generations.pdf")
    save_figure(fig, output_path)
    plt.close(fig)
