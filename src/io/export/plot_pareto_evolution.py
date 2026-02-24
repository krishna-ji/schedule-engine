"""
Pareto Evolution Scatter Plot

Shows the population flowing towards the origin across generations.
Points are coloured by generation number using the viridis colormap,
revealing how the Pareto front converges over time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.output_paths import get_nsga_plot_dir

from .thesis_style import apply_thesis_style, create_thesis_figure, save_figure

# Apply thesis styling
apply_thesis_style()


def plot_pareto_evolution(
    f_history: list[np.ndarray],
    output_dir: str,
) -> None:
    """Create a generation-coloured scatter of all F arrays.

    Parameters
    ----------
    f_history : list[ndarray]
        One ``(pop_size, 2)`` array per generation.
        ``F[:, 0]`` = hard constraint violations,
        ``F[:, 1]`` = soft constraint penalties.
    output_dir : str
        Base output directory.

    Saves
    -----
    ``plots/nsga/pareto_evolution.pdf``
    """
    if not f_history:
        return

    plot_dir = get_nsga_plot_dir(output_dir)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Concatenate all generations into one array, plus a generation label
    all_hard: list[float] = []
    all_soft: list[float] = []
    all_gen: list[int] = []

    for gen_idx, F in enumerate(f_history):
        n = F.shape[0]
        all_hard.extend(F[:, 0].tolist())
        all_soft.extend(F[:, 1].tolist())
        all_gen.extend([gen_idx + 1] * n)

    hard_arr = np.asarray(all_hard)
    soft_arr = np.asarray(all_soft)
    gen_arr = np.asarray(all_gen)

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 7))

    scatter = ax.scatter(
        hard_arr,
        soft_arr,
        c=gen_arr,
        cmap="viridis",
        s=12,
        alpha=0.45,
        edgecolors="none",
        rasterized=True,  # keeps PDF small
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Generation", fontsize=12)

    # Highlight the final generation's Pareto front
    final_F = f_history[-1]
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    nds = NonDominatedSorting()
    fronts = nds.do(final_F)
    pf = final_F[fronts[0]]
    pf_sorted = pf[pf[:, 0].argsort()]
    ax.plot(
        pf_sorted[:, 0],
        pf_sorted[:, 1],
        color="red",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label=f"Final Pareto Front (gen {len(f_history)})",
        zorder=10,
    )

    ax.set_xlabel("Hard Constraint Violations", fontsize=12)
    ax.set_ylabel("Soft Constraint Penalty", fontsize=12)
    ax.set_title(
        "Pareto Front Evolution Across Generations",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_figure(fig, plot_dir / "pareto_evolution.pdf")
