import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.output_paths import get_csv_dir, get_nsga_plot_dir

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


# ── pymoo-compatible version (works with F matrix directly) ──────────


def plot_pareto_front_from_F(F: np.ndarray, output_dir: str) -> None:
    """Pareto-front plot from a pymoo objective matrix *F* (shape N×2).

    Columns: ``F[:, 0]`` = hard-constraint violations,
             ``F[:, 1]`` = soft-constraint penalty.

    Produces the same plot as ``plot_pareto_front`` using the pymoo
    objective matrix directly.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    hard_vals = F[:, 0].tolist()
    soft_vals = F[:, 1].tolist()

    csv_dir = get_csv_dir(output_dir)

    csv_path = Path(csv_dir) / "population_fitness.csv"
    with csv_path.open("w", newline="") as f:
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

    # Non-dominated sorting to find Pareto front
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    pf_idxs = fronts[0]
    pareto_hard = F[pf_idxs, 0].tolist()
    pareto_soft = F[pf_idxs, 1].tolist()

    csv_path = Path(csv_dir) / "pareto_front.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Pareto_Index", "Hard_Constraint_Violations", "Soft_Constraint_Penalties"]
        )
        for idx, (h, s) in enumerate(zip(pareto_hard, pareto_soft, strict=False)):
            writer.writerow([idx, h, s])

    plot_dir = get_nsga_plot_dir(output_dir)

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
        label=f"Pareto Front ({len(pf_idxs)} solutions)",
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )

    # Knee point
    knee_point: tuple[float, float] | None = None
    if len(pf_idxs) >= 3:
        sorted_pts = sorted(zip(pareto_hard, pareto_soft, strict=False))
        h_arr = np.array([p[0] for p in sorted_pts], dtype=float)
        s_arr = np.array([p[1] for p in sorted_pts], dtype=float)
        h_range = float(np.ptp(h_arr)) or 1.0
        s_range = float(np.ptp(s_arr)) or 1.0
        hn = (h_arr - h_arr.min()) / h_range
        sn = (s_arr - s_arr.min()) / s_range
        x1, y1, x2, y2 = hn[0], sn[0], hn[-1], sn[-1]
        denom = math.hypot(y2 - y1, x2 - x1)
        if denom > 0:
            dist = np.abs((y2 - y1) * hn - (x2 - x1) * sn + x2 * y1 - y2 * x1) / denom
            ki = int(np.argmax(dist))
            knee_point = (h_arr[ki], s_arr[ki])

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

    feasible = [
        (h, s) for h, s in zip(pareto_hard, pareto_soft, strict=False) if h == 0
    ]
    if feasible:
        best_f = min(feasible, key=lambda p: p[1])
        ax.scatter(
            [best_f[0]],
            [best_f[1]],
            color=get_color("green"),
            s=120,
            marker="D",
            edgecolors="black",
            linewidth=1.0,
            label="Best Feasible Tradeoff",
            zorder=6,
        )

    n_unique = len(set(zip(hard_vals, soft_vals, strict=False)))
    format_axis(
        ax,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title=(
            f"Final Population Fitness Distribution\n"
            f"({len(hard_vals)} individuals, {n_unique} unique solutions)"
        ),
        legend=True,
    )
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "pareto_front_population_and_nondominated.pdf")
