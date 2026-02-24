"""
Memetic / Repair-Specific Academic Plots

Generates publication-quality plots that prove the impact of the
BitsetSchedulingRepair local search operator for an academic thesis:

1. **Repair Intervention Plot** — convergence line with highlighted
   repair generations and annotated deltas at the largest drops.
2. **Pareto Repair Shift Plot** — pre/post repair population scatter
   showing how the local search moves the Pareto front.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .thesis_style import (
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

# Apply thesis styling
apply_thesis_style()


# =====================================================================
#  1. Repair Intervention Plot
# =====================================================================


def plot_repair_interventions(
    best_hards: list[float],
    repair_gens: list[int],
    output_dir: str | Path,
    *,
    top_k_annotations: int = 5,
) -> None:
    """Convergence line with repair-intervention highlights and delta annotations.

    Parameters
    ----------
    best_hards : list[float]
        Best hard-constraint violation per generation (0-indexed).
    repair_gens : list[int]
        **1-based** generation numbers where repair actually fired.
    output_dir : str | Path
        Base output dir — plot saved under ``plots/memetic_analysis/``.
    top_k_annotations : int
        How many of the largest post-repair drops to annotate.
    """
    if not best_hards or not repair_gens:
        return

    out = Path(output_dir) / "plots" / "memetic_analysis"
    out.mkdir(parents=True, exist_ok=True)

    n_gens = len(best_hards)
    generations = np.arange(n_gens)

    fig, ax = create_thesis_figure(1, 1, figsize=(12, 6))

    # ── Main convergence line ────────────────────────────────────
    ax.plot(
        generations,
        best_hards,
        color=get_color("blue"),
        linewidth=2.0,
        label="Best Hard Violations",
        zorder=3,
    )

    # ── Background spans at repair generations ───────────────────
    # Convert 1-based repair_gens to 0-based indices for the array
    for rg in repair_gens:
        idx = rg - 1  # 0-based
        if 0 <= idx < n_gens:
            ax.axvspan(
                idx - 0.4,
                idx + 0.4,
                alpha=0.12,
                color=get_color("green"),
                zorder=1,
            )

    # Sentinel vertical lines (lighter, behind the data)
    for rg in repair_gens:
        idx = rg - 1
        if 0 <= idx < n_gens:
            ax.axvline(
                idx,
                color=get_color("green"),
                linestyle="--",
                linewidth=0.7,
                alpha=0.45,
                zorder=2,
            )

    # ── Compute deltas at repair generations ─────────────────────
    deltas: list[tuple[int, float]] = []  # (0-based gen, delta)
    for rg in repair_gens:
        idx = rg - 1  # 0-based
        if 1 <= idx < n_gens:
            delta = best_hards[idx] - best_hards[idx - 1]
            if delta < 0:  # improvement
                deltas.append((idx, delta))

    # Sort by magnitude (largest negative = biggest improvement)
    deltas.sort(key=lambda t: t[1])

    # ── Annotate top-K largest improvements ──────────────────────
    for rank, (idx, delta) in enumerate(deltas[:top_k_annotations]):
        y_val = best_hards[idx]
        ax.annotate(
            f"$\\Delta$ {int(delta)}",
            xy=(idx, y_val),
            xytext=(12, -18 - 14 * rank),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=get_color("red"),
            arrowprops={
                "arrowstyle": "->",
                "color": get_color("red"),
                "lw": 1.2,
            },
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": get_color("red"),
                "alpha": 0.9,
            },
            zorder=5,
        )

    # ── Summary statistics in text box ───────────────────────────
    total_improvement = sum(d for _, d in deltas)
    n_effective = len(deltas)
    textstr = (
        f"Repair activations: {len(repair_gens)}\n"
        f"Effective (Δ<0): {n_effective}\n"
        f"Total Δ: {int(total_improvement)}"
    )
    ax.text(
        0.98,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#CCCCCC",
        },
        zorder=6,
    )

    # Add a dummy artist for the legend
    from matplotlib.patches import Patch

    repair_patch = Patch(
        facecolor=get_color("green"),
        alpha=0.25,
        label=f"Repair Intervention (n={len(repair_gens)})",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(repair_patch)
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)

    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Hard Constraint Violations",
        title="Convergence with Repair Interventions",
        legend=False,
        y_from_zero=True,
    )

    plt.tight_layout()
    save_figure(fig, out / "repair_interventions.pdf")


# =====================================================================
#  2. Pareto Repair Shift Plot
# =====================================================================


def plot_pareto_repair_shift(
    f_history: list[np.ndarray],
    repair_gens: list[int],
    output_dir: str | Path,
) -> None:
    """Scatter showing population shift immediately before/after each repair.

    For every repair generation *g*, the population at *g-1* (pre-repair)
    is plotted in red crosses and the population at *g* (post-repair) in
    green circles.  When multiple repair generations exist, a single
    combined plot is generated plus one focused plot for the **first**
    repair event (cleanest visual).

    Parameters
    ----------
    f_history : list[ndarray]
        Per-generation F matrices (``(pop_size, 2)``), 0-indexed.
    repair_gens : list[int]
        **1-based** generation numbers where repair fired.
    output_dir : str | Path
    """
    if not f_history or not repair_gens:
        return

    out = Path(output_dir) / "plots" / "memetic_analysis"
    out.mkdir(parents=True, exist_ok=True)

    n = len(f_history)

    # ── Focused plot: first repair event ─────────────────────────
    _plot_single_shift(f_history, repair_gens[0], n, out, tag="first")

    # ── Overlay plot: ALL repair events combined ─────────────────
    if len(repair_gens) > 1:
        _plot_combined_shift(f_history, repair_gens, n, out)


def _plot_single_shift(
    f_history: list[np.ndarray],
    repair_gen: int,
    n: int,
    out: Path,
    tag: str,
) -> None:
    """Single pre/post scatter for one repair generation."""
    pre_idx = repair_gen - 2  # 0-based index for gen *before* repair
    post_idx = repair_gen - 1  # 0-based index for gen *of* repair
    if pre_idx < 0 or post_idx >= n:
        return

    F_pre = f_history[pre_idx]
    F_post = f_history[post_idx]

    fig, ax = create_thesis_figure(1, 1, figsize=(9, 7))

    ax.scatter(
        F_pre[:, 0],
        F_pre[:, 1],
        c=get_color("red"),
        marker="x",
        s=50,
        alpha=0.65,
        linewidths=1.5,
        label=f"Pre-Repair (Gen {repair_gen - 1})",
        zorder=3,
    )
    ax.scatter(
        F_post[:, 0],
        F_post[:, 1],
        c=get_color("green"),
        marker="o",
        s=40,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.4,
        label=f"Post-Repair (Gen {repair_gen})",
        zorder=4,
    )

    # Draw arrows from pre centroid to post centroid
    pre_c = F_pre.mean(axis=0)
    post_c = F_post.mean(axis=0)
    ax.annotate(
        "",
        xy=post_c,
        xytext=pre_c,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 2.0,
            "mutation_scale": 15,
        },
        zorder=5,
    )
    ax.text(
        (pre_c[0] + post_c[0]) / 2,
        (pre_c[1] + post_c[1]) / 2 + 15,
        f"ΔHard={post_c[0] - pre_c[0]:+.0f}\nΔSoft={post_c[1] - pre_c[1]:+.0f}",
        fontsize=9,
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "lightyellow",
            "edgecolor": "gray",
            "alpha": 0.9,
        },
        zorder=6,
    )

    format_axis(
        ax,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title=f"Pareto Shift — Repair at Generation {repair_gen}",
        legend=True,
        y_from_zero=True,
    )
    ax.set_xlim(left=0)

    plt.tight_layout()
    save_figure(fig, out / f"pareto_repair_shift_{tag}.pdf")


def _plot_combined_shift(
    f_history: list[np.ndarray],
    repair_gens: list[int],
    n: int,
    out: Path,
) -> None:
    """Overlay all pre/post pairs with transparency."""
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 7))

    pre_drawn = False
    post_drawn = False

    for rg in repair_gens:
        pre_idx = rg - 2
        post_idx = rg - 1
        if pre_idx < 0 or post_idx >= n:
            continue

        F_pre = f_history[pre_idx]
        F_post = f_history[post_idx]

        kw_pre: dict[str, Any] = {
            "c": get_color("red"),
            "marker": "x",
            "s": 25,
            "alpha": 0.30,
            "linewidths": 1.0,
            "zorder": 3,
        }
        kw_post: dict[str, Any] = {
            "c": get_color("green"),
            "marker": "o",
            "s": 20,
            "alpha": 0.30,
            "edgecolors": "black",
            "linewidths": 0.3,
            "zorder": 4,
        }

        # Only label the first pair for a clean legend
        if not pre_drawn:
            kw_pre["label"] = "Pre-Repair"
            pre_drawn = True
        if not post_drawn:
            kw_post["label"] = "Post-Repair"
            post_drawn = True

        ax.scatter(F_pre[:, 0], F_pre[:, 1], **kw_pre)
        ax.scatter(F_post[:, 0], F_post[:, 1], **kw_post)

    format_axis(
        ax,
        xlabel="Hard Constraint Violations",
        ylabel="Soft Constraint Penalty",
        title=f"Combined Pareto Shift — {len(repair_gens)} Repair Events",
        legend=True,
        y_from_zero=True,
    )
    ax.set_xlim(left=0)

    plt.tight_layout()
    save_figure(fig, out / "pareto_repair_shift_combined.pdf")
