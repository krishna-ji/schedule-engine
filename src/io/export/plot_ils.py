"""
ILS (Iterated Local Search) diagnostic plots — thesis-ready.

Generates ILS-specific visualisation that is meaningful for single-solution
search (unlike NSGA-II Pareto/diversity/hypervolume plots which need a
population).

Plots generated:
  1. Hard violation convergence with improvement & restart markers
  2. Per-constraint breakdown stacked area over ILS iterations
  3. Repair operator efficacy (det-repair, gene-LS, RepairEngine) per iteration
  4. Cumulative improvement waterfall per repair stage
  5. Iteration wall-time profile
  6. Search dynamics — candidate vs best with acceptance markers
  7. Perturbation size over iterations
  8. Rescheduling event impact
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.utils.output_paths import _ensure_dir

from .thesis_style import (
    LINE_STYLES,
    MARKERS,
    PALETTE,
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

apply_thesis_style()


def _get_ils_plot_dir(output_dir: str | Path) -> Path:
    return _ensure_dir(Path(output_dir) / "plots" / "ils")


# ─────────────────────────────────────────────────────────────────────
# 1.  Hard constraint convergence (best-so-far) with event markers
# ─────────────────────────────────────────────────────────────────────


def plot_ils_convergence(
    iterations: list[int],
    best_hard: list[float],
    best_soft: list[float],
    improvement_iters: list[int],
    restart_iters: list[int],
    output_dir: str,
) -> None:
    """Best hard & soft over ILS iterations with improvement / restart markers."""
    plot_dir = _get_ils_plot_dir(output_dir)
    iters = np.array(iterations)

    # --- Hard convergence ---
    fig, ax = create_thesis_figure(figsize=(10, 5.5))
    ax.plot(iters, best_hard, color=get_color("red"), linewidth=2.5, label="Best Hard")

    if improvement_iters:
        imp_idx = [i for i, v in enumerate(iterations) if v in improvement_iters]
        ax.scatter(
            [iterations[i] for i in imp_idx],
            [best_hard[i] for i in imp_idx],
            marker="v",
            s=60,
            color=get_color("green"),
            zorder=5,
            label="Improvement",
        )
    if restart_iters:
        rst_idx = [i for i, v in enumerate(iterations) if v in restart_iters]
        for ri in rst_idx:
            ax.axvline(
                iterations[ri], color=get_color("orange"), ls="--", lw=1.2, alpha=0.7
            )
        # Single legend entry
        ax.axvline(
            -999, color=get_color("orange"), ls="--", lw=1.2, alpha=0.7, label="Restart"
        )

    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Hard Violations",
        title="Hard Constraint Convergence (ILS)",
        legend=True,
    )
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_hard_convergence.pdf")

    # --- Soft convergence ---
    fig, ax = create_thesis_figure(figsize=(10, 5.5))
    ax.plot(iters, best_soft, color=get_color("blue"), linewidth=2.5, label="Best Soft")
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Soft Penalty",
        title="Soft Penalty Convergence (ILS)",
        legend=True,
    )
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_soft_convergence.pdf")

    # --- Combined dual-axis ---
    fig, ax1 = create_thesis_figure(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    ln1 = ax1.plot(
        iters, best_hard, color=get_color("red"), linewidth=2.5, label="Hard"
    )
    ln2 = ax2.plot(
        iters, best_soft, color=get_color("blue"), linewidth=2.0, ls="--", label="Soft"
    )
    ax1.set_xlabel("ILS Iteration")
    ax1.set_ylabel("Hard Violations", color=get_color("red"))
    ax2.set_ylabel("Soft Penalty", color=get_color("blue"))
    ax1.set_title("Hard & Soft Convergence (ILS)", fontweight="bold", pad=15)
    lns = ln1 + ln2
    labs = [line.get_label() for line in lns]
    ax1.legend(lns, labs, loc="upper right")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_combined_convergence.pdf")


# ─────────────────────────────────────────────────────────────────────
# 2.  Per-constraint breakdown stacked area
# ─────────────────────────────────────────────────────────────────────


def plot_ils_constraint_breakdown(
    iterations: list[int],
    constraint_history: dict[str, list[float]],
    output_dir: str,
) -> None:
    """Stacked area of per-constraint hard violations over ILS iterations."""
    plot_dir = _get_ils_plot_dir(output_dir)
    if not constraint_history:
        return

    # Filter to constraints that ever have non-zero values
    active = {k: v for k, v in constraint_history.items() if any(x > 0 for x in v)}
    if not active:
        return

    # Sort by total descending
    sorted_names = sorted(active.keys(), key=lambda k: -sum(active[k]))
    iters = np.array(iterations)
    data = np.array([active[n] for n in sorted_names])

    fig, ax = create_thesis_figure(figsize=(11, 6))
    ax.stackplot(
        iters,
        data,
        labels=sorted_names,
        alpha=0.85,
        colors=PALETTE[: len(sorted_names)],
    )
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Hard Violations",
        title="Per-Constraint Breakdown Over ILS Iterations",
        legend=True,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_constraint_breakdown_stacked.pdf")

    # Also individual lines
    fig, ax = create_thesis_figure(figsize=(11, 6))
    for i, name in enumerate(sorted_names):
        ax.plot(
            iters,
            active[name],
            color=PALETTE[i % len(PALETTE)],
            linewidth=2.0,
            ls=LINE_STYLES[i % len(LINE_STYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=4,
            markevery=max(1, len(iters) // 15),
            label=name,
        )
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Hard Violations",
        title="Per-Constraint Trends Over ILS Iterations",
        legend=True,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_constraint_breakdown_lines.pdf")


# ─────────────────────────────────────────────────────────────────────
# 3.  Repair operator efficacy per iteration
# ─────────────────────────────────────────────────────────────────────


def plot_ils_repair_efficacy(
    iterations: list[int],
    det_fixes: list[int],
    ls_delta: list[int],
    engine_steps: list[int],
    output_dir: str,
) -> None:
    """Per-iteration repair contributions from each repair stage."""
    plot_dir = _get_ils_plot_dir(output_dir)
    iters = np.array(iterations)

    fig, ax = create_thesis_figure(figsize=(10, 5.5))
    ax.plot(
        iters,
        det_fixes,
        color=get_color("blue"),
        linewidth=2.0,
        label="Deterministic Repair (fixes)",
    )
    ax.plot(
        iters,
        ls_delta,
        color=get_color("green"),
        linewidth=2.0,
        label="Gene-LS (Δ score)",
    )
    ax.plot(
        iters,
        engine_steps,
        color=get_color("orange"),
        linewidth=2.0,
        label="RepairEngine (steps)",
    )
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Repair Contribution",
        title="Repair Operator Efficacy Over Iterations",
        legend=True,
    )
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_repair_efficacy.pdf")

    # Stacked bar version (grouped bars every N iterations)
    n = len(iterations)
    if n > 50:
        step = max(1, n // 30)
    else:
        step = 1
    sel_idx = list(range(0, n, step))
    sel_iters = [iterations[i] for i in sel_idx]
    sel_det = [det_fixes[i] for i in sel_idx]
    sel_ls = [ls_delta[i] for i in sel_idx]
    sel_eng = [engine_steps[i] for i in sel_idx]

    x = np.arange(len(sel_iters))
    width = 0.8
    fig, ax = create_thesis_figure(figsize=(12, 5.5))
    ax.bar(x, sel_det, width, label="Det Repair", color=get_color("blue"), alpha=0.85)
    ax.bar(
        x,
        sel_ls,
        width,
        bottom=sel_det,
        label="Gene-LS",
        color=get_color("green"),
        alpha=0.85,
    )
    ax.bar(
        x,
        sel_eng,
        width,
        bottom=[d + ls for d, ls in zip(sel_det, sel_ls, strict=False)],
        label="RepairEngine",
        color=get_color("orange"),
        alpha=0.85,
    )
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Repair Contribution",
        title="Stacked Repair Contribution Per Iteration",
        legend=True,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sel_iters], rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_repair_stacked_bar.pdf")


# ─────────────────────────────────────────────────────────────────────
# 4.  Cumulative improvement waterfall
# ─────────────────────────────────────────────────────────────────────


def plot_ils_improvement_waterfall(
    phase1_hard: float,
    final_hard: float,
    improvement_events: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Waterfall chart showing where each improvement came from.

    improvement_events: list of {"iter": int, "delta": float, "source": str}
    source ∈ {"perturb+repair", "rescheduling", "restart"}
    """
    plot_dir = _get_ils_plot_dir(output_dir)
    if not improvement_events:
        return

    # Aggregate by source
    source_delta: dict[str, float] = {}
    for ev in improvement_events:
        src = ev.get("source", "unknown")
        source_delta[src] = source_delta.get(src, 0.0) + ev["delta"]

    labels = ["Phase 1\n(init)", *list(source_delta.keys()), "Final"]
    values = [phase1_hard]
    running = phase1_hard
    for delta in source_delta.values():
        values.append(-delta)
        running -= delta
    values.append(running)

    # Waterfall logic
    cumulative = [phase1_hard]
    for i in range(1, len(values) - 1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(final_hard)

    bottoms = [0.0] * len(values)
    heights = list(values)
    colors = [get_color("blue")]  # start
    for i in range(1, len(values) - 1):
        if values[i] < 0:
            bottoms[i] = cumulative[i]
            heights[i] = -values[i]
            colors.append(get_color("green"))
        else:
            bottoms[i] = cumulative[i - 1]
            colors.append(get_color("red"))
    colors.append(get_color("purple"))  # final bar
    bottoms[0] = 0
    heights[0] = phase1_hard
    bottoms[-1] = 0
    heights[-1] = final_hard

    fig, ax = create_thesis_figure(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        heights,
        bottom=bottoms,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        width=0.65,
    )

    # Value labels
    for i, bar in enumerate(bars):
        y = bar.get_y() + bar.get_height()
        if i == 0 or i == len(bars) - 1:
            txt = f"{heights[i]:.0f}"
        else:
            txt = f"-{heights[i]:.0f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + 1,
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Connector lines
    for i in range(len(x) - 1):
        y_conn = cumulative[i]
        ax.plot(
            [x[i] + 0.325, x[i + 1] - 0.325],
            [y_conn, y_conn],
            color="gray",
            ls="--",
            lw=0.8,
        )

    format_axis(
        ax,
        xlabel="",
        ylabel="Hard Violations",
        title="ILS Improvement Waterfall",
        legend=False,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_improvement_waterfall.pdf")


# ─────────────────────────────────────────────────────────────────────
# 5.  Iteration wall-time profile
# ─────────────────────────────────────────────────────────────────────


def plot_ils_timing(
    iterations: list[int],
    iter_times: list[float],
    output_dir: str,
) -> None:
    """Wall-time per ILS iteration."""
    plot_dir = _get_ils_plot_dir(output_dir)
    iters = np.array(iterations)
    times = np.array(iter_times)

    fig, ax = create_thesis_figure(figsize=(10, 5))
    ax.plot(iters, times, color=get_color("purple"), linewidth=1.5, alpha=0.6)
    # Rolling average
    if len(times) > 10:
        window = min(20, len(times) // 5)
        kernel = np.ones(window) / window
        rolling = np.convolve(times, kernel, mode="valid")
        ax.plot(
            iters[window - 1 :],
            rolling,
            color=get_color("purple"),
            linewidth=2.5,
            label=f"Rolling avg ({window})",
        )
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Wall Time (s)",
        title="Wall Time Per ILS Iteration",
        legend=True,
    )
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_iteration_time.pdf")

    # Cumulative time
    fig, ax = create_thesis_figure(figsize=(10, 5))
    cumulative = np.cumsum(times)
    ax.plot(iters, cumulative / 60.0, color=get_color("brown"), linewidth=2.5)
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Cumulative Time (min)",
        title="Cumulative Wall Time",
        legend=False,
    )
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_cumulative_time.pdf")


# ─────────────────────────────────────────────────────────────────────
# 6.  Search dynamics — candidate vs best
# ─────────────────────────────────────────────────────────────────────


def plot_ils_search_dynamics(
    iterations: list[int],
    candidate_hard: list[float],
    best_hard: list[float],
    output_dir: str,
) -> None:
    """Candidate hard vs best hard, showing search dynamics."""
    plot_dir = _get_ils_plot_dir(output_dir)
    iters = np.array(iterations)

    fig, ax = create_thesis_figure(figsize=(11, 5.5))
    ax.scatter(
        iters,
        candidate_hard,
        s=12,
        alpha=0.35,
        color=get_color("gray"),
        label="Candidate",
        zorder=2,
    )
    ax.plot(
        iters,
        best_hard,
        color=get_color("red"),
        linewidth=2.5,
        label="Best-so-far",
        zorder=3,
    )
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Hard Violations",
        title="ILS Search Dynamics: Candidate vs Best",
        legend=True,
    )
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_search_dynamics.pdf")


# ─────────────────────────────────────────────────────────────────────
# 7.  Perturbation size over iterations
# ─────────────────────────────────────────────────────────────────────


def plot_ils_perturbation_size(
    iterations: list[int],
    perturb_sizes: list[int],
    output_dir: str,
) -> None:
    """Number of genes perturbed per ILS iteration."""
    plot_dir = _get_ils_plot_dir(output_dir)
    iters = np.array(iterations)

    fig, ax = create_thesis_figure(figsize=(10, 5))
    ax.plot(iters, perturb_sizes, color=get_color("cyan"), linewidth=2.0)
    format_axis(
        ax,
        xlabel="ILS Iteration",
        ylabel="Genes Perturbed",
        title="Perturbation Size Over Iterations",
        legend=False,
    )
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_perturbation_size.pdf")


# ─────────────────────────────────────────────────────────────────────
# 8.  Rescheduling impact
# ─────────────────────────────────────────────────────────────────────


def plot_ils_rescheduling_impact(
    reschedule_events: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Bar chart of rescheduling events: before/after hard violations.

    reschedule_events: [{"iter": int, "type": str, "before": float, "after": float}]
    """
    plot_dir = _get_ils_plot_dir(output_dir)
    if not reschedule_events:
        return

    labels = [f"Iter {e['iter']}\n({e['type']})" for e in reschedule_events]
    before = [e["before"] for e in reschedule_events]
    after = [e["after"] for e in reschedule_events]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = create_thesis_figure(figsize=(max(8, len(labels) * 1.2), 5.5))
    ax.bar(
        x - width / 2, before, width, label="Before", color=get_color("red"), alpha=0.85
    )
    ax.bar(
        x + width / 2, after, width, label="After", color=get_color("green"), alpha=0.85
    )

    # Delta labels
    for i in range(len(labels)):
        delta = after[i] - before[i]
        y = max(before[i], after[i]) + 1
        colour = get_color("green") if delta < 0 else get_color("red")
        ax.text(
            x[i],
            y,
            f"{delta:+.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=colour,
        )

    format_axis(
        ax,
        xlabel="",
        ylabel="Hard Violations",
        title="Rescheduling Pass Impact",
        legend=True,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_rescheduling_impact.pdf")


# ─────────────────────────────────────────────────────────────────────
# 9.  ILS summary dashboard (multi-panel)
# ─────────────────────────────────────────────────────────────────────


def plot_ils_dashboard(
    iterations: list[int],
    best_hard: list[float],
    best_soft: list[float],
    candidate_hard: list[float],
    det_fixes: list[int],
    ls_delta: list[int],
    engine_steps: list[int],
    iter_times: list[float],
    constraint_history: dict[str, list[float]],
    improvement_iters: list[int],
    restart_iters: list[int],
    output_dir: str,
) -> None:
    """6-panel ILS diagnostic dashboard."""
    plot_dir = _get_ils_plot_dir(output_dir)
    iters = np.array(iterations)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Hard convergence
    ax = axes[0, 0]
    ax.plot(iters, best_hard, color=get_color("red"), linewidth=2.0)
    if improvement_iters:
        imp_idx = [i for i, v in enumerate(iterations) if v in improvement_iters]
        ax.scatter(
            [iterations[i] for i in imp_idx],
            [best_hard[i] for i in imp_idx],
            marker="v",
            s=30,
            color=get_color("green"),
            zorder=5,
        )
    for ri in restart_iters:
        if ri in iterations:
            ax.axvline(ri, color=get_color("orange"), ls="--", lw=0.8, alpha=0.5)
    format_axis(
        ax, xlabel="Iteration", ylabel="Hard", title="Hard Convergence", legend=False
    )
    ax.set_xlim(left=0)

    # Panel 2: Search dynamics
    ax = axes[0, 1]
    ax.scatter(iters, candidate_hard, s=8, alpha=0.25, color=get_color("gray"))
    ax.plot(iters, best_hard, color=get_color("red"), linewidth=2.0)
    format_axis(
        ax, xlabel="Iteration", ylabel="Hard", title="Candidate vs Best", legend=False
    )
    ax.set_xlim(left=0)

    # Panel 3: Repair efficacy
    ax = axes[0, 2]
    ax.plot(
        iters, det_fixes, color=get_color("blue"), linewidth=1.5, alpha=0.6, label="Det"
    )
    ax.plot(
        iters, ls_delta, color=get_color("green"), linewidth=1.5, alpha=0.6, label="LS"
    )
    ax.plot(
        iters,
        engine_steps,
        color=get_color("orange"),
        linewidth=1.5,
        alpha=0.6,
        label="Engine",
    )
    format_axis(
        ax,
        xlabel="Iteration",
        ylabel="Contribution",
        title="Repair Efficacy",
        legend=True,
    )
    ax.set_xlim(left=0)

    # Panel 4: Constraint breakdown stacked
    ax = axes[1, 0]
    if constraint_history:
        active = {k: v for k, v in constraint_history.items() if any(x > 0 for x in v)}
        if active:
            sorted_names = sorted(active.keys(), key=lambda k: -sum(active[k]))
            data = np.array([active[n] for n in sorted_names])
            ax.stackplot(
                iters,
                data,
                labels=sorted_names,
                alpha=0.8,
                colors=PALETTE[: len(sorted_names)],
            )
            ax.legend(loc="upper right", fontsize=6, ncol=2)
    format_axis(
        ax,
        xlabel="Iteration",
        ylabel="Hard",
        title="Constraint Breakdown",
        legend=False,
    )
    ax.set_xlim(left=0)

    # Panel 5: Wall time
    ax = axes[1, 1]
    times = np.array(iter_times)
    ax.plot(iters, times, color=get_color("purple"), linewidth=1.0, alpha=0.5)
    if len(times) > 10:
        w = min(20, len(times) // 5)
        kernel = np.ones(w) / w
        rolling = np.convolve(times, kernel, mode="valid")
        ax.plot(iters[w - 1 :], rolling, color=get_color("purple"), linewidth=2.0)
    format_axis(
        ax,
        xlabel="Iteration",
        ylabel="Time (s)",
        title="Wall Time / Iteration",
        legend=False,
    )
    ax.set_xlim(left=0)

    # Panel 6: Combined hard+soft
    ax = axes[1, 2]
    ax.plot(iters, best_hard, color=get_color("red"), linewidth=2.0, label="Hard")
    ax2 = ax.twinx()
    ax2.plot(
        iters, best_soft, color=get_color("blue"), linewidth=1.5, ls="--", label="Soft"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hard", color=get_color("red"))
    ax2.set_ylabel("Soft", color=get_color("blue"))
    ax.set_title("Hard & Soft", fontweight="bold", pad=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.suptitle("ILS Diagnostic Dashboard", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, plot_dir / "ils_dashboard.pdf")
