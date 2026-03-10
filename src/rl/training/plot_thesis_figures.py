r"""Generate publication-ready thesis figures from RL training CSVs.

Produces up to 12 PDF figures suitable for direct \LaTeX\ inclusion:

1.  **Learning Curve** — episode reward with rolling-window smoothing.
2.  **AOS Policy Map** — action selection scatter over generations.
3.  **Convergence Trajectory** — hard / soft penalty descent (dual axis).
4.  **Baseline Comparison** — PPO vs 6 static baselines (hard + soft).
5.  **Policy & Value Loss** — SB3 training loss curves.
6.  **Entropy Curve** — policy entropy over training.
7.  **Action Frequency** — stacked-area of action proportions per episode.
8.  **Step Reward & Hard Descent** — step-level reward + best_hard.
9.  **Feasibility Rate** — feasible fraction over training steps.
10. **Delta-Hard Distribution** — per-action box-plot of repair effectiveness.
11. **Mask Usage Trend** — MaskablePPO mask blocking percentage.
12. **Cross-Agent Comparison** — bar chart of final metrics across agents.

All figures use Times New Roman (serif), 300 DPI, and the Okabe-Ito
colourblind-safe palette.

Usage
-----

.. code-block:: python

    from src.rl.training.plot_thesis_figures import generate_plots
    pdfs = generate_plots(
        Path("output/rl_phase54/20260307_120000"),
        baselines_csv=Path("output/rl_phase54/static_baselines.csv"),
    )
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Okabe-Ito colourblind-safe palette
_CB = [
    "#E69F00",  # 0 orange
    "#56B4E9",  # 1 sky blue
    "#009E73",  # 2 bluish green
    "#F0E442",  # 3 yellow
    "#0072B2",  # 4 blue
    "#D55E00",  # 5 vermillion
    "#CC79A7",  # 6 reddish purple
    "#000000",  # 7 black
]

# Canonical action labels (Phase 53 pipeline LLH space)
_ACTION_LABELS: dict[int, str] = {
    0: "Conservative (3-pass)",
    1: "Aggressive (7-pass)",
    2: "Memetic Elite",
    3: "Soft-Focus",
    4: "Destruct.-Construct.",
    5: "Intensified (5-pass)",
}


# ------------------------------------------------------------------
# Style setup
# ------------------------------------------------------------------


def _setup_style():
    """Configure matplotlib for Times New Roman academic styling."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "text.usetex": False,
        }
    )
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts (string values)."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple centred rolling average (pads edges with original values)."""
    if window < 2 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


# ------------------------------------------------------------------
# Figure generators
# ------------------------------------------------------------------


def _fig_01_learning_curve(
    run_dir: Path,
    plt,
    window: int = 10,
) -> Path | None:
    """Fig 1: Episode reward with rolling-window smoothing."""
    csv_path = run_dir / "training_curve.csv"
    if not csv_path.exists():
        logger.warning("training_curve.csv not found — skipping Fig 1")
        return None

    rows = _read_csv(csv_path)
    episodes = np.array([int(r["episode"]) for r in rows])
    rewards = np.array([float(r["episode_reward"]) for r in rows])
    smoothed = _rolling_mean(rewards, window)

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Raw per-episode bars
    ax.bar(
        episodes,
        rewards,
        alpha=0.25,
        color=_CB[0],
        width=0.8,
        label="Episode Reward",
    )

    # Smoothed overlay
    ax.plot(
        episodes,
        smoothed,
        color=_CB[4],
        linewidth=2,
        label=f"Rolling Mean (w={window})",
    )

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("PPO Training: Learning Curve")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_01_learning_curve.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_02_aos_policy(run_dir: Path, plt) -> Path | None:
    """Fig 2: Adaptive Operator Selection policy map."""
    csv_path = run_dir / "evaluation_trajectory.csv"
    if not csv_path.exists():
        csv_path = run_dir / "evaluation_trajectory_200.csv"
    if not csv_path.exists():
        logger.warning("evaluation_trajectory*.csv not found — skipping Fig 2")
        return None

    rows = _read_csv(csv_path)
    # Filter out init row (action_id == -1)
    rows = [r for r in rows if int(r["action_id"]) >= 0]
    gens = np.array([int(r["generation"]) for r in rows])
    actions = np.array([int(r["action_id"]) for r in rows])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # One colour per unique action
    for a_id in sorted(set(actions)):
        mask = actions == a_id
        ax.scatter(
            gens[mask],
            actions[mask],
            c=_CB[a_id % len(_CB)],
            label=_ACTION_LABELS.get(a_id, f"Action {a_id}"),
            s=40,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

    # Stepped connecting line
    ax.step(gens, actions, where="mid", color="gray", alpha=0.4, linewidth=1, zorder=1)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Action ID")
    ax.set_yticks(list(_ACTION_LABELS.keys()))
    ax.set_yticklabels(
        [_ACTION_LABELS[k] for k in sorted(_ACTION_LABELS.keys())],
        fontsize=8,
    )
    ax.set_title("Learned Heuristic Selection Policy (AOS)")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_02_heuristic_policy.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_03_convergence(run_dir: Path, plt) -> Path | None:
    """Fig 3: Hard / soft penalty convergence (dual Y-axis)."""
    csv_path = run_dir / "evaluation_trajectory.csv"
    if not csv_path.exists():
        csv_path = run_dir / "evaluation_trajectory_200.csv"
    if not csv_path.exists():
        logger.warning("evaluation_trajectory*.csv not found — skipping Fig 3")
        return None

    rows = _read_csv(csv_path)
    gens = np.array([int(r["generation"]) for r in rows])

    # Gracefully handle missing columns
    has_hard = "best_hard" in rows[0]
    has_soft = "best_soft" in rows[0]
    if not has_hard and not has_soft:
        logger.warning("No best_hard / best_soft columns — skipping Fig 3")
        return None

    fig, ax_h = plt.subplots(figsize=(6.5, 4))

    if has_hard:
        hard = np.array([float(r["best_hard"]) for r in rows])
        ln1 = ax_h.plot(
            gens,
            hard,
            color=_CB[5],
            linewidth=2,
            marker="o",
            markersize=3,
            label="Best Hard Penalty",
        )
        ax_h.set_ylabel("Hard Penalty (violations)", color=_CB[5])
        ax_h.tick_params(axis="y", labelcolor=_CB[5])

    if has_soft:
        soft = np.array([float(r["best_soft"]) for r in rows])
        ax_s = ax_h.twinx()
        ln2 = ax_s.plot(
            gens,
            soft,
            color=_CB[2],
            linewidth=2,
            marker="s",
            markersize=3,
            linestyle="--",
            label="Best Soft Penalty",
        )
        ax_s.set_ylabel("Soft Penalty", color=_CB[2])
        ax_s.tick_params(axis="y", labelcolor=_CB[2])
        ax_s.spines["right"].set_visible(True)

    ax_h.set_xlabel("Generation")
    ax_h.set_title("Evaluation: Constraint Violation Descent")

    # Combined legend
    handles, labels = [], []
    if has_hard:
        handles += ln1
        labels += [h.get_label() for h in ln1]
    if has_soft:
        handles += ln2
        labels += [h.get_label() for h in ln2]
    ax_h.legend(handles, labels, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_03_eval_convergence.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


def _fig_04_baseline_comparison(
    run_dir: Path,
    plt,
    baselines_csv: str | Path | None = None,
) -> Path | None:
    """Fig 4: PPO adaptive vs 6 static baselines (hard + soft panels).

    Main panel: Best Hard Constraint vs Generation.
    Inset/secondary panel: Best Soft Constraint vs Generation.
    """
    # -- Load PPO evaluation trajectory ----------------------------------
    eval_csv = run_dir / "evaluation_trajectory.csv"
    if not eval_csv.exists():
        logger.warning("evaluation_trajectory.csv not found — skipping Fig 4")
        return None

    if baselines_csv is None:
        logger.warning("No baselines_csv provided — skipping Fig 4")
        return None
    baselines_csv = Path(baselines_csv)
    if not baselines_csv.exists():
        logger.warning("static_baselines.csv not found — skipping Fig 4")
        return None

    ppo_rows = _read_csv(eval_csv)
    bl_rows = _read_csv(baselines_csv)

    ppo_gens = np.array([int(r["generation"]) for r in ppo_rows])
    ppo_hard = np.array([float(r["best_hard"]) for r in ppo_rows])
    ppo_soft = np.array([float(r["best_soft"]) for r in ppo_rows])

    # -- Aggregate baselines (mean across seeds per generation) ----------
    action_ids = sorted(set(int(r["action_id"]) for r in bl_rows))

    # Build per-action trajectories
    bl_data: dict[int, dict] = {}
    for aid in action_ids:
        aid_rows = [r for r in bl_rows if int(r["action_id"]) == aid]
        seeds = sorted(set(int(r["seed"]) for r in aid_rows))

        # Collect per-seed trajectories, then average
        gen_set = sorted(set(int(r["generation"]) for r in aid_rows))
        hard_by_gen = {g: [] for g in gen_set}
        soft_by_gen = {g: [] for g in gen_set}

        for r in aid_rows:
            g = int(r["generation"])
            hard_by_gen[g].append(float(r["best_hard"]))
            soft_by_gen[g].append(float(r["best_soft"]))

        gens = np.array(gen_set)
        avg_hard = np.array([np.mean(hard_by_gen[g]) for g in gen_set])
        avg_soft = np.array([np.mean(soft_by_gen[g]) for g in gen_set])
        name = aid_rows[0].get("action_name", f"action_{aid}")

        bl_data[aid] = {
            "gens": gens,
            "hard": avg_hard,
            "soft": avg_soft,
            "name": name,
            "n_seeds": len(seeds),
        }

    # -- Plot: 2-panel figure (hard on top, soft on bottom) --------------
    fig, (ax_h, ax_s) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    # Static baselines (dashed)
    for aid in action_ids:
        d = bl_data[aid]
        label = _ACTION_LABELS.get(aid, d["name"])
        color = _CB[aid % len(_CB)]
        ax_h.plot(
            d["gens"],
            d["hard"],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"Static: {label}",
        )
        ax_s.plot(
            d["gens"],
            d["soft"],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"Static: {label}",
        )

    # PPO adaptive (solid, bold, black)
    ax_h.plot(
        ppo_gens,
        ppo_hard,
        color=_CB[7],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label="PPO Adaptive",
        zorder=10,
    )
    ax_s.plot(
        ppo_gens,
        ppo_soft,
        color=_CB[7],
        linewidth=2.5,
        marker="s",
        markersize=3,
        label="PPO Adaptive",
        zorder=10,
    )

    ax_h.set_ylabel("Best Hard Constraint Penalty")
    ax_h.set_title("PPO Adaptive vs Static Baselines — Hard Constraints")
    ax_h.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)

    ax_s.set_xlabel("Generation")
    ax_s.set_ylabel("Best Soft Constraint Penalty")
    ax_s.set_title("PPO Adaptive vs Static Baselines — Soft Constraints")
    ax_s.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_04_baseline_comparison.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ------------------------------------------------------------------
# Fig 5-12: New thesis figures
# ------------------------------------------------------------------


def _fig_05_loss_curves(run_dir: Path, plt) -> Path | None:
    """Fig 5: Policy gradient loss and value loss over training."""
    csv_path = run_dir / "sb3_training_metrics.csv"
    if not csv_path.exists():
        logger.warning("sb3_training_metrics.csv not found — skipping Fig 5")
        return None

    rows = _read_csv(csv_path)
    if not rows:
        return None

    ts = np.array([int(r["timestep"]) for r in rows])
    has_pg = "policy_gradient_loss" in rows[0]
    has_vl = "value_loss" in rows[0]
    if not has_pg and not has_vl:
        logger.warning("No loss columns in sb3_training_metrics — skipping Fig 5")
        return None

    fig, ax1 = plt.subplots(figsize=(6.5, 4))

    if has_pg:
        pg_loss = np.array([float(r["policy_gradient_loss"]) for r in rows])
        ax1.plot(ts, pg_loss, color=_CB[4], linewidth=1.5, label="Policy Loss")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Policy Gradient Loss", color=_CB[4])
    ax1.tick_params(axis="y", labelcolor=_CB[4])

    if has_vl:
        ax2 = ax1.twinx()
        vl = np.array([float(r["value_loss"]) for r in rows])
        ax2.plot(
            ts, vl, color=_CB[5], linewidth=1.5, linestyle="--", label="Value Loss"
        )
        ax2.set_ylabel("Value Loss", color=_CB[5])
        ax2.tick_params(axis="y", labelcolor=_CB[5])
        ax2.spines["right"].set_visible(True)

    ax1.set_title("Training Loss Curves")

    # Combined legend
    handles, labels = ax1.get_legend_handles_labels()
    if has_vl:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax1.legend(handles, labels, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_05_loss_curves.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_06_entropy(run_dir: Path, plt) -> Path | None:
    """Fig 6: Policy entropy over training (exploration measure)."""
    csv_path = run_dir / "sb3_training_metrics.csv"
    if not csv_path.exists():
        logger.warning("sb3_training_metrics.csv not found — skipping Fig 6")
        return None

    rows = _read_csv(csv_path)
    if not rows or "entropy_loss" not in rows[0]:
        logger.warning("No entropy_loss column — skipping Fig 6")
        return None

    ts = np.array([int(r["timestep"]) for r in rows])
    entropy = np.array([float(r["entropy_loss"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(ts, entropy, color=_CB[2], linewidth=1.5)
    ax.fill_between(ts, entropy, alpha=0.15, color=_CB[2])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Entropy Loss")
    ax.set_title("Policy Entropy Over Training")

    # Add secondary metrics if available
    has_kl = "approx_kl" in rows[0]
    has_clip = "clip_fraction" in rows[0]
    if has_kl or has_clip:
        ax2 = ax.twinx()
        if has_kl:
            kl = np.array([float(r["approx_kl"]) for r in rows])
            ax2.plot(
                ts, kl, color=_CB[6], linewidth=1, linestyle=":", label="Approx KL"
            )
        if has_clip:
            clip = np.array([float(r["clip_fraction"]) for r in rows])
            ax2.plot(
                ts, clip, color=_CB[0], linewidth=1, linestyle="-.", label="Clip Frac"
            )
        ax2.set_ylabel("KL / Clip Fraction")
        ax2.spines["right"].set_visible(True)
        ax2.legend(loc="upper left", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_06_entropy.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_07_action_frequency(run_dir: Path, plt) -> Path | None:
    """Fig 7: Stacked-area of action proportions per episode."""
    csv_path = run_dir / "training_curve.csv"
    if not csv_path.exists():
        return None

    rows = _read_csv(csv_path)
    if not rows:
        return None

    # Detect action count columns
    action_cols = sorted(
        [c for c in rows[0] if c.startswith("action_") and c.endswith("_count")]
    )
    if not action_cols:
        logger.warning("No action_N_count columns — skipping Fig 7")
        return None

    episodes = np.array([int(r["episode"]) for r in rows])
    counts = np.array([[int(r[c]) for c in action_cols] for r in rows], dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1  # avoid div-by-zero
    proportions = counts / totals

    fig, ax = plt.subplots(figsize=(7, 4))
    n_actions = proportions.shape[1]
    colors = [_CB[i % len(_CB)] for i in range(n_actions)]
    labels = [_ACTION_LABELS.get(i, f"Action {i}") for i in range(n_actions)]

    ax.stackplot(episodes, proportions.T, labels=labels, colors=colors, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Proportion")
    ax.set_title("Action Selection Distribution Over Training")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_07_action_frequency.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_08_step_reward(run_dir: Path, plt, window: int = 50) -> Path | None:
    """Fig 8: Step-level reward scatter with rolling mean + best_hard descent."""
    csv_path = run_dir / "step_log.csv"
    if not csv_path.exists():
        return None

    rows = _read_csv(csv_path)
    if not rows or len(rows) < 2:
        return None

    ts = np.array([int(r["timestep"]) for r in rows])
    rewards = np.array([float(r["reward"]) for r in rows])
    best_hard = np.array([float(r["best_hard"]) for r in rows])

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Reward scatter (decimated for large datasets)
    stride = max(1, len(ts) // 2000)
    ax1.scatter(
        ts[::stride],
        rewards[::stride],
        s=1,
        alpha=0.15,
        color=_CB[1],
        rasterized=True,
    )
    smoothed_r = _rolling_mean(rewards, window)
    ax1.plot(ts, smoothed_r, color=_CB[4], linewidth=1.5, label=f"Reward (MA-{window})")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Step Reward", color=_CB[4])
    ax1.tick_params(axis="y", labelcolor=_CB[4])

    # Best hard on twin axis
    ax2 = ax1.twinx()
    ax2.plot(ts, best_hard, color=_CB[5], linewidth=1, alpha=0.7, label="Best Hard")
    ax2.set_ylabel("Best Hard Violations", color=_CB[5])
    ax2.tick_params(axis="y", labelcolor=_CB[5])
    ax2.spines["right"].set_visible(True)

    ax1.set_title("Step-Level Reward and Hard Constraint Descent")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2, labels1 + labels2, loc="upper right", framealpha=0.9
    )

    fig.tight_layout()
    out = run_dir / "fig_08_step_reward.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_09_feasibility(run_dir: Path, plt, window: int = 50) -> Path | None:
    """Fig 9: Feasibility rate over training steps."""
    csv_path = run_dir / "step_log.csv"
    if not csv_path.exists():
        return None

    rows = _read_csv(csv_path)
    if not rows or "feasible_frac" not in rows[0]:
        return None

    ts = np.array([int(r["timestep"]) for r in rows])
    feas = np.array([float(r["feasible_frac"]) for r in rows])
    # Replace NaN with 0
    feas = np.nan_to_num(feas, nan=0.0)
    smoothed = _rolling_mean(feas, window)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.fill_between(ts, smoothed, alpha=0.2, color=_CB[2])
    ax.plot(
        ts, smoothed, color=_CB[2], linewidth=1.5, label=f"Feasible Frac (MA-{window})"
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Feasible Fraction")
    ax.set_title("Population Feasibility Rate Over Training")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_09_feasibility_rate.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_10_delta_hard_boxplot(run_dir: Path, plt) -> Path | None:
    """Fig 10: Per-action box-plot of delta_hard (repair effectiveness)."""
    csv_path = run_dir / "step_log.csv"
    if not csv_path.exists():
        return None

    rows = _read_csv(csv_path)
    if not rows or "delta_hard" not in rows[0]:
        return None

    # Group delta_hard by action
    action_deltas: dict[int, list[float]] = {}
    for r in rows:
        act = int(r["action"])
        if act < 0:
            continue
        dh = float(r["delta_hard"])
        if np.isnan(dh):
            continue
        action_deltas.setdefault(act, []).append(dh)

    if not action_deltas:
        return None

    action_ids = sorted(action_deltas.keys())
    data = [action_deltas[a] for a in action_ids]
    labels = [_ACTION_LABELS.get(a, f"Act {a}") for a in action_ids]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,  # outliers clutter thesis plots
        medianprops=dict(color="black", linewidth=1.5),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_CB[action_ids[i] % len(_CB)])
        patch.set_alpha(0.7)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
    ax.set_ylabel(r"$\Delta$ Hard Violations (negative = improvement)")
    ax.set_title("Per-Action Repair Effectiveness")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    out = run_dir / "fig_10_delta_hard_boxplot.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_11_mask_usage(run_dir: Path, plt) -> Path | None:
    """Fig 11: MaskablePPO mask blocking percentage over episodes."""
    csv_path = run_dir / "training_curve.csv"
    if not csv_path.exists():
        return None

    rows = _read_csv(csv_path)
    if not rows or "mask_blocks_pct" not in rows[0]:
        return None

    episodes = np.array([int(r["episode"]) for r in rows])
    mask_pct = np.array([float(r["mask_blocks_pct"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(episodes, mask_pct, alpha=0.4, color=_CB[6], width=0.8, label="Mask Block %")
    smoothed = _rolling_mean(mask_pct, max(3, len(episodes) // 10))
    ax.plot(episodes, smoothed, color=_CB[4], linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Mask Blocking (%)")
    ax.set_title("State-Conditioned Action Masking Over Training")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = run_dir / "fig_11_mask_usage.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def _fig_12_cross_agent(
    run_dir: Path, plt, baselines_dir: Path | None = None
) -> Path | None:
    """Fig 12: Bar chart comparing final metrics across agents."""
    if baselines_dir is None:
        baselines_dir = run_dir.parent.parent / "baselines"
    if not baselines_dir.exists():
        logger.warning("baselines/ not found — skipping Fig 12")
        return None

    # Collect final-row metrics from each baseline CSV
    agents: dict[str, dict[str, float]] = {}
    for csv_file in sorted(baselines_dir.glob("*_eval_200.csv")):
        agent_name = csv_file.stem.replace("_eval_200", "").upper()
        rows = _read_csv(csv_file)
        if rows:
            final = rows[-1]
            agents[agent_name] = {
                "best_hard": float(final["best_hard"]),
                "best_soft": float(final["best_soft"]),
            }

    if len(agents) < 2:
        logger.warning("Need >=2 agent baselines for comparison — skipping Fig 12")
        return None

    names = list(agents.keys())
    hard_vals = [agents[n]["best_hard"] for n in names]
    soft_vals = [agents[n]["best_soft"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 4))
    bars1 = ax1.bar(
        x - width / 2, hard_vals, width, label="Final Hard", color=_CB[5], alpha=0.8
    )
    ax1.set_ylabel("Hard Constraint Violations", color=_CB[5])
    ax1.tick_params(axis="y", labelcolor=_CB[5])

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2, soft_vals, width, label="Final Soft", color=_CB[2], alpha=0.8
    )
    ax2.set_ylabel("Soft Constraint Penalty", color=_CB[2])
    ax2.tick_params(axis="y", labelcolor=_CB[2])
    ax2.spines["right"].set_visible(True)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_title("Cross-Agent Final Performance Comparison")
    ax1.legend(
        [bars1, bars2], ["Final Hard", "Final Soft"], loc="upper right", framealpha=0.9
    )

    fig.tight_layout()
    out = run_dir / "fig_12_cross_agent_comparison.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def generate_plots(
    run_dir: str | Path,
    rolling_window: int = 10,
    baselines_csv: str | Path | None = None,
    baselines_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all thesis figures from CSVs in *run_dir*.

    Args:
        run_dir: Directory containing training CSV files.
        rolling_window: Smoothing window for learning curve (Fig 1).
        baselines_csv: Path to static_baselines.csv for Fig 4.
        baselines_dir: Directory containing per-agent eval CSVs for Fig 12.

    Returns:
        Paths to the generated PDF files.
    """
    run_dir = Path(run_dir)
    bl_dir = Path(baselines_dir) if baselines_dir else None
    plt = _setup_style()
    pdfs: list[Path] = []

    for fig_fn in (
        lambda: _fig_01_learning_curve(run_dir, plt, rolling_window),
        lambda: _fig_02_aos_policy(run_dir, plt),
        lambda: _fig_03_convergence(run_dir, plt),
        lambda: _fig_04_baseline_comparison(run_dir, plt, baselines_csv),
        lambda: _fig_05_loss_curves(run_dir, plt),
        lambda: _fig_06_entropy(run_dir, plt),
        lambda: _fig_07_action_frequency(run_dir, plt),
        lambda: _fig_08_step_reward(run_dir, plt, rolling_window),
        lambda: _fig_09_feasibility(run_dir, plt, rolling_window),
        lambda: _fig_10_delta_hard_boxplot(run_dir, plt),
        lambda: _fig_11_mask_usage(run_dir, plt),
        lambda: _fig_12_cross_agent(run_dir, plt, bl_dir),
    ):
        p = fig_fn()
        if p is not None:
            pdfs.append(p)

    return pdfs
