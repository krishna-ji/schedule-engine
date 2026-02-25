r"""Generate publication-ready thesis figures from RL training CSVs.

Produces three PDF figures suitable for direct \LaTeX\ inclusion:

1. **Learning Curve** — episode reward with rolling-window smoothing.
2. **AOS Policy Map** — action selection scatter over generations.
3. **Convergence Trajectory** — hard / soft penalty descent (dual axis).

All figures use Times New Roman (serif), 300 DPI, and the Okabe-Ito
colourblind-safe palette.

Usage
-----

.. code-block:: python

    from src.rl.training.plot_thesis_figures import generate_plots
    pdfs = generate_plots(Path("output/rl_vectorized/20260225_120000"))
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

# Canonical action labels (must agree with vectorized_ops.py Elite 8)
_ACTION_LABELS: dict[int, str] = {
    0: "Spatial Resource (SRE)",
    1: "Faculty Temporal (FTE)",
    2: "Cohort Temporal (CTE)",
    3: "Subcohort Sync (SSCP)",
    4: "Universal Feasibility",
    5: "Quanta Perturbation",
    6: "Spatial Perturbation",
    7: "Meridian Compaction",
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
        logger.warning("evaluation_trajectory.csv not found — skipping Fig 2")
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
        logger.warning("evaluation_trajectory.csv not found — skipping Fig 3")
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


def generate_plots(run_dir: str | Path, rolling_window: int = 10) -> list[Path]:
    """Generate all three thesis figures from CSVs in *run_dir*.

    Parameters
    ----------
    run_dir : str | Path
        Directory containing ``training_curve.csv`` and
        ``evaluation_trajectory.csv``.
    rolling_window : int
        Smoothing window for the learning curve (Fig 1).

    Returns
    -------
    list[Path]
        Paths to the generated PDF files.
    """
    run_dir = Path(run_dir)
    plt = _setup_style()
    pdfs: list[Path] = []

    for fig_fn in (
        lambda: _fig_01_learning_curve(run_dir, plt, rolling_window),
        lambda: _fig_02_aos_policy(run_dir, plt),
        lambda: _fig_03_convergence(run_dir, plt),
    ):
        p = fig_fn()
        if p is not None:
            pdfs.append(p)

    return pdfs
