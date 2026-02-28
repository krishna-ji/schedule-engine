#!/usr/bin/env python3
r"""Master Thesis Plot -- 6-Trajectory Comparison.

Reads 4 baseline CSVs (PPO, Random, Round-Robin, UCB1) from
output/baselines/, plus the best GA Baseline and Memetic GA
convergence histories, and generates a single publication-ready
figure showing best_hard vs generation for all 6 trajectories.

Usage::

    python runs/plot_master_thesis.py

Outputs::

    output/figures/master_trajectory_comparison.pdf
    output/figures/master_trajectory_comparison.png
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plot_master_thesis")

# ======================================================================
# Configuration
# ======================================================================

BASELINES_DIR = PROJECT_ROOT / "output" / "baselines"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"

# Baseline CSV files
BASELINE_FILES = {
    "PPO (Ours)": BASELINES_DIR / "ppo_eval_200.csv",
    "DQN": BASELINES_DIR / "dqn_eval_200.csv",
    "UCB1 Bandit": BASELINES_DIR / "ucb1_eval_200.csv",
    "Round-Robin": BASELINES_DIR / "round_robin_eval_200.csv",
    "Random": BASELINES_DIR / "random_eval_200.csv",
}

# GA log directories (auto-detect latest)
GA_BASELINE_DIR = PROJECT_ROOT / "output" / "ga_baseline"
GA_MEMETIC_DIR = PROJECT_ROOT / "output" / "ga_memetic"


# ======================================================================
# Data Loading
# ======================================================================


def load_baseline_csv(path: Path) -> tuple[list[int], list[float]]:
    """Load generation vs best_hard from a baseline CSV."""
    gens, hards = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gens.append(int(row["generation"]))
            hards.append(float(row["best_hard"]))
    return gens, hards


def find_best_ga_run(ga_dir: Path) -> Path | None:
    """Find the GA run with the lowest final best_hard."""
    if not ga_dir.exists():
        return None

    best_path = None
    best_final = float("inf")

    for run_dir in sorted(ga_dir.iterdir()):
        conv_csv = run_dir / "convergence_history.csv"
        if not conv_csv.exists():
            continue
        try:
            with open(conv_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    final_hard = float(rows[-1]["Best_Hard"])
                    if final_hard < best_final:
                        best_final = final_hard
                        best_path = conv_csv
        except (KeyError, ValueError, OSError):
            continue

    return best_path


def load_ga_convergence(path: Path) -> tuple[list[int], list[float]]:
    """Load Gen vs Best_Hard from GA convergence_history.csv."""
    gens, hards = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gens.append(int(row["Gen"]))
            hards.append(float(row["Best_Hard"]))
    return gens, hards


# ======================================================================
# Plotting
# ======================================================================


def plot_master_figure():
    """Generate the master trajectory comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # Publication-quality settings
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color and style definitions
    styles = {
        "GA Baseline": {
            "color": "#7f7f7f",
            "linestyle": "-.",
            "linewidth": 1.5,
            "alpha": 0.7,
            "marker": None,
        },
        "Memetic GA": {
            "color": "#2ca02c",
            "linestyle": "--",
            "linewidth": 1.8,
            "alpha": 0.8,
            "marker": None,
        },
        "Random": {
            "color": "#d62728",
            "linestyle": ":",
            "linewidth": 1.5,
            "alpha": 0.6,
            "marker": None,
        },
        "Round-Robin": {
            "color": "#ff7f0e",
            "linestyle": ":",
            "linewidth": 1.5,
            "alpha": 0.6,
            "marker": None,
        },
        "UCB1 Bandit": {
            "color": "#9467bd",
            "linestyle": "--",
            "linewidth": 2.0,
            "alpha": 0.8,
            "marker": None,
        },
        "DQN": {
            "color": "#17becf",
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 0.9,
            "marker": None,
        },
        "PPO (Ours)": {
            "color": "#1f77b4",
            "linestyle": "-",
            "linewidth": 2.5,
            "alpha": 1.0,
            "marker": None,
        },
    }

    plotted = []

    # 1. GA Baseline
    ga_base_path = find_best_ga_run(GA_BASELINE_DIR)
    if ga_base_path:
        gens, hards = load_ga_convergence(ga_base_path)
        s = styles["GA Baseline"]
        ax.plot(gens, hards, label="GA Baseline", **s)
        plotted.append(("GA Baseline", hards[-1] if hards else "N/A"))
        logger.info("Loaded GA Baseline: %s (final=%.1f)", ga_base_path, hards[-1])
    else:
        logger.warning("No GA Baseline convergence data found")

    # 2. Memetic GA
    ga_mem_path = find_best_ga_run(GA_MEMETIC_DIR)
    if ga_mem_path:
        gens, hards = load_ga_convergence(ga_mem_path)
        s = styles["Memetic GA"]
        ax.plot(gens, hards, label="Memetic GA", **s)
        plotted.append(("Memetic GA", hards[-1] if hards else "N/A"))
        logger.info("Loaded Memetic GA: %s (final=%.1f)", ga_mem_path, hards[-1])
    else:
        logger.warning("No Memetic GA convergence data found")

    # 3-7. Baseline CSVs
    for label, csv_path in BASELINE_FILES.items():
        if csv_path.exists():
            gens, hards = load_baseline_csv(csv_path)
            s = styles.get(
                label,
                {
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 1.5,
                    "alpha": 0.8,
                    "marker": None,
                },
            )
            ax.plot(gens, hards, label=label, **s)
            plotted.append((label, hards[-1] if hards else "N/A"))
            logger.info("Loaded %s: %s (final=%.1f)", label, csv_path, hards[-1])
        else:
            logger.warning("Missing: %s at %s", label, csv_path)

    # Labels and styling
    ax.set_xlabel("Generation", fontweight="bold")
    ax.set_ylabel("Best Hard Constraint Violations", fontweight="bold")
    ax.set_title(
        "Trajectory Comparison: GA vs. Traditional AOS vs. Deep RL\n"
        "on University Timetabling Hyper-Heuristic",
        fontweight="bold",
        pad=12,
    )

    # Legend with frame
    legend = ax.legend(
        loc="upper right",
        framealpha=0.9,
        edgecolor="gray",
        fancybox=True,
        shadow=True,
    )

    # Y-axis: don't go below 0
    ax.set_ylim(bottom=0)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "master_trajectory_comparison.pdf"
    png_path = FIGURES_DIR / "master_trajectory_comparison.png"

    fig.savefig(str(pdf_path), format="pdf")
    fig.savefig(str(png_path), format="png")
    plt.close(fig)

    logger.info("Master plot saved: %s", pdf_path)
    logger.info("Master plot saved: %s", png_path)

    # Print summary table
    print("\n" + "=" * 55)
    print("  MASTER TRAJECTORY COMPARISON -- Final best_hard")
    print("=" * 55)
    for label, val in plotted:
        if isinstance(val, float):
            print(f"  {label:<20s}  {val:>10.1f}")
        else:
            print(f"  {label:<20s}  {val:>10s}")
    print("=" * 55 + "\n")


def main():
    plot_master_figure()


if __name__ == "__main__":
    main()
