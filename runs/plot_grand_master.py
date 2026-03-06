#!/usr/bin/env python3
r"""Grand Master Plot — Publication-Ready Comparative Analysis.

Ingests convergence data from all solvers and plots
**Generation vs.\ Best Hard Constraint** on a single graph:

  1. Pure GA      (Mode A — ``output/ga_baseline/``)
  2. Memetic GA   (Mode B — ``output/ga_memetic/``)
  3. Random       (``output/baselines/random_eval_200.csv``)
  4. Round-Robin  (``output/baselines/round_robin_eval_200.csv``)
  5. UCB1         (``output/baselines/ucb1_eval_200.csv``)
  6. MaskablePPO  (Titan — ``output/titan/`` or step log)

Usage::

    python runs/plot_grand_master.py

Output::

    output/figures/grand_master_hard_convergence.pdf
    output/figures/grand_master_hard_convergence.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plot_grand_master")


# ======================================================================
# Data Loaders
# ======================================================================


def load_ga_baseline() -> tuple[np.ndarray, np.ndarray] | None:
    """Load Pure GA convergence: (generations, best_hard)."""
    base = PROJECT_ROOT / "output" / "ga_baseline"
    if not base.exists():
        logger.warning("GA baseline dir not found: %s", base)
        return None

    # Find most recent run with convergence_history.csv
    runs = sorted(base.iterdir(), reverse=True)
    for run_dir in runs:
        csv_path = run_dir / "convergence_history.csv"
        if csv_path.exists():
            import csv as csvmod

            gens, hards = [], []
            with open(csv_path) as f:
                reader = csvmod.DictReader(f)
                for row in reader:
                    gens.append(int(row["Gen"]))
                    hards.append(float(row["Best_Hard"]))
            logger.info("GA Baseline: %d gens from %s", len(gens), csv_path)
            return np.array(gens), np.array(hards)

    # Fallback: try results.json
    for run_dir in runs:
        json_path = run_dir / "results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            ch = data.get("convergence_hard", [])
            if ch:
                gens = np.arange(1, len(ch) + 1)
                logger.info("GA Baseline: %d gens from results.json", len(ch))
                return gens, np.array(ch, dtype=float)

    logger.warning("No GA baseline convergence data found")
    return None


def load_ga_memetic() -> tuple[np.ndarray, np.ndarray] | None:
    """Load Memetic GA convergence: (generations, best_hard)."""
    base = PROJECT_ROOT / "output" / "ga_memetic"
    if not base.exists():
        logger.warning("GA memetic dir not found: %s", base)
        return None

    runs = sorted(base.iterdir(), reverse=True)
    for run_dir in runs:
        csv_path = run_dir / "convergence_history.csv"
        if csv_path.exists():
            import csv as csvmod

            gens, hards = [], []
            with open(csv_path) as f:
                reader = csvmod.DictReader(f)
                for row in reader:
                    gens.append(int(row["Gen"]))
                    hards.append(float(row["Best_Hard"]))
            logger.info("Memetic GA: %d gens from %s", len(gens), csv_path)
            return np.array(gens), np.array(hards)

    for run_dir in runs:
        json_path = run_dir / "results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            ch = data.get("convergence_hard", [])
            if ch:
                gens = np.arange(1, len(ch) + 1)
                logger.info("Memetic GA: %d gens from results.json", len(ch))
                return gens, np.array(ch, dtype=float)

    logger.warning("No GA memetic convergence data found")
    return None


def load_baseline_csv(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load RL baseline CSV: (generations, best_hard)."""
    csv_path = PROJECT_ROOT / "output" / "baselines" / f"{name}_eval_200.csv"
    if not csv_path.exists():
        logger.warning("Baseline CSV not found: %s", csv_path)
        return None

    import csv as csvmod

    gens, hards = [], []
    with open(csv_path) as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            gens.append(int(row["generation"]))
            hards.append(float(row["best_hard"]))

    logger.info("%s baseline: %d gens from %s", name, len(gens), csv_path)
    return np.array(gens), np.array(hards)


def load_titan_model() -> tuple[np.ndarray, np.ndarray] | None:
    """Load Titan MaskablePPO training trajectory.

    Attempts to load from the titan training log CSV.  Falls back to
    evaluating the saved model for 200 generations if no training log
    exists.
    """
    # Try training log first
    titan_dir = PROJECT_ROOT / "output" / "titan"
    if titan_dir.exists():
        runs = sorted(titan_dir.iterdir(), reverse=True)
        for run_dir in runs:
            log_path = run_dir / "titan_training_log.csv"
            if log_path.exists():
                import csv as csvmod

                episodes, best_hards = [], []
                with open(log_path) as f:
                    reader = csvmod.DictReader(f)
                    for row in reader:
                        episodes.append(int(row["episode"]))
                        best_hards.append(float(row["best_hard_ever"]))
                if episodes:
                    # Convert episodes to generation-equivalents
                    # Each episode = MAX_GENERATIONS (50) GA generations
                    GENS_PER_EPISODE = 50
                    gen_equiv = np.array(episodes) * GENS_PER_EPISODE
                    logger.info(
                        "Titan: %d episodes (gen-equiv %d→%d) from training log",
                        len(episodes),
                        gen_equiv[0],
                        gen_equiv[-1],
                    )
                    return gen_equiv, np.array(best_hards)

    # Try evaluating saved model
    model_path = PROJECT_ROOT / "output" / "models" / "maskable_ppo_titan.zip"
    if not model_path.exists():
        logger.warning("No Titan model or training log found")
        return None

    logger.info("Evaluating Titan model for 200 generations...")
    return _evaluate_titan_model(model_path)


def _evaluate_titan_model(
    model_path: Path,
    n_gens: int = 200,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run saved Titan model through 200 generations and record trajectory."""
    try:
        from sb3_contrib import MaskablePPO

        from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

        model = MaskablePPO.load(str(model_path))

        env = PymooHyperHeuristicEnv(
            pkl_path=".cache/events_with_domains.pkl",
            max_generations=n_gens,
            pop_size=120,
            algorithm_name="nsga2",
            seed=42,
            acceptance_tolerance=0.0,
        )

        obs, info = env.reset()
        gens = [1]
        best_hards = [info.get("best_hard", np.inf)]
        best_ever = best_hards[0]

        for gen in range(2, n_gens + 2):
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, terminated, truncated, info = env.step(int(action))

            best_h = info.get("best_hard", np.inf)
            best_ever = min(best_ever, best_h)
            gens.append(gen)
            best_hards.append(best_ever)

            if terminated or truncated:
                break

        env.close()
        logger.info(
            "Titan model evaluated: %d gens, final hard=%.1f", len(gens), best_ever
        )
        return np.array(gens), np.array(best_hards)

    except Exception as e:
        logger.error("Failed to evaluate Titan model: %s", e)
        return None


# ======================================================================
# Plotting
# ======================================================================


def plot_grand_master():
    """Generate the publication-ready comparison plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = PROJECT_ROOT / "output" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -- Load all data sources ---------------------------------------------
    datasets = {}

    ga_base = load_ga_baseline()
    if ga_base is not None:
        datasets["Pure GA (Mode A)"] = ga_base

    ga_mem = load_ga_memetic()
    if ga_mem is not None:
        datasets["Memetic GA (Mode B)"] = ga_mem

    random_data = load_baseline_csv("random")
    if random_data is not None:
        datasets["Random Heuristic"] = random_data

    rr_data = load_baseline_csv("round_robin")
    if rr_data is not None:
        datasets["Round-Robin"] = rr_data

    ucb1_data = load_baseline_csv("ucb1")
    if ucb1_data is not None:
        datasets["UCB1 Bandit"] = ucb1_data

    titan_data = load_titan_model()
    if titan_data is not None:
        datasets["MaskablePPO (Titan)"] = titan_data

    if not datasets:
        logger.error("No data sources found! Run baselines and/or Titan first.")
        sys.exit(1)

    logger.info("Loaded %d data sources for plotting", len(datasets))

    # -- Style configuration -----------------------------------------------
    # Publication-quality settings
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    # Color and style for each solver
    style_map = {
        "Pure GA (Mode A)": {"color": "#888888", "ls": "--", "lw": 1.5, "alpha": 0.7},
        "Memetic GA (Mode B)": {"color": "#2ca02c", "ls": "-", "lw": 2.0, "alpha": 0.9},
        "Random Heuristic": {"color": "#7f7f7f", "ls": ":", "lw": 1.2, "alpha": 0.6},
        "Round-Robin": {"color": "#9467bd", "ls": "-.", "lw": 1.3, "alpha": 0.7},
        "UCB1 Bandit": {"color": "#ff7f0e", "ls": "-.", "lw": 1.5, "alpha": 0.8},
        "MaskablePPO (Titan)": {"color": "#d62728", "ls": "-", "lw": 2.5, "alpha": 1.0},
    }

    # -- Create figure -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, (gens, hards) in datasets.items():
        # Compute running minimum (monotonic best-so-far)
        best_so_far = np.minimum.accumulate(hards)
        style = style_map.get(
            name, {"color": "black", "ls": "-", "lw": 1.5, "alpha": 0.8}
        )
        ax.plot(
            gens,
            best_so_far,
            label=name,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            alpha=style["alpha"],
        )

        # Annotate final value
        final_val = best_so_far[-1]
        ax.annotate(
            f"{final_val:.0f}",
            xy=(gens[-1], final_val),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=style["color"],
            va="center",
        )

    # -- Formatting --------------------------------------------------------
    ax.set_xlabel("Generation (RL episodes × 50 for Titan)")
    ax.set_ylabel("Best Hard Constraint Violations")
    ax.set_title("Convergence Comparison: Hard Constraint Violations")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add feasibility line at y=0
    ax.axhline(
        y=0, color="green", linestyle="--", linewidth=0.8, alpha=0.5, label="_Feasible"
    )

    plt.tight_layout()

    # -- Save --------------------------------------------------------------
    pdf_path = fig_dir / "grand_master_hard_convergence.pdf"
    png_path = fig_dir / "grand_master_hard_convergence.png"
    fig.savefig(str(pdf_path), format="pdf")
    fig.savefig(str(png_path), format="png")
    plt.close(fig)

    logger.info("Saved: %s", pdf_path)
    logger.info("Saved: %s", png_path)

    # -- Print summary table -----------------------------------------------
    print("\n" + "=" * 70)
    print("  GRAND MASTER COMPARISON — FINAL RESULTS")
    print("=" * 70)
    print(f"  {'Solver':<25s} {'Gens':>5s} {'Final Hard':>12s} {'Min Hard':>12s}")
    print("  " + "-" * 56)
    for name, (gens, hards) in sorted(datasets.items(), key=lambda x: np.min(x[1][1])):
        best_so_far = np.minimum.accumulate(hards)
        print(
            f"  {name:<25s} {len(gens):>5d} {best_so_far[-1]:>12.1f} {best_so_far.min():>12.1f}"
        )
    print("=" * 70)
    print(f"  Figures: {fig_dir}")
    print("=" * 70)


# ======================================================================
# Main
# ======================================================================


def main():
    plot_grand_master()


if __name__ == "__main__":
    main()
