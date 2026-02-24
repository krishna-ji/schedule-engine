#!/usr/bin/env python3
r"""RL 02 — Train PPO on PymooHyperHeuristicEnv + Evaluation + Thesis Plots.

End-to-end pipeline:
  1. Train PPO (SB3) inside PymooHyperHeuristicEnv with ThesisLoggingCallback.
  2. Run deterministic 50-generation evaluation episode.
  3. Export training_curve.csv and evaluation_trajectory.csv.
  4. Generate 3 publication-ready PDF figures (Times New Roman, 300 DPI).

Usage::

    python runs/rl_02_train_vectorized.py

Outputs (in ``output/rl_vectorized/<timestamp>/``)::

    ppo_vectorized_hh.zip          — saved SB3 model
    training_curve.csv             — per-episode training metrics
    step_log.csv                   — per-step training metrics
    evaluation_trajectory.csv      — 50-gen evaluation trace
    fig_01_learning_curve.pdf      — cumulative reward vs episode
    fig_02_heuristic_policy.pdf    — action selection scatter
    fig_03_eval_convergence.pdf    — hard/soft descent trajectory
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (allow running from repo root: python runs/rl_02_...)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_02_train_vectorized")

# ======================================================================
# Configuration
# ======================================================================

SEED = 42
POP_SIZE = 120
MAX_GENERATIONS = 50          # episode length (env steps)
TOTAL_TIMESTEPS = 2_000       # PPO training budget
EVAL_GENERATIONS = 50         # evaluation episode length
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
NET_ARCH = [64, 64]
PKL_PATH = ".cache/events_with_domains.pkl"


# ======================================================================
# 1. Training
# ======================================================================

def train(run_dir: Path) -> None:
    """Train PPO and save model + CSVs."""
    from stable_baselines3 import PPO

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv
    from src.rl.training.thesis_callback import ThesisLoggingCallback

    logger.info("=" * 60)
    logger.info("Phase 36 — PPO Training on PymooHyperHeuristicEnv")
    logger.info("  run_dir : %s", run_dir)
    logger.info("  pop_size: %d  max_gen: %d  timesteps: %d", POP_SIZE, MAX_GENERATIONS, TOTAL_TIMESTEPS)
    logger.info("=" * 60)

    # -- Environment -------------------------------------------------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
    )

    # -- Agent -------------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE,
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=SEED,
        verbose=1,
    )

    # -- Callback -----------------------------------------------------------
    callback = ThesisLoggingCallback(run_dir=run_dir, verbose=1)

    # -- Train --------------------------------------------------------------
    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    train_time = time.perf_counter() - t0
    logger.info("Training complete in %.1fs", train_time)

    # -- Save model ---------------------------------------------------------
    model_path = run_dir / "ppo_vectorized_hh.zip"
    model.save(str(model_path))
    logger.info("Model saved: %s", model_path)

    env.close()
    return model


# ======================================================================
# 2. Evaluation
# ======================================================================

def evaluate(model, run_dir: Path) -> Path:
    """Run deterministic 50-generation evaluation and export CSV."""
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv
    from src.rl.actions.vectorized_ops import ACTION_NAMES

    logger.info("-" * 60)
    logger.info("Deterministic evaluation (%d generations)", EVAL_GENERATIONS)
    logger.info("-" * 60)

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED + 1000,  # different seed for eval
    )

    obs, info = env.reset()
    rows: list[dict] = []

    # Record initial state (gen 1 from reset)
    rows.append({
        "generation": info["generation"],
        "action_id": -1,
        "action_name": "init",
        "best_hard": info["best_hard"],
        "best_soft": info["best_soft"],
        "mean_hard": info["mean_hard"],
        "mean_soft": info["mean_soft"],
        "feasible_frac": info["feasible_frac"],
        "reward": 0.0,
    })

    cumulative_reward = 0.0
    for gen in range(EVAL_GENERATIONS - 1):  # -1 because reset already ran gen 1
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        rows.append({
            "generation": info["generation"],
            "action_id": action,
            "action_name": ACTION_NAMES.get(action, f"action_{action}"),
            "best_hard": info["best_hard"],
            "best_soft": info["best_soft"],
            "mean_hard": info["mean_hard"],
            "mean_soft": info["mean_soft"],
            "feasible_frac": info["feasible_frac"],
            "reward": reward,
        })

        logger.info(
            "  Gen %2d | act=%d (%s) | hard=%.0f soft=%.0f | feas=%.2f | r=%.4f",
            info["generation"], action, ACTION_NAMES.get(action, "?"),
            info["best_hard"], info["best_soft"],
            info["feasible_frac"], reward,
        )

        if terminated or truncated:
            break

    env.close()

    # -- Export CSV ----------------------------------------------------------
    csv_path = run_dir / "evaluation_trajectory.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Evaluation CSV saved: %s (%d rows)", csv_path, len(rows))
    logger.info("Cumulative eval reward: %.4f", cumulative_reward)
    return csv_path


# ======================================================================
# 3. Publication-Ready Plotting
# ======================================================================

# Colorblind-safe palette (Okabe-Ito)
_CB_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]


def _setup_thesis_style():
    """Configure matplotlib for Times New Roman academic styling."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    plt.rcParams.update({
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
    })
    return plt


def generate_thesis_plots(run_dir: Path) -> list[Path]:
    """Read CSVs and generate 3 publication-ready PDF figures.

    Returns list of generated PDF paths.
    """
    import csv as _csv

    plt = _setup_thesis_style()
    pdfs: list[Path] = []

    # ------------------------------------------------------------------
    # Fig 1: Learning Curve (Cumulative Reward vs Episode)
    # ------------------------------------------------------------------
    training_csv = run_dir / "training_curve.csv"
    if training_csv.exists():
        with open(training_csv) as f:
            reader = _csv.DictReader(f)
            train_rows = list(reader)

        episodes = [int(r["episode"]) for r in train_rows]
        ep_rewards = [float(r["episode_reward"]) for r in train_rows]
        cum_rewards = np.cumsum(ep_rewards).tolist()

        fig, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(episodes, cum_rewards, color=_CB_COLORS[4], linewidth=2, label="Cumulative Reward")
        ax1.fill_between(episodes, 0, cum_rewards, alpha=0.1, color=_CB_COLORS[4])

        # Overlay per-episode reward on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(episodes, ep_rewards, alpha=0.35, color=_CB_COLORS[0], width=0.8, label="Episode Reward")
        ax2.set_ylabel("Episode Reward", color=_CB_COLORS[0])
        ax2.tick_params(axis="y", labelcolor=_CB_COLORS[0])
        ax2.spines["right"].set_visible(True)

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Cumulative Reward", color=_CB_COLORS[4])
        ax1.tick_params(axis="y", labelcolor=_CB_COLORS[4])
        ax1.set_title("PPO Training: Learning Curve")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

        fig.tight_layout()
        p = run_dir / "fig_01_learning_curve.pdf"
        fig.savefig(str(p))
        plt.close(fig)
        pdfs.append(p)
        logger.info("Saved: %s", p)
    else:
        logger.warning("training_curve.csv not found — skipping Fig 1")

    # ------------------------------------------------------------------
    # Fig 2: Heuristic Policy (Action Selection over Generations)
    # ------------------------------------------------------------------
    eval_csv = run_dir / "evaluation_trajectory.csv"
    if eval_csv.exists():
        with open(eval_csv) as f:
            reader = _csv.DictReader(f)
            eval_rows = list(reader)

        gens = [int(r["generation"]) for r in eval_rows if int(r["action_id"]) >= 0]
        actions = [int(r["action_id"]) for r in eval_rows if int(r["action_id"]) >= 0]
        action_labels = {
            0: "Room Repair (SRE)",
            1: "Instructor Repair (FTE)",
            2: "SSCP Sync",
            3: "Random Perturb",
            4: "Group Repair (CTE)",
            5: "Full Repair",
        }

        fig, ax = plt.subplots(figsize=(7, 3.5))

        # Scatter with colour per action
        unique_actions = sorted(set(actions))
        for a_id in unique_actions:
            mask = [i for i, a in enumerate(actions) if a == a_id]
            ax.scatter(
                [gens[i] for i in mask],
                [actions[i] for i in mask],
                c=_CB_COLORS[a_id % len(_CB_COLORS)],
                label=action_labels.get(a_id, f"Action {a_id}"),
                s=40,
                edgecolors="black",
                linewidths=0.3,
                zorder=3,
            )

        # Stepped line connecting them
        ax.step(gens, actions, where="mid", color="gray", alpha=0.4, linewidth=1, zorder=1)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Action ID")
        ax.set_yticks(list(action_labels.keys()))
        ax.set_yticklabels([action_labels[k] for k in sorted(action_labels.keys())], fontsize=8)
        ax.set_title("Learned Heuristic Selection Policy")
        ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
        fig.tight_layout()

        p = run_dir / "fig_02_heuristic_policy.pdf"
        fig.savefig(str(p))
        plt.close(fig)
        pdfs.append(p)
        logger.info("Saved: %s", p)
    else:
        logger.warning("evaluation_trajectory.csv not found — skipping Fig 2")

    # ------------------------------------------------------------------
    # Fig 3: Evaluation Convergence (Hard & Soft descent)
    # ------------------------------------------------------------------
    if eval_csv.exists():
        with open(eval_csv) as f:
            reader = _csv.DictReader(f)
            eval_rows = list(reader)

        all_gens = [int(r["generation"]) for r in eval_rows]
        best_hard = [float(r["best_hard"]) for r in eval_rows]
        best_soft = [float(r["best_soft"]) for r in eval_rows]

        fig, ax_h = plt.subplots(figsize=(6.5, 4))

        # Hard penalty (left y-axis)
        ln1 = ax_h.plot(all_gens, best_hard, color=_CB_COLORS[5], linewidth=2, marker="o", markersize=3, label="Best Hard Penalty")
        ax_h.set_xlabel("Generation")
        ax_h.set_ylabel("Hard Penalty (violations)", color=_CB_COLORS[5])
        ax_h.tick_params(axis="y", labelcolor=_CB_COLORS[5])

        # Soft penalty (right y-axis)
        ax_s = ax_h.twinx()
        ln2 = ax_s.plot(all_gens, best_soft, color=_CB_COLORS[2], linewidth=2, marker="s", markersize=3, linestyle="--", label="Best Soft Penalty")
        ax_s.set_ylabel("Soft Penalty", color=_CB_COLORS[2])
        ax_s.tick_params(axis="y", labelcolor=_CB_COLORS[2])
        ax_s.spines["right"].set_visible(True)

        ax_h.set_title("Evaluation: Constraint Violation Descent")

        # Combined legend
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax_h.legend(lns, labs, loc="upper right", framealpha=0.9)

        fig.tight_layout()
        p = run_dir / "fig_03_eval_convergence.pdf"
        fig.savefig(str(p))
        plt.close(fig)
        pdfs.append(p)
        logger.info("Saved: %s", p)

    return pdfs


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    """Full pipeline: train → evaluate → plot."""
    # -- Timestamped run directory ----------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_vectorized" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Train -----------------------------------------------------------
    model = train(run_dir)

    # -- 2. Evaluate --------------------------------------------------------
    evaluate(model, run_dir)

    # -- 3. Plot ------------------------------------------------------------
    pdfs = generate_thesis_plots(run_dir)

    # -- Summary ------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Run complete: %s", run_dir)
    logger.info("  Model   : ppo_vectorized_hh.zip")
    logger.info("  CSVs    : training_curve.csv, evaluation_trajectory.csv")
    logger.info("  Figures : %s", ", ".join(p.name for p in pdfs))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
