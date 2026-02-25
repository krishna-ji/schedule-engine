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
MAX_GENERATIONS = 50  # episode length (env steps)
TOTAL_TIMESTEPS = 10_000  # PPO training budget (10k for validation)
EVAL_GENERATIONS = 50  # evaluation episode length
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
    logger.info(
        "  pop_size: %d  max_gen: %d  timesteps: %d",
        POP_SIZE,
        MAX_GENERATIONS,
        TOTAL_TIMESTEPS,
    )
    logger.info("=" * 60)

    # -- Environment -------------------------------------------------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=0.0,  # strict mode: reject any hard degradation
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


def _build_eval_row(info: dict, action_id: int, action_name: str, reward: float) -> dict:
    """Build one evaluation CSV row with full constraint breakdown."""
    from src.rl.gym_env.fast_state_encoder import (
        HARD_CONSTRAINT_NAMES,
        SOFT_CONSTRAINT_NAMES,
    )

    row: dict[str, object] = {
        "generation": info["generation"],
        "action_id": action_id,
        "action_name": action_name,
        "best_hard": info["best_hard"],
        "best_soft": info["best_soft"],
        "mean_hard": info["mean_hard"],
        "mean_soft": info["mean_soft"],
        "feasible_frac": info["feasible_frac"],
        "reward": reward,
        "rejected": info.get("rejected", False),
        "delta_hard": info.get("delta_hard", 0.0),
    }
    # 8 hard constraint columns
    for name in HARD_CONSTRAINT_NAMES:
        row[f"cv_{name}"] = info.get(f"cv_{name}", 0.0)
    # 4 soft constraint columns
    for name in SOFT_CONSTRAINT_NAMES:
        row[f"cv_{name}"] = info.get(f"cv_{name}", 0.0)
    return row


def evaluate(model, run_dir: Path) -> Path:
    """Run deterministic evaluation and export CSV with constraint breakdown."""
    from src.rl.actions.vectorized_ops import ACTION_NAMES
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    logger.info("-" * 60)
    logger.info("Deterministic evaluation (%d generations)", EVAL_GENERATIONS)
    logger.info("-" * 60)

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED + 1000,  # different seed for eval
        acceptance_tolerance=0.0,  # strict mode for eval
    )

    obs, info = env.reset()
    rows: list[dict] = []

    # Record initial state (gen 1 from reset)
    rows.append(_build_eval_row(info, action_id=-1, action_name="init", reward=0.0))

    cumulative_reward = 0.0
    for gen in range(EVAL_GENERATIONS - 1):  # -1 because reset already ran gen 1
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        rows.append(
            _build_eval_row(
                info,
                action_id=action,
                action_name=ACTION_NAMES.get(action, f"action_{action}"),
                reward=reward,
            )
        )

        logger.info(
            "  Gen %2d | act=%d (%s) | hard=%.0f soft=%.0f | feas=%.2f | r=%.4f | rej=%s",
            info["generation"],
            action,
            ACTION_NAMES.get(action, "?"),
            info["best_hard"],
            info["best_soft"],
            info["feasible_frac"],
            reward,
            info.get("rejected", False),
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
    from src.rl.training.plot_thesis_figures import generate_plots

    pdfs = generate_plots(run_dir)

    # -- Summary ------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Run complete: %s", run_dir)
    logger.info("  Model   : ppo_vectorized_hh.zip")
    logger.info("  CSVs    : training_curve.csv, evaluation_trajectory.csv")
    logger.info("  Figures : %s", ", ".join(p.name for p in pdfs))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
