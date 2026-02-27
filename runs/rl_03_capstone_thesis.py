#!/usr/bin/env python3
r"""RL 03 — Capstone Thesis Run: 150k Training + 200-Gen Evaluation.

End-to-end pipeline:
  1. Train PPO (SB3) for 150,000 timesteps with acceptance_tolerance=10.0
     to allow the agent to cross fitness valleys.
  2. Print the **Heuristic Efficacy Matrix** (per-action ΔHard / ΔSoft).
  3. Run deterministic 200-generation strict evaluation (tolerance=0.0).
  4. Export evaluation_trajectory_200.csv.
  5. Generate thesis-ready PDF figures.

Usage::

    python runs/rl_03_capstone_thesis.py

Outputs (in ``output/rl_capstone/<timestamp>/``)::

    ppo_capstone_final.zip             — saved SB3 model
    training_curve.csv                 — per-episode training metrics
    step_log.csv                       — per-step training metrics
    heuristic_efficacy_matrix.txt      — per-action efficacy audit
    evaluation_trajectory_200.csv      — 200-gen evaluation trace
    fig_01_learning_curve.pdf          — cumulative reward vs episode
    fig_02_heuristic_policy.pdf        — action selection scatter
    fig_03_eval_convergence.pdf        — hard/soft descent trajectory
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
# Path bootstrap (allow running from repo root: python runs/rl_03_...)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_03_capstone_thesis")

# ======================================================================
# Configuration
# ======================================================================

SEED = 42
POP_SIZE = 120
MAX_GENERATIONS = 50  # episode length during training
TOTAL_TIMESTEPS = 150_000  # PPO training budget (capstone)
EVAL_GENERATIONS = 200  # extended evaluation episode length
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
NET_ARCH = [64, 64]
PKL_PATH = ".cache/events_with_domains.pkl"

# Acceptance tolerance: 10.0 during training (explore), 0.0 during eval (exploit)
TRAIN_TOLERANCE = 10.0
EVAL_TOLERANCE = 0.0


# ======================================================================
# 1. Training
# ======================================================================


def train(run_dir: Path):
    """Train PPO with tolerance annealing and save model + CSVs."""
    from stable_baselines3 import PPO

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv
    from src.rl.training.thesis_callback import ThesisLoggingCallback

    logger.info("=" * 60)
    logger.info("Phase 40 — Capstone Thesis Run: PPO Training")
    logger.info("  run_dir           : %s", run_dir)
    logger.info(
        "  pop_size: %d  max_gen: %d  timesteps: %d",
        POP_SIZE,
        MAX_GENERATIONS,
        TOTAL_TIMESTEPS,
    )
    logger.info("  acceptance_tolerance (train): %.1f", TRAIN_TOLERANCE)
    logger.info("=" * 60)

    # -- Environment -------------------------------------------------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=TRAIN_TOLERANCE,  # allow +10 hard to explore
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
    logger.info("Training complete in %.1fs (%.1f min)", train_time, train_time / 60)

    # -- Save model ---------------------------------------------------------
    # Save to run_dir and also to canonical output/models/ path
    model_run = run_dir / "ppo_capstone_final.zip"
    model.save(str(model_run))
    logger.info("Model saved (run): %s", model_run)

    canonical_dir = PROJECT_ROOT / "output" / "models"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / "ppo_capstone_final.zip"
    model.save(str(canonical_path))
    logger.info("Model saved (canonical): %s", canonical_path)

    env.close()
    return model


# ======================================================================
# 2. Evaluation
# ======================================================================


def _build_eval_row(
    info: dict, action_id: int, action_name: str, reward: float
) -> dict:
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
        "delta_soft": info.get("delta_soft", 0.0),
    }
    # 8 hard constraint columns
    for name in HARD_CONSTRAINT_NAMES:
        row[f"cv_{name}"] = info.get(f"cv_{name}", 0.0)
    # 4 soft constraint columns
    for name in SOFT_CONSTRAINT_NAMES:
        row[f"cv_{name}"] = info.get(f"cv_{name}", 0.0)
    return row


def evaluate(model, run_dir: Path) -> Path:
    """Run strict deterministic 200-gen evaluation and export CSV."""
    from src.rl.actions.vectorized_ops import ACTION_NAMES
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    logger.info("-" * 60)
    logger.info(
        "Deterministic evaluation (%d generations, tolerance=%.1f)",
        EVAL_GENERATIONS,
        EVAL_TOLERANCE,
    )
    logger.info("-" * 60)

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED + 1000,  # different seed for eval
        acceptance_tolerance=EVAL_TOLERANCE,  # strict: reject any hard degradation
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

        if gen % 10 == 0 or gen >= EVAL_GENERATIONS - 5:
            logger.info(
                "  Gen %3d | act=%d (%s) | hard=%.0f soft=%.0f | feas=%.2f | r=%.4f | rej=%s",
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
    csv_path = run_dir / "evaluation_trajectory_200.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Evaluation CSV saved: %s (%d rows)", csv_path, len(rows))
    logger.info("Cumulative eval reward: %.4f", cumulative_reward)

    # -- Print final 10 lines -----------------------------------------------
    print("\n" + "=" * 80)
    print("  FINAL 10 LINES OF evaluation_trajectory_200.csv")
    print("=" * 80)
    for row in rows[-10:]:
        print(
            f"  Gen {row['generation']:>3} | {row['action_name']:<32} | "
            f"Hard={row['best_hard']:>8.1f} | Soft={row['best_soft']:>8.1f} | "
            f"Feas={row['feasible_frac']:.2f} | R={row['reward']:>+7.4f} | "
            f"Rej={row['rejected']}"
        )
    print("=" * 80)
    print(
        f"  FINAL @ Gen {rows[-1]['generation']}: "
        f"Best_Hard = {rows[-1]['best_hard']:.1f}  |  "
        f"Best_Soft = {rows[-1]['best_soft']:.1f}"
    )
    print("=" * 80 + "\n")

    return csv_path


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Full pipeline: train → evaluate → plot."""
    # -- Timestamped run directory ----------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_capstone" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CAPSTONE THESIS RUN")
    logger.info("  Run dir: %s", run_dir)
    logger.info(
        "  Train: %d timesteps, tolerance=%.1f", TOTAL_TIMESTEPS, TRAIN_TOLERANCE
    )
    logger.info(
        "  Eval:  %d generations, tolerance=%.1f", EVAL_GENERATIONS, EVAL_TOLERANCE
    )
    logger.info("=" * 60)

    # -- 1. Train -----------------------------------------------------------
    model = train(run_dir)

    # -- 2. Evaluate --------------------------------------------------------
    evaluate(model, run_dir)

    # -- 3. Plot ------------------------------------------------------------
    from src.rl.training.plot_thesis_figures import generate_plots

    pdfs = generate_plots(run_dir)

    # -- Summary ------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Capstone run complete: %s", run_dir)
    logger.info("  Model   : ppo_capstone_final.zip")
    logger.info("  CSVs    : training_curve.csv, evaluation_trajectory_200.csv")
    logger.info("  Efficacy: heuristic_efficacy_matrix.txt")
    logger.info("  Figures : %s", ", ".join(p.name for p in pdfs))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
