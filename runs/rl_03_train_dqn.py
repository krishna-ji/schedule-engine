#!/usr/bin/env python3
r"""RL 04 -- DQN Competitor: 150k Training + 200-Gen Evaluation.

Identical pipeline to the PPO capstone, but using Stable-Baselines3 DQN.

1. Train DQN for 150,000 timesteps (acceptance_tolerance=10.0).
2. Print Heuristic Efficacy Matrix.
3. Run 200-gen strict evaluation (tolerance=0.0).
4. Export dqn_eval_200.csv to output/baselines/.

Usage::

    python runs/rl_03_train_dqn.py

Outputs::

    output/rl_dqn/<timestamp>/dqn_capstone_final.zip
    output/rl_dqn/<timestamp>/training_curve.csv
    output/rl_dqn/<timestamp>/step_log.csv
    output/rl_dqn/<timestamp>/heuristic_efficacy_matrix.txt
    output/baselines/dqn_eval_200.csv
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
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_03_train_dqn")

# ======================================================================
# Configuration
# ======================================================================

SEED = 42
POP_SIZE = 120
MAX_GENERATIONS = 50  # episode length during training
TOTAL_TIMESTEPS = 150_000  # DQN training budget
EVAL_GENERATIONS = 200
LEARNING_RATE = 1e-4
NET_ARCH = [64, 64]
PKL_PATH = ".cache/events_with_domains.pkl"

TRAIN_TOLERANCE = 10.0
EVAL_TOLERANCE = 0.0

# DQN-specific
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
TARGET_UPDATE_INTERVAL = 1000
EXPLORATION_FRACTION = 0.1
EXPLORATION_FINAL_EPS = 0.05


# ======================================================================
# 1. Training
# ======================================================================


def train(run_dir: Path):
    """Train DQN and save model + CSVs."""
    from stable_baselines3 import DQN

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv
    from src.rl.training.thesis_callback import ThesisLoggingCallback

    logger.info("=" * 60)
    logger.info("DQN COMPETITOR -- Training")
    logger.info("  run_dir           : %s", run_dir)
    logger.info(
        "  pop_size: %d  max_gen: %d  timesteps: %d",
        POP_SIZE,
        MAX_GENERATIONS,
        TOTAL_TIMESTEPS,
    )
    logger.info("  acceptance_tolerance (train): %.1f", TRAIN_TOLERANCE)
    logger.info(
        "  buffer_size: %d  batch_size: %d  target_update: %d",
        BUFFER_SIZE,
        BATCH_SIZE,
        TARGET_UPDATE_INTERVAL,
    )
    logger.info("=" * 60)

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=TRAIN_TOLERANCE,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=SEED,
        verbose=1,
    )

    callback = ThesisLoggingCallback(run_dir=run_dir, verbose=1)

    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    train_time = time.perf_counter() - t0
    logger.info(
        "DQN training complete in %.1fs (%.1f min)", train_time, train_time / 60
    )

    # Save model
    model_run = run_dir / "dqn_capstone_final.zip"
    model.save(str(model_run))
    logger.info("Model saved (run): %s", model_run)

    canonical_dir = PROJECT_ROOT / "output" / "models"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / "dqn_capstone_final.zip"
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
    """Build one evaluation CSV row."""
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
    for name in HARD_CONSTRAINT_NAMES:
        row[f"cv_{name}"] = info.get(f"cv_{name}", 0.0)
    for name in SOFT_CONSTRAINT_NAMES:
        row[f"cv_{name}"] = info.get(f"cv_{name}", 0.0)
    return row


def evaluate(model, run_dir: Path) -> Path:
    """200-gen strict evaluation and export CSV."""
    from src.rl.actions.vectorized_ops import ACTION_NAMES
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    logger.info("-" * 60)
    logger.info(
        "Deterministic DQN evaluation (%d gens, tolerance=%.1f)",
        EVAL_GENERATIONS,
        EVAL_TOLERANCE,
    )
    logger.info("-" * 60)

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED + 1000,
        acceptance_tolerance=EVAL_TOLERANCE,
    )

    obs, info = env.reset()
    rows: list[dict] = []
    rows.append(_build_eval_row(info, action_id=-1, action_name="init", reward=0.0))

    cumulative_reward = 0.0
    for gen in range(EVAL_GENERATIONS - 1):
        action, _ = model.predict(obs, deterministic=True)
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
                "  Gen %3d | act=%d (%s) | hard=%.0f soft=%.0f | feas=%.2f | r=%.4f",
                info["generation"],
                action,
                ACTION_NAMES.get(action, "?"),
                info["best_hard"],
                info["best_soft"],
                info["feasible_frac"],
                reward,
            )

        if terminated or truncated:
            break

    env.close()

    # Export to both run_dir and canonical baselines path
    csv_run = run_dir / "evaluation_trajectory_200.csv"
    csv_baselines = PROJECT_ROOT / "output" / "baselines" / "dqn_eval_200.csv"

    for csv_path in [csv_run, csv_baselines]:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("DQN eval CSV saved: %s (%d rows)", csv_path, len(rows))

    logger.info("Cumulative eval reward: %.4f", cumulative_reward)

    print("\n" + "=" * 80)
    print("  FINAL 10 LINES OF DQN evaluation_trajectory_200.csv")
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
        f"  DQN FINAL @ Gen {rows[-1]['generation']}: "
        f"Best_Hard = {rows[-1]['best_hard']:.1f}  |  "
        f"Best_Soft = {rows[-1]['best_soft']:.1f}"
    )
    print("=" * 80 + "\n")

    return csv_baselines


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_dqn" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DQN COMPETITOR RUN")
    logger.info("  Run dir: %s", run_dir)
    logger.info(
        "  Train: %d timesteps, tolerance=%.1f", TOTAL_TIMESTEPS, TRAIN_TOLERANCE
    )
    logger.info(
        "  Eval:  %d generations, tolerance=%.1f", EVAL_GENERATIONS, EVAL_TOLERANCE
    )
    logger.info("=" * 60)

    # 1. Train
    model = train(run_dir)

    # 2. Evaluate
    evaluate(model, run_dir)

    # 3. Summary
    logger.info("=" * 60)
    logger.info("DQN competitor run complete: %s", run_dir)
    logger.info("  Model   : dqn_capstone_final.zip")
    logger.info("  Baselines CSV: output/baselines/dqn_eval_200.csv")
    logger.info("=" * 60)

    # 4. Generate thesis plots
    from src.rl.training.plot_thesis_figures import generate_plots

    generate_plots(run_dir)


if __name__ == "__main__":
    main()
