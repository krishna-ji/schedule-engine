#!/usr/bin/env python3
r"""Eval All Baselines -- PPO, Random, Round-Robin, UCB1.

Runs four separate 200-generation deterministic evaluations on
PymooHyperHeuristicEnv (max_generations=200, acceptance_tolerance=0.0)
and saves CSV files to output/baselines/.

Usage::

    python runs/eval_all_baselines.py

Outputs::

    output/baselines/ppo_eval_200.csv
    output/baselines/random_eval_200.csv
    output/baselines/round_robin_eval_200.csv
    output/baselines/ucb1_eval_200.csv
"""

from __future__ import annotations

import csv
import logging
import math
import sys
import time
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
logger = logging.getLogger("eval_all_baselines")

# ======================================================================
# Configuration
# ======================================================================

SEED = 42
POP_SIZE = 120
EVAL_GENERATIONS = 200
EVAL_TOLERANCE = 0.0
PKL_PATH = ".cache/events_with_domains.pkl"
NUM_ACTIONS = 8  # Discrete(8) action space

OUTPUT_DIR = PROJECT_ROOT / "output" / "baselines"
MODEL_PATH = PROJECT_ROOT / "output" / "models" / "ppo_capstone_final.zip"


# ======================================================================
# Shared evaluation harness
# ======================================================================


def _build_row(info: dict, action_id: int, action_name: str, reward: float) -> dict:
    """Build one CSV row with constraint breakdown."""
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


def run_evaluation(
    strategy_name: str,
    action_fn,
    csv_path: Path,
    *,
    model=None,
) -> list[dict]:
    """Run a single 200-gen evaluation with the given action selection function.

    Parameters
    ----------
    strategy_name : str
        Human-readable label (PPO, Random, Round-Robin, UCB1).
    action_fn : callable(obs, gen, env, model) -> int
        Returns the action to take at each step.
    csv_path : Path
        Output CSV path.
    model : optional
        SB3 model (only for PPO).

    Returns
    -------
    rows : list[dict]
    """
    from src.rl.actions.vectorized_ops import ACTION_NAMES
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    logger.info("=" * 60)
    logger.info(
        "Strategy: %s  |  %d generations  |  tolerance=%.1f",
        strategy_name,
        EVAL_GENERATIONS,
        EVAL_TOLERANCE,
    )
    logger.info("=" * 60)

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
    rows.append(_build_row(info, action_id=-1, action_name="init", reward=0.0))

    cumulative_reward = 0.0
    t0 = time.perf_counter()

    for gen in range(EVAL_GENERATIONS - 1):
        action = action_fn(obs, gen, env, model)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        rows.append(
            _build_row(
                info,
                action_id=action,
                action_name=ACTION_NAMES.get(action, f"action_{action}"),
                reward=reward,
            )
        )

        if gen % 20 == 0 or gen >= EVAL_GENERATIONS - 5:
            logger.info(
                "  [%s] Gen %3d | act=%d | hard=%.0f soft=%.0f | feas=%.2f | r=%.4f",
                strategy_name,
                info["generation"],
                action,
                info["best_hard"],
                info["best_soft"],
                info["feasible_frac"],
                reward,
            )

        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t0
    env.close()

    # Write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "[%s] CSV saved: %s (%d rows, %.1fs)",
        strategy_name,
        csv_path,
        len(rows),
        elapsed,
    )

    # Print summary
    final = rows[-1]
    print(f"\n{'='*60}")
    print(f"  {strategy_name} FINAL @ Gen {final['generation']}:")
    print(f"    Best_Hard = {final['best_hard']:.1f}")
    print(f"    Best_Soft = {final['best_soft']:.1f}")
    print(f"    Cumulative Reward = {cumulative_reward:.4f}")
    print(f"    Time = {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return rows


# ======================================================================
# Action Selection Strategies
# ======================================================================


def action_ppo(obs, gen, env, model):
    """PPO: predict from trained model."""
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def action_random(obs, gen, env, model):
    """Random: sample from action space."""
    return env.action_space.sample()


def action_round_robin(obs, gen, env, model):
    """Round-Robin: cycle through actions 0..7."""
    return gen % NUM_ACTIONS


class UCB1Bandit:
    """Standard UCB1 multi-armed bandit for action selection.

    At each step, selects the action maximizing:
        UCB1(a) = Q(a) + c * sqrt(ln(N) / N(a))

    where Q(a) is the average reward for action a,
    N is the total number of plays, and N(a) is the
    number of times action a has been played.
    """

    def __init__(self, n_actions: int, c: float = 1.414):
        self.n_actions = n_actions
        self.c = c
        self.counts = np.zeros(n_actions, dtype=np.float64)
        self.values = np.zeros(n_actions, dtype=np.float64)  # Q(a)
        self.total = 0

    def select(self) -> int:
        # Play each action once first (exploration phase)
        for a in range(self.n_actions):
            if self.counts[a] == 0:
                return a

        # UCB1 formula
        ucb_values = self.values + self.c * np.sqrt(np.log(self.total) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        self.total += 1
        # Incremental mean update
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n


# Global UCB1 instance (reset per run)
_ucb1: UCB1Bandit | None = None


def action_ucb1(obs, gen, env, model):
    """UCB1: multi-armed bandit action selection."""
    global _ucb1
    return _ucb1.select()


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    global _ucb1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Run 1: PPO ---------------------------------------------------
    ppo_model = None
    ppo_path = OUTPUT_DIR / "ppo_eval_200.csv"

    if MODEL_PATH.exists():
        from stable_baselines3 import PPO

        logger.info("Loading PPO model from %s", MODEL_PATH)
        ppo_model = PPO.load(str(MODEL_PATH))
        run_evaluation("PPO", action_ppo, ppo_path, model=ppo_model)
    else:
        # Try to find model in rl_capstone directories
        capstone_dirs = sorted(
            (PROJECT_ROOT / "output" / "rl_capstone").glob("*/ppo_capstone_final.zip"),
            reverse=True,
        )
        # Also try rl_vectorized directories
        vectorized_dirs = sorted(
            (PROJECT_ROOT / "output" / "rl_vectorized").glob("*/ppo_vectorized_hh.zip"),
            reverse=True,
        )
        found = None
        if capstone_dirs:
            found = capstone_dirs[0]
        elif vectorized_dirs:
            found = vectorized_dirs[0]

        if found:
            from stable_baselines3 import PPO

            logger.info("Loading PPO model from %s", found)
            try:
                ppo_model = PPO.load(str(found))
                run_evaluation("PPO", action_ppo, ppo_path, model=ppo_model)
            except Exception as e:
                logger.warning(
                    "Failed to load PPO model (%s). Skipping PPO eval. "
                    "Re-train with runs/rl_03_capstone_thesis.py",
                    e,
                )
        else:
            logger.warning(
                "No PPO model found at %s, rl_capstone/, or rl_vectorized/. Skipping PPO eval.",
                MODEL_PATH,
            )

    # ---- Run 2: Random ------------------------------------------------
    np.random.seed(SEED)
    run_evaluation(
        "Random",
        action_random,
        OUTPUT_DIR / "random_eval_200.csv",
    )

    # ---- Run 3: Round-Robin -------------------------------------------
    run_evaluation(
        "Round-Robin",
        action_round_robin,
        OUTPUT_DIR / "round_robin_eval_200.csv",
    )

    # ---- Run 4: UCB1 --------------------------------------------------
    _ucb1 = UCB1Bandit(n_actions=NUM_ACTIONS, c=1.414)

    def action_ucb1_with_update(obs, gen, env, model):
        """UCB1 with reward update after each step."""
        act = _ucb1.select()
        return act

    # For UCB1, we need a wrapper that also updates after each step
    # We'll use a custom evaluation that hooks into the reward
    from src.rl.actions.vectorized_ops import ACTION_NAMES
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    logger.info("=" * 60)
    logger.info(
        "Strategy: UCB1  |  %d generations  |  tolerance=%.1f",
        EVAL_GENERATIONS,
        EVAL_TOLERANCE,
    )
    logger.info("=" * 60)

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
    rows.append(_build_row(info, action_id=-1, action_name="init", reward=0.0))
    cumulative_reward = 0.0
    t0 = time.perf_counter()

    for gen in range(EVAL_GENERATIONS - 1):
        action = _ucb1.select()
        obs, reward, terminated, truncated, info = env.step(action)
        _ucb1.update(action, reward)
        cumulative_reward += reward

        rows.append(
            _build_row(
                info,
                action_id=action,
                action_name=ACTION_NAMES.get(action, f"action_{action}"),
                reward=reward,
            )
        )

        if gen % 20 == 0 or gen >= EVAL_GENERATIONS - 5:
            logger.info(
                "  [UCB1] Gen %3d | act=%d | hard=%.0f soft=%.0f | feas=%.2f | r=%.4f",
                info["generation"],
                action,
                info["best_hard"],
                info["best_soft"],
                info["feasible_frac"],
                reward,
            )

        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t0
    env.close()

    ucb1_csv = OUTPUT_DIR / "ucb1_eval_200.csv"
    fieldnames = list(rows[0].keys())
    with open(ucb1_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    final = rows[-1]
    logger.info("[UCB1] CSV saved: %s (%d rows, %.1fs)", ucb1_csv, len(rows), elapsed)
    print(f"\n{'='*60}")
    print(f"  UCB1 FINAL @ Gen {final['generation']}:")
    print(f"    Best_Hard = {final['best_hard']:.1f}")
    print(f"    Best_Soft = {final['best_soft']:.1f}")
    print(f"    Cumulative Reward = {cumulative_reward:.4f}")
    print(f"    Time = {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # ---- Summary --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ALL BASELINES COMPLETE")
    print("=" * 70)
    for csv_name in [
        "ppo_eval_200.csv",
        "random_eval_200.csv",
        "round_robin_eval_200.csv",
        "ucb1_eval_200.csv",
    ]:
        p = OUTPUT_DIR / csv_name
        if p.exists():
            print(f"  [OK] {p}")
        else:
            print(f"  [--] {p} (not generated)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
