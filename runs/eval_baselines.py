#!/usr/bin/env python3
r"""Baseline Generation Suite — 200-Gen Evaluation Trajectories.

Generates three baseline evaluation CSVs using the full Elite 8
micro-memetic action space (no learning, just heuristic policies):

  1. **Random**:      action = env.action_space.sample()
  2. **Round-Robin**:  action = gen % 8
  3. **UCB1**:         action = argmax[Q_a + c * sqrt(ln N / n_a)]

Each baseline runs the PymooHyperHeuristicEnv for 200 generations
with acceptance_tolerance=0.0 (strict hill-climbing) and saves
per-generation metrics to CSV.

Usage::

    python runs/eval_baselines.py

Outputs (in ``output/baselines/``)::

    random_eval_200.csv
    round_robin_eval_200.csv
    ucb1_eval_200.csv
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
logger = logging.getLogger("eval_baselines")

# ======================================================================
# Configuration
# ======================================================================

SEED = 42
POP_SIZE = 120  # Match Titan configuration
MAX_GENERATIONS = 200  # Full evaluation trajectory
ACCEPTANCE_TOLERANCE = 0.0  # Strict: never accept worse hard
PKL_PATH = ".cache/events_with_domains.pkl"
NUM_ACTIONS = 8

# UCB1 exploration constant
UCB1_C = 1.414  # sqrt(2)

# CSV columns
CSV_COLUMNS = [
    "generation",
    "action_id",
    "action_name",
    "best_hard",
    "best_soft",
    "mean_hard",
    "mean_soft",
    "feasible_frac",
    "reward",
    "rejected",
    "delta_hard",
    "delta_soft",
    "cv_CTE",
    "cv_FTE",
    "cv_SRE",
    "cv_FPC",
    "cv_FFC",
    "cv_FCA",
    "cv_CQF",
    "cv_ICTD",
    "cv_CSC",
    "cv_FSC",
    "cv_MIP",
    "cv_SSCP",
]

# Hard / soft constraint names from the environment
HARD_NAMES = ["CTE", "FTE", "SRE", "FPC", "FFC", "FCA", "CQF", "ICTD"]
SOFT_NAMES = ["CSC", "FSC", "MIP", "SSCP"]


# ======================================================================
# Evaluation Functions
# ======================================================================


def _create_env(seed: int):
    """Create a fresh environment for evaluation."""
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    return PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=seed,
        acceptance_tolerance=ACCEPTANCE_TOLERANCE,
    )


def _info_to_row(
    gen: int,
    action_id: int,
    action_name: str,
    info: dict,
    reward: float,
    rejected: bool,
    delta_hard: float,
    delta_soft: float,
) -> list:
    """Convert environment info dict to CSV row."""
    row = [
        gen,
        action_id,
        action_name,
        info.get("best_hard", np.inf),
        info.get("best_soft", np.inf),
        info.get("mean_hard", np.inf),
        info.get("mean_soft", np.inf),
        info.get("feasible_frac", 0.0),
        reward,
        rejected,
        delta_hard,
        delta_soft,
    ]
    # Hard constraint breakdowns
    for name in HARD_NAMES:
        row.append(info.get(f"cv_{name}", 0.0))
    # Soft constraint breakdowns
    for name in SOFT_NAMES:
        row.append(info.get(f"cv_{name}", 0.0))
    return row


def run_evaluation(
    policy_name: str,
    policy_fn,
    output_path: Path,
    seed: int = SEED,
):
    """Run a 200-generation evaluation with the given policy function.

    Parameters
    ----------
    policy_name : str
        Human-readable name for logging.
    policy_fn : callable(gen, env, obs, info, state) -> (action, state)
        Policy function returning (action_id, updated_state).
    output_path : Path
        CSV output path.
    seed : int
        Random seed.
    """
    from src.rl.actions.vectorized_ops import ACTION_NAMES

    env = _create_env(seed)
    obs, info = env.reset()

    logger.info("Running %s evaluation (200 generations)...", policy_name)
    t0 = time.perf_counter()

    rows = []
    # Initial state row
    rows.append(_info_to_row(1, -1, "init", info, 0.0, False, 0.0, 0.0))

    policy_state = None  # For stateful policies like UCB1

    for gen in range(2, MAX_GENERATIONS + 2):  # gen 2..201
        action, policy_state = policy_fn(gen, env, obs, info, policy_state)
        action_name = ACTION_NAMES.get(action, f"action_{action}")

        prev_hard = info.get("best_hard", np.inf)
        prev_soft = info.get("best_soft", np.inf)

        obs, reward, terminated, truncated, info = env.step(action)

        delta_hard = info.get("best_hard", np.inf) - prev_hard
        delta_soft = info.get("best_soft", np.inf) - prev_soft
        rejected = info.get("rejected", False)

        rows.append(
            _info_to_row(
                gen,
                action,
                action_name,
                info,
                reward,
                rejected,
                delta_hard,
                delta_soft,
            )
        )

        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t0
    env.close()

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)

    final_hard = rows[-1][3]
    final_soft = rows[-1][4]
    logger.info(
        "  %s: %d gens in %.1fs | final hard=%.1f soft=%.1f | saved: %s",
        policy_name,
        len(rows),
        elapsed,
        final_hard,
        final_soft,
        output_path,
    )
    return rows


# ======================================================================
# Policy Functions
# ======================================================================


def policy_random(gen, env, obs, info, state):
    """Random action selection."""
    masks = env.action_masks()
    valid = np.where(masks)[0]
    action = np.random.choice(valid)
    return int(action), state


def policy_round_robin(gen, env, obs, info, state):
    """Round-robin cycling through all 8 actions."""
    action = (gen - 2) % NUM_ACTIONS
    # Respect mask: if masked, skip to next valid
    masks = env.action_masks()
    if not masks[action]:
        valid = np.where(masks)[0]
        action = int(valid[gen % len(valid)])
    return int(action), state


def policy_ucb1(gen, env, obs, info, state):
    """UCB1 bandit policy: balance exploitation vs exploration.

    UCB1 formula: action = argmax[ Q_a + c * sqrt(ln(N) / n_a) ]
    where Q_a = average reward for action a, N = total pulls, n_a = pulls of a.
    """
    if state is None:
        state = {
            "counts": np.zeros(NUM_ACTIONS, dtype=np.float64),
            "rewards": np.zeros(NUM_ACTIONS, dtype=np.float64),
            "total": 0,
            "prev_action": None,
        }

    masks = env.action_masks()

    # Update stats from previous action
    if state["prev_action"] is not None:
        a = state["prev_action"]
        # Use improvement in hard constraint as reward signal
        delta_hard = info.get("delta_hard", 0.0)
        r = -delta_hard if delta_hard is not None else 0.0  # Positive = improvement
        state["counts"][a] += 1
        state["rewards"][a] += r

    state["total"] += 1

    # UCB1 selection
    ucb_values = np.full(NUM_ACTIONS, -np.inf)
    for a in range(NUM_ACTIONS):
        if not masks[a]:
            continue
        if state["counts"][a] == 0:
            ucb_values[a] = np.inf  # Explore unvisited actions first
        else:
            q_a = state["rewards"][a] / state["counts"][a]
            exploration = UCB1_C * math.sqrt(
                math.log(state["total"]) / state["counts"][a]
            )
            ucb_values[a] = q_a + exploration

    action = int(np.argmax(ucb_values))
    state["prev_action"] = action
    return action, state


# ======================================================================
# Main
# ======================================================================


def main():
    baselines_dir = PROJECT_ROOT / "output" / "baselines"
    np.random.seed(SEED)

    logger.info("=" * 60)
    logger.info("  BASELINE GENERATION SUITE")
    logger.info("  200-gen trajectories with Elite 8 micro-memetic actions")
    logger.info(
        "  pop_size=%d  acceptance_tolerance=%.1f", POP_SIZE, ACCEPTANCE_TOLERANCE
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()

    # 1. Random baseline
    run_evaluation(
        "Random",
        policy_random,
        baselines_dir / "random_eval_200.csv",
        seed=SEED,
    )

    # 2. Round-robin baseline
    run_evaluation(
        "Round-Robin",
        policy_round_robin,
        baselines_dir / "round_robin_eval_200.csv",
        seed=SEED,
    )

    # 3. UCB1 baseline
    run_evaluation(
        "UCB1",
        policy_ucb1,
        baselines_dir / "ucb1_eval_200.csv",
        seed=SEED,
    )

    total_time = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info(
        "  ALL BASELINES COMPLETE in %.1fs (%.1f min)", total_time, total_time / 60
    )
    logger.info("  Output: %s", baselines_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
