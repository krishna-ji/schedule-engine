#!/usr/bin/env python3
r"""Phase 54 — Static Baseline Evaluation for Pipeline LLH Space.

Runs each of the 6 pipeline-configuration LLHs in isolation for 50
generations, repeated with 3 seeds for robustness.  Produces a single
``static_baselines.csv`` with per-generation, per-seed, per-action
trajectories for thesis comparison against the PPO adaptive policy.

Baselines
---------

=====  ==========================  ========================
  ID   Name                        Strategy
=====  ==========================  ========================
    0  ConservativeRepair          ``repair_batch(passes=3)``
    1  AggressiveRepair            ``repair_batch(passes=7)``
    2  MemeticEliteRepair          3 + 4 extra on worst 15%
    3  SoftFocusRepair             3 + time-compaction
    4  DestructiveConstructive     ruin 10% + 5-pass rebuild
    5  IntensifiedRepair           ``repair_batch(passes=5)``
=====  ==========================  ========================

Usage::

    python runs/rl_01_static_baselines.py

Outputs (in ``output/rl_phase54/``)::

    static_baselines.csv
"""

from __future__ import annotations

import csv
import logging
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
logger = logging.getLogger("rl_01_static_baselines")

# ======================================================================
# Configuration
# ======================================================================

POP_SIZE = 120
MAX_GENERATIONS = 50
PKL_PATH = ".cache/events_with_domains.pkl"
SEEDS = [42, 123, 7]  # 3 seeds for robustness

CSV_COLUMNS = [
    "seed",
    "action_id",
    "action_name",
    "generation",
    "best_hard",
    "best_soft",
    "mean_hard",
    "mean_soft",
    "feasible_frac",
    "delta_hard",
    "delta_soft",
    "step_time_s",
]


# ======================================================================
# Helpers
# ======================================================================


def _create_env(seed: int):
    """Create a fresh environment for baseline evaluation."""
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    return PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=seed,
    )


def run_static_baseline(action_id: int, action_name: str, seed: int) -> list[dict]:
    """Run one static baseline: apply a single action for all generations."""
    env = _create_env(seed)
    obs, info = env.reset()

    rows: list[dict] = []

    # Record initial state (generation 1 from reset)
    rows.append(
        {
            "seed": seed,
            "action_id": action_id,
            "action_name": action_name,
            "generation": 1,
            "best_hard": info["best_hard"],
            "best_soft": info["best_soft"],
            "mean_hard": info["mean_hard"],
            "mean_soft": info["mean_soft"],
            "feasible_frac": info["feasible_frac"],
            "delta_hard": 0.0,
            "delta_soft": 0.0,
            "step_time_s": 0.0,
        }
    )

    for gen in range(MAX_GENERATIONS - 1):  # reset already ran gen 1
        obs, reward, terminated, truncated, info = env.step(action_id)
        rows.append(
            {
                "seed": seed,
                "action_id": action_id,
                "action_name": action_name,
                "generation": info["generation"],
                "best_hard": info["best_hard"],
                "best_soft": info["best_soft"],
                "mean_hard": info["mean_hard"],
                "mean_soft": info["mean_soft"],
                "feasible_frac": info["feasible_frac"],
                "delta_hard": info.get("delta_hard", 0.0),
                "delta_soft": info.get("delta_soft", 0.0),
                "step_time_s": info.get("step_time_s", 0.0),
            }
        )
        if terminated or truncated:
            break

    env.close()
    return rows


# ======================================================================
# Main
# ======================================================================


def main() -> Path:
    """Run all 6 static baselines × 3 seeds and export CSV."""
    from src.rl.actions.vectorized_ops import ACTION_NAMES, NUM_ACTIONS

    out_dir = PROJECT_ROOT / "output" / "rl_phase54"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "static_baselines.csv"

    logger.info("=" * 65)
    logger.info(
        "Phase 54 — Static Baselines: %d actions × %d seeds", NUM_ACTIONS, len(SEEDS)
    )
    logger.info("  pop_size=%d  max_gen=%d", POP_SIZE, MAX_GENERATIONS)
    logger.info("  Output: %s", csv_path)
    logger.info("=" * 65)

    all_rows: list[dict] = []
    t0_total = time.perf_counter()

    for action_id in range(NUM_ACTIONS):
        action_name = ACTION_NAMES[action_id]
        for seed in SEEDS:
            t0 = time.perf_counter()
            rows = run_static_baseline(action_id, action_name, seed)
            dt = time.perf_counter() - t0

            final = rows[-1]
            logger.info(
                "  Action %d (%s) seed=%d | hard=%.0f soft=%.0f | %.1fs",
                action_id,
                action_name,
                seed,
                final["best_hard"],
                final["best_soft"],
                dt,
            )
            all_rows.extend(rows)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    total_time = time.perf_counter() - t0_total
    logger.info("-" * 65)
    logger.info(
        "Static baselines complete: %d rows in %.1fs", len(all_rows), total_time
    )

    # ---- Summary table ----
    print("\n")
    print("=" * 75)
    print("  STATIC BASELINE SUMMARY  (averaged across %d seeds)" % len(SEEDS))
    print("=" * 75)
    hdr = f"{'ID':>3}  {'Action':<28}  {'Final Hard':>10}  {'Final Soft':>10}  {'Best Hard':>10}"
    print(hdr)
    print("-" * len(hdr))

    for action_id in range(NUM_ACTIONS):
        action_name = ACTION_NAMES[action_id]
        # Get final-generation rows for this action across seeds
        finals = [
            r
            for r in all_rows
            if r["action_id"] == action_id and r["generation"] == MAX_GENERATIONS
        ]
        # Fallback: use last row per seed if generation doesn't match exactly
        if not finals:
            for seed in SEEDS:
                seed_rows = [
                    r
                    for r in all_rows
                    if r["action_id"] == action_id and r["seed"] == seed
                ]
                if seed_rows:
                    finals.append(seed_rows[-1])

        if finals:
            avg_hard = np.mean([r["best_hard"] for r in finals])
            avg_soft = np.mean([r["best_soft"] for r in finals])
            # Find best hard ever across all generations/seeds
            all_action = [r for r in all_rows if r["action_id"] == action_id]
            best_hard_ever = min(r["best_hard"] for r in all_action)
            print(
                f"{action_id:>3}  {action_name:<28}  {avg_hard:>10.1f}  {avg_soft:>10.1f}  {best_hard_ever:>10.1f}"
            )

    print("=" * 75)
    logger.info("CSV saved: %s", csv_path)
    return csv_path


if __name__ == "__main__":
    main()
