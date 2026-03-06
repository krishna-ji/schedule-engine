#!/usr/bin/env python3
r"""Phase 49 — Heuristic Payload Isolation Test.

Stress-tests each of the Elite 8 low-level heuristics in isolation
to empirically measure their individual constraint-resolution power.

For each action $a \in \{0,\dots,7\}$:
  1. Reset environment (fresh random population, pop=120)
  2. Execute action $a$ for 200 consecutive steps
  3. Record initial vs final hard constraint score

This answers: "Can heuristic $a$ alone drive Hard→0?"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.actions.vectorized_ops import ACTION_NAMES, NUM_ACTIONS
from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

N_STEPS = 200
POP_SIZE = 120
MAX_GENS = 200
SEED = 42


def main():
    env = PymooHyperHeuristicEnv(
        pkl_path=".cache/events_with_domains.pkl",
        max_generations=MAX_GENS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=0.0,
    )

    results = []

    for action_id in range(NUM_ACTIONS):
        action_name = ACTION_NAMES.get(action_id, f"action_{action_id}")

        obs, info = env.reset(seed=SEED)
        initial_hard = info.get("best_hard", np.inf)

        t0 = time.perf_counter()
        for step in range(N_STEPS):
            obs, reward, terminated, truncated, info = env.step(action_id)
            if terminated or truncated:
                break
        dt = time.perf_counter() - t0

        final_hard = info.get("best_hard", np.inf)
        delta = final_hard - initial_hard
        steps_done = step + 1

        results.append(
            {
                "id": action_id,
                "name": action_name,
                "initial": initial_hard,
                "final": final_hard,
                "delta": delta,
                "steps": steps_done,
                "time": dt,
            }
        )

        print(
            f"  Action {action_id} ({action_name}): {initial_hard:.0f} → {final_hard:.0f}  (Δ={delta:+.0f})  [{dt:.1f}s]"
        )

    env.close()

    # Final formatted table
    print()
    print("=" * 90)
    print("  HEURISTIC PAYLOAD ISOLATION TEST — RESULTS")
    print("=" * 90)
    print(
        f"  {'ID':>2s} | {'Action Name':<40s} | {'Init Hard':>9s} | {'Final Hard':>10s} | {'Delta':>8s}"
    )
    print("  " + "-" * 84)
    for r in results:
        print(
            f"  {r['id']:>2d} | {r['name']:<40s} | {r['initial']:>9.0f} | {r['final']:>10.0f} | {r['delta']:>+8.0f}"
        )
    print("=" * 90)

    # Summary
    best = min(results, key=lambda r: r["final"])
    print(f"\n  Best single-heuristic result: Action {best['id']} ({best['name']})")
    print(
        f"  Final Hard = {best['final']:.0f}  (from {best['initial']:.0f}, Δ={best['delta']:+.0f})"
    )
    print()


if __name__ == "__main__":
    main()
