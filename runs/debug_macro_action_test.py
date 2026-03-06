#!/usr/bin/env python3
r"""Phase 51 — Meta-Heuristic Macro Action Benchmark.

Tests the upgraded Discrete(8) action space with LNS Ruin & Recreate
(Action 2) and Kempe Chain Interchange (Action 5) to verify they
can shatter the 963 hard constraint barrier.

Runs each action in isolation for 200 steps (identical to the
Phase 49 isolation test), then runs all 8 actions in round-robin
for 200 steps and random policy for 200 steps.
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


def run_isolation_test(env, action_id, action_name):
    """Run a single action for N_STEPS and return results."""
    obs, info = env.reset(seed=SEED)
    initial_hard = info.get("best_hard", np.inf)

    t0 = time.perf_counter()
    best_hard_ever = initial_hard
    for step in range(N_STEPS):
        obs, reward, terminated, truncated, info = env.step(action_id)
        h = info.get("best_hard", np.inf)
        best_hard_ever = min(best_hard_ever, h)
        if terminated or truncated:
            break
    dt = time.perf_counter() - t0

    final_hard = info.get("best_hard", np.inf)
    return {
        "id": action_id,
        "name": action_name,
        "initial": initial_hard,
        "final": final_hard,
        "best_ever": best_hard_ever,
        "delta": final_hard - initial_hard,
        "steps": step + 1,
        "time": dt,
        "fps": (step + 1) / dt,
    }


def run_policy_test(env, name, policy_fn, n_steps=N_STEPS):
    """Run a policy function for n_steps."""
    obs, info = env.reset(seed=SEED)
    initial_hard = info.get("best_hard", np.inf)

    t0 = time.perf_counter()
    best_hard_ever = initial_hard
    for step in range(n_steps):
        action = policy_fn(step, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        h = info.get("best_hard", np.inf)
        best_hard_ever = min(best_hard_ever, h)
        if terminated or truncated:
            break
    dt = time.perf_counter() - t0

    final_hard = info.get("best_hard", np.inf)
    return {
        "name": name,
        "initial": initial_hard,
        "final": final_hard,
        "best_ever": best_hard_ever,
        "delta": final_hard - initial_hard,
        "steps": step + 1,
        "time": dt,
        "fps": (step + 1) / dt,
    }


def main():
    env = PymooHyperHeuristicEnv(
        pkl_path=".cache/events_with_domains.pkl",
        max_generations=MAX_GENS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=0.0,
    )

    # ===== SECTION 1: Isolation Tests =====
    print()
    print("=" * 95)
    print("  SECTION 1: INDIVIDUAL ACTION ISOLATION (200 steps each)")
    print("=" * 95)

    isolation_results = []
    for action_id in range(NUM_ACTIONS):
        action_name = ACTION_NAMES.get(action_id, f"action_{action_id}")
        print(f"  Testing Action {action_id} ({action_name})...", end="", flush=True)
        result = run_isolation_test(env, action_id, action_name)
        isolation_results.append(result)
        print(
            f" {result['initial']:.0f} → {result['final']:.0f} (Δ={result['delta']:+.0f}) [{result['time']:.1f}s, {result['fps']:.1f} FPS]"
        )

    print()
    print(
        f"  {'ID':>2s} | {'Action Name':<35s} | {'Init':>6s} | {'Final':>6s} | {'Best':>6s} | {'Delta':>7s} | {'FPS':>5s}"
    )
    print("  " + "-" * 87)
    for r in isolation_results:
        marker = (
            " ★"
            if r["name"] in ("large_neighborhood_search", "kempe_chain_interchange")
            else ""
        )
        print(
            f"  {r['id']:>2d} | {r['name']:<35s} | {r['initial']:>6.0f} | {r['final']:>6.0f} | {r['best_ever']:>6.0f} | {r['delta']:>+7.0f} | {r['fps']:>5.1f}{marker}"
        )

    # ===== SECTION 2: Policy Tests =====
    print()
    print("=" * 95)
    print("  SECTION 2: POLICY COMPARISON (200 steps each)")
    print("=" * 95)

    rng = np.random.default_rng(SEED)
    policy_results = []

    # Round-Robin
    print("  Testing Round-Robin...", end="", flush=True)
    result = run_policy_test(
        env, "Round-Robin (all 8)", lambda s, o, i: s % NUM_ACTIONS
    )
    policy_results.append(result)
    print(
        f" {result['initial']:.0f} → {result['final']:.0f} (Δ={result['delta']:+.0f}) [{result['time']:.1f}s]"
    )

    # Random
    print("  Testing Random...", end="", flush=True)
    result = run_policy_test(
        env, "Random (all 8)", lambda s, o, i: int(rng.integers(0, NUM_ACTIONS))
    )
    policy_results.append(result)
    print(
        f" {result['initial']:.0f} → {result['final']:.0f} (Δ={result['delta']:+.0f}) [{result['time']:.1f}s]"
    )

    # LNS-heavy (70% LNS, 30% other repairs)
    def lns_heavy_policy(step, obs, info):
        r = rng.random()
        if r < 0.4:
            return 2  # LNS
        if r < 0.6:
            return 5  # Kempe
        if r < 0.75:
            return 0  # Spatial repair
        if r < 0.85:
            return 1  # Faculty repair
        if r < 0.95:
            return 4  # Universal feasibility
        return 6  # Spatial perturb

    print("  Testing LNS+Kempe Heavy...", end="", flush=True)
    result = run_policy_test(env, "LNS+Kempe Heavy (40/20)", lns_heavy_policy)
    policy_results.append(result)
    print(
        f" {result['initial']:.0f} → {result['final']:.0f} (Δ={result['delta']:+.0f}) [{result['time']:.1f}s]"
    )

    # Meta-Heuristic Only (alternating LNS and Kempe)
    print("  Testing Meta-Heuristic Only...", end="", flush=True)
    result = run_policy_test(
        env, "Meta-Only (LNS↔Kempe)", lambda s, o, i: 2 if s % 2 == 0 else 5
    )
    policy_results.append(result)
    print(
        f" {result['initial']:.0f} → {result['final']:.0f} (Δ={result['delta']:+.0f}) [{result['time']:.1f}s]"
    )

    print()
    print(
        f"  {'Policy':<35s} | {'Init':>6s} | {'Final':>6s} | {'Best':>6s} | {'Delta':>7s} | {'Time':>6s}"
    )
    print("  " + "-" * 78)
    for r in policy_results:
        print(
            f"  {r['name']:<35s} | {r['initial']:>6.0f} | {r['final']:>6.0f} | {r['best_ever']:>6.0f} | {r['delta']:>+7.0f} | {r['time']:>5.1f}s"
        )

    # ===== SECTION 3: Verdict =====
    print()
    print("=" * 95)
    print("  VERDICT: META-HEURISTIC OVERHAUL ASSESSMENT")
    print("=" * 95)

    old_barrier = 963  # Previous best single-heuristic result
    lns_result = next(
        r for r in isolation_results if r["name"] == "large_neighborhood_search"
    )
    kempe_result = next(
        r for r in isolation_results if r["name"] == "kempe_chain_interchange"
    )

    print(f"  Old Barrier (Phase 49)     : {old_barrier}")
    print(
        f"  LNS Ruin & Recreate (solo) : {lns_result['best_ever']:.0f}  (Δ from barrier: {lns_result['best_ever'] - old_barrier:+.0f})"
    )
    print(
        f"  Kempe Chain (solo)         : {kempe_result['best_ever']:.0f}  (Δ from barrier: {kempe_result['best_ever'] - old_barrier:+.0f})"
    )

    best_policy = min(policy_results, key=lambda r: r["best_ever"])
    print(
        f"  Best Policy Combo          : {best_policy['name']} → {best_policy['best_ever']:.0f}"
    )

    overall_best = min(
        min(r["best_ever"] for r in isolation_results),
        min(r["best_ever"] for r in policy_results),
    )
    print(f"  Overall Best Achieved      : {overall_best:.0f}")

    if overall_best < old_barrier:
        print(
            f"\n  ★ BARRIER SHATTERED! {old_barrier} → {overall_best:.0f} (improvement: {old_barrier - overall_best:.0f})"
        )
    else:
        print(
            f"\n  ✗ Barrier NOT broken (best: {overall_best:.0f}, target: < {old_barrier})"
        )

    print("=" * 95)

    env.close()


if __name__ == "__main__":
    main()
