#!/usr/bin/env python3
"""Test C: Offspring repair test.

Validates that the repair operator can handle realistic offspring perturbations:
- Start from a construct_feasible parent
- Perturb k events (simulating crossover/mutation)
- Repair
- Target: hard violations <= baseline + tolerance

This tests the REAL use-case: repair after crossover, not repair from random garbage.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.fast_evaluator import fast_evaluate_hard
from src.pipeline.repair_operator import SchedulingRepair

PKL_PATH = ".cache/events_with_domains.pkl"


def perturb_chromosome(
    chrom: np.ndarray,
    k: int,
    pkl_data: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb k random events in a chromosome (simulating crossover/mutation).

    For each perturbed event, randomly reassign instructor, room, and/or time
    to a value from the allowed domain (not necessarily the same as original).
    """
    out = chrom.copy()
    E = len(pkl_data["events"])
    events_to_perturb = rng.choice(E, size=min(k, E), replace=False)

    for e in events_to_perturb:
        ai = pkl_data["allowed_instructors"][e]
        ar = pkl_data["allowed_rooms"][e]
        at = pkl_data["allowed_starts"][e]

        # Randomly choose which genes to perturb (1-3 of inst/room/time)
        which = rng.choice([True, False], size=3)
        if not which.any():
            which[rng.integers(3)] = True  # At least one

        if which[0] and ai:
            out[3 * e + 0] = rng.choice(ai)
        if which[1] and ar:
            out[3 * e + 1] = rng.choice(ar)
        if which[2] and at:
            out[3 * e + 2] = rng.choice(at)

    return out


def evaluate_hard(chrom: np.ndarray, pkl_data: dict) -> dict[str, int]:
    """Evaluate hard constraints for a chromosome."""
    inst, room, time = chrom[0::3], chrom[1::3], chrom[2::3]
    return fast_evaluate_hard(
        pkl_data["events"],
        inst,
        room,
        time,
        pkl_data["allowed_instructors"],
        pkl_data["allowed_rooms"],
        pkl_data["instructor_available_quanta"],
        pkl_data["room_available_quanta"],
    )


def run_offspring_repair_test(
    k_values: list[int] | None = None,
    n_trials: int = 10,
    tolerance_factor: float = 1.5,
) -> dict[int, dict]:
    """Run offspring repair test for different perturbation sizes.

    Args:
        k_values: List of perturbation sizes to test.
        n_trials: Number of trials per k value.
        tolerance_factor: Repair result must be <= baseline * factor.

    Returns:
        Dict mapping k -> results dict.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    with open(PKL_PATH, "rb") as f:
        pkl_data = pickle.load(f)

    repairer = SchedulingRepair(PKL_PATH)

    # Establish baseline: construct + repair
    baselines = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        parent = repairer.construct_feasible(rng)
        repaired_parent = repairer.repair(parent)
        h = evaluate_hard(repaired_parent, pkl_data)
        baselines.append(sum(h.values()))
    baseline_avg = sum(baselines) / len(baselines)
    print(
        f"Baseline (construct+repair): avg={baseline_avg:.1f} min={min(baselines)} max={max(baselines)}"
    )

    results = {}
    for k in k_values:
        print(f"\n--- k={k} perturbations ---")
        pre_repair_scores = []
        post_repair_scores = []
        pass_count = 0
        threshold = max(baseline_avg * tolerance_factor, baseline_avg + 20)

        for trial in range(n_trials):
            rng = np.random.default_rng(42 + trial)
            # Construct a good parent
            parent = repairer.construct_feasible(rng)
            repaired_parent = repairer.repair(parent)
            parent_score = sum(evaluate_hard(repaired_parent, pkl_data).values())

            # Perturb to simulate offspring
            offspring = perturb_chromosome(repaired_parent, k, pkl_data, rng)
            pre_h = evaluate_hard(offspring, pkl_data)
            pre_score = sum(pre_h.values())
            pre_repair_scores.append(pre_score)

            # Repair offspring
            repaired = repairer.repair(offspring)
            post_h = evaluate_hard(repaired, pkl_data)
            post_score = sum(post_h.values())
            post_repair_scores.append(post_score)

            if post_score <= threshold:
                pass_count += 1

            if trial < 3:
                print(
                    f"  trial={trial}: parent={parent_score} "
                    f"perturbed={pre_score} -> repaired={post_score} "
                    f"{'PASS' if post_score <= threshold else 'FAIL'}"
                )

        avg_pre = sum(pre_repair_scores) / len(pre_repair_scores)
        avg_post = sum(post_repair_scores) / len(post_repair_scores)
        pass_rate = pass_count / n_trials
        reduction = (1 - avg_post / avg_pre) * 100 if avg_pre > 0 else 0

        results[k] = {
            "avg_pre": avg_pre,
            "avg_post": avg_post,
            "min_post": min(post_repair_scores),
            "max_post": max(post_repair_scores),
            "reduction_pct": reduction,
            "pass_rate": pass_rate,
            "threshold": threshold,
        }

        print(
            f"  Summary: pre={avg_pre:.1f} -> post={avg_post:.1f} "
            f"(reduction={reduction:.0f}%)  "
            f"pass_rate={pass_rate * 100:.0f}% (threshold={threshold:.0f})"
        )

    return results


def main():
    print("=" * 70)
    print("TEST C: OFFSPRING REPAIR TEST")
    print("=" * 70)
    results = run_offspring_repair_test()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for k, r in sorted(results.items()):
        status = "PASS" if r["pass_rate"] >= 0.8 else "FAIL"
        if r["pass_rate"] < 0.8:
            all_pass = False
        print(
            f"  k={k:3d}: post_avg={r['avg_post']:.0f} "
            f"reduction={r['reduction_pct']:.0f}% "
            f"pass_rate={r['pass_rate'] * 100:.0f}% [{status}]"
        )

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  OVERALL: {overall}")
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
