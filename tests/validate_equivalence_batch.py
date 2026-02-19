#!/usr/bin/env python3
"""Validate equivalence between original fast_evaluator and batch fast_evaluator.

Compares per-constraint violation counts for N random individuals.
Must produce 0 mismatches.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.fast_evaluator import fast_evaluate_hard
from src.pipeline.fast_evaluator_batch import (
    HARD_CONSTRAINT_NAMES,
    BatchEvalData,
    fast_evaluate_hard_batch,
    fast_evaluate_hard_single,
    prepare_batch_data,
)
from src.pipeline.repair_operator import SchedulingRepair

PKL_PATH = "events_with_domains.pkl"


def generate_individuals(pkl_data: dict, n: int, seed: int = 42) -> np.ndarray:
    """Generate N random-but-domain-respecting individuals."""
    repairer = SchedulingRepair(PKL_PATH)
    E = len(pkl_data["events"])
    X = np.zeros((n, 3 * E), dtype=int)

    for i in range(n):
        rng = np.random.default_rng(seed + i)
        chrom = repairer.construct_feasible(rng)
        # Optionally perturb some to get diversity (half repaired, half raw)
        if i % 2 == 0:
            chrom = repairer.repair(chrom)
        X[i] = chrom

    return X


def evaluate_original(x: np.ndarray, pkl_data: dict) -> dict[str, int]:
    """Evaluate a single individual using the original fast_evaluate_hard."""
    inst = x[0::3]
    room = x[1::3]
    time = x[2::3]
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


def main():
    N = 50
    print(f"Validating equivalence: original vs batch evaluator on {N} individuals")
    print("=" * 70)

    with open(PKL_PATH, "rb") as f:
        pkl_data = pickle.load(f)

    batch_data = prepare_batch_data(pkl_data)

    print(f"Generating {N} individuals (constructive + repair mix)...")
    t0 = time.time()
    X = generate_individuals(pkl_data, N)
    print(f"  Generated in {time.time() - t0:.1f}s")

    # Evaluate with original (one by one)
    print("Evaluating with original evaluator...")
    t0 = time.time()
    original_results = []
    for i in range(N):
        result = evaluate_original(X[i], pkl_data)
        original_results.append(result)
    t_orig = time.time() - t0
    print(f"  Original: {t_orig:.3f}s ({t_orig/N*1000:.2f} ms/ind)")

    # Evaluate with batch evaluator
    print("Evaluating with batch evaluator...")
    t0 = time.time()
    G = fast_evaluate_hard_batch(X, batch_data)
    t_batch = time.time() - t0
    print(f"  Batch: {t_batch:.3f}s ({t_batch/N*1000:.2f} ms/ind)")

    # Compare
    mismatches = 0
    mismatch_details = []
    for i in range(N):
        orig = original_results[i]
        for j, cname in enumerate(HARD_CONSTRAINT_NAMES):
            orig_val = orig.get(cname, 0)
            batch_val = int(G[i, j])
            if orig_val != batch_val:
                mismatches += 1
                if len(mismatch_details) < 3:
                    mismatch_details.append(
                        {
                            "individual": i,
                            "constraint": cname,
                            "original": orig_val,
                            "batch": batch_val,
                        }
                    )

    print(f"\n{'=' * 70}")
    print(
        f"RESULTS: {mismatches} mismatches out of {N * len(HARD_CONSTRAINT_NAMES)} comparisons"
    )

    if mismatches > 0:
        print(f"\nFirst {min(3, len(mismatch_details))} mismatches:")
        for md in mismatch_details:
            print(
                f"  Individual {md['individual']}, {md['constraint']}: "
                f"original={md['original']} batch={md['batch']}"
            )

        # Show full breakdown for first mismatched individual
        if mismatch_details:
            idx = mismatch_details[0]["individual"]
            print(f"\nFull breakdown for individual {idx}:")
            orig = original_results[idx]
            print(f"  {'Constraint':<35} {'Original':>10} {'Batch':>10} {'Match':>6}")
            for j, cname in enumerate(HARD_CONSTRAINT_NAMES):
                ov = orig.get(cname, 0)
                bv = int(G[idx, j])
                match = "OK" if ov == bv else "FAIL"
                print(f"  {cname:<35} {ov:>10} {bv:>10} {match:>6}")
    else:
        print("PASS - All constraint values match exactly")

    # Timing comparison
    speedup = t_orig / t_batch if t_batch > 0 else float("inf")
    print(
        f"\nTiming: original={t_orig*1000:.1f}ms batch={t_batch*1000:.1f}ms "
        f"speedup={speedup:.1f}x"
    )

    # Also validate single-individual API
    print("\nValidating fast_evaluate_hard_single API...")
    for i in range(min(5, N)):
        single_result = fast_evaluate_hard_single(X[i], batch_data)
        orig = original_results[i]
        for cname in HARD_CONSTRAINT_NAMES:
            assert single_result[cname] == orig.get(cname, 0), (
                f"Single API mismatch at ind={i}, {cname}: "
                f"{single_result[cname]} != {orig.get(cname, 0)}"
            )
    print("  Single API: PASS")

    return mismatches == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
