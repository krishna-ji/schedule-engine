#!/usr/bin/env python3
"""Benchmark: evaluator and repair before/after bitset acceleration.

Measures:
1. Evaluator: per-individual time (original vs batch)
2. Repair: per-individual time (original vs bitset-accelerated)
3. Pymoo _evaluate: full population batch

Reports mean / median / p95 in ms.
Saves results to results/bench_eval.json.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

PKL_PATH = "events_with_domains.pkl"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def make_random_population(data, N, rng):
    E = len(data["events"])
    X = np.zeros((N, 3 * E), dtype=int)
    for n in range(N):
        for e in range(E):
            ai = data["allowed_instructors"][e]
            ar = data["allowed_rooms"][e]
            at = data["allowed_starts"][e]
            X[n, 3 * e] = rng.choice(ai) if ai else 0
            X[n, 3 * e + 1] = rng.choice(ar) if ar else 0
            X[n, 3 * e + 2] = rng.choice(at) if at else 0
    return X


def percentile(arr, p):
    return float(np.percentile(arr, p))


# ------------------------------------------------------------------
# Benchmark: Evaluator
# ------------------------------------------------------------------


def bench_evaluator_original(data, X):
    """Benchmark original per-individual evaluator."""
    from src.pipeline.fast_evaluator import fast_evaluate_hard

    events = data["events"]
    ai = data["allowed_instructors"]
    ar = data["allowed_rooms"]
    ia = data["instructor_available_quanta"]
    ra = data["room_available_quanta"]

    N = X.shape[0]
    times = []
    for i in range(N):
        xi = X[i].astype(int)
        inst, room, tm = xi[0::3], xi[1::3], xi[2::3]
        t0 = time.perf_counter()
        fast_evaluate_hard(events, inst, room, tm, ai, ar, ia, ra)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.array(times)


def bench_evaluator_batch(data, X):
    """Benchmark batch evaluator (full population at once)."""
    from src.pipeline.fast_evaluator_batch import (
        fast_evaluate_hard_batch,
        prepare_batch_data,
    )

    bd = prepare_batch_data(data)
    N = X.shape[0]

    # Warm-up
    fast_evaluate_hard_batch(X[:1], bd)

    t0 = time.perf_counter()
    fast_evaluate_hard_batch(X, bd)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    per_ind_ms = total_ms / N
    return total_ms, per_ind_ms


# ------------------------------------------------------------------
# Benchmark: Repair
# ------------------------------------------------------------------


def bench_repair_original(data, X, n_repair):
    """Benchmark original SchedulingRepair."""
    from src.pipeline.repair_operator import SchedulingRepair

    rep = SchedulingRepair(PKL_PATH)
    times = []
    for i in range(n_repair):
        xi = X[i].copy()
        t0 = time.perf_counter()
        rep.repair(xi)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.array(times)


def bench_repair_bitset(data, X, n_repair):
    """Benchmark BitsetSchedulingRepair."""
    from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

    rep = BitsetSchedulingRepair(PKL_PATH)
    times = []
    for i in range(n_repair):
        xi = X[i].copy()
        t0 = time.perf_counter()
        rep.repair(xi)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.array(times)


# ------------------------------------------------------------------
# Benchmark: Pymoo _evaluate
# ------------------------------------------------------------------


def bench_pymoo_evaluate(data, X):
    """Benchmark pymoo SchedulingProblem._evaluate (batch)."""
    from src.pipeline.scheduling_problem import SchedulingProblem

    prob = SchedulingProblem(PKL_PATH)
    out = {}

    # Warm-up
    prob._evaluate(X[:1], out)

    t0 = time.perf_counter()
    prob._evaluate(X, out)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    return total_ms, total_ms / X.shape[0]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def stats(arr):
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(percentile(arr, 95), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def main():
    print("=" * 60)
    print("BENCHMARK: Evaluator & Repair Acceleration")
    print("=" * 60)

    data = load_data()
    E = len(data["events"])
    rng = np.random.default_rng(42)

    N_EVAL = 200
    N_REPAIR = 20

    X_eval = make_random_population(data, N_EVAL, rng)
    X_repair = make_random_population(data, N_REPAIR, rng)

    results = {"n_events": E, "n_eval": N_EVAL, "n_repair": N_REPAIR}

    # --- Evaluator benchmarks ---
    print(f"\n--- Evaluator (N={N_EVAL}) ---")

    t_orig = bench_evaluator_original(data, X_eval)
    s_orig = stats(t_orig)
    results["evaluator_original"] = s_orig
    print(
        f"  Original:  mean={s_orig['mean_ms']:.3f} ms/ind  "
        f"median={s_orig['median_ms']:.3f}  p95={s_orig['p95_ms']:.3f}"
    )

    total_batch, per_batch = bench_evaluator_batch(data, X_eval)
    results["evaluator_batch"] = {
        "total_ms": round(total_batch, 3),
        "per_ind_ms": round(per_batch, 3),
    }
    print(f"  Batch:     total={total_batch:.1f} ms  per_ind={per_batch:.3f} ms/ind")

    speedup_eval = s_orig["mean_ms"] / per_batch if per_batch > 0 else float("inf")
    results["evaluator_speedup"] = round(speedup_eval, 2)
    print(f"  Speedup:   {speedup_eval:.2f}x")

    # --- Repair benchmarks ---
    print(f"\n--- Repair (N={N_REPAIR}) ---")

    t_repair_orig = bench_repair_original(data, X_repair, N_REPAIR)
    s_repair_orig = stats(t_repair_orig)
    results["repair_original"] = s_repair_orig
    print(
        f"  Original:  mean={s_repair_orig['mean_ms']:.1f} ms/ind  "
        f"median={s_repair_orig['median_ms']:.1f}  p95={s_repair_orig['p95_ms']:.1f}"
    )

    t_repair_bs = bench_repair_bitset(data, X_repair, N_REPAIR)
    s_repair_bs = stats(t_repair_bs)
    results["repair_bitset"] = s_repair_bs
    print(
        f"  Bitset:    mean={s_repair_bs['mean_ms']:.1f} ms/ind  "
        f"median={s_repair_bs['median_ms']:.1f}  p95={s_repair_bs['p95_ms']:.1f}"
    )

    speedup_repair = (
        s_repair_orig["mean_ms"] / s_repair_bs["mean_ms"]
        if s_repair_bs["mean_ms"] > 0
        else float("inf")
    )
    results["repair_speedup"] = round(speedup_repair, 2)
    print(f"  Speedup:   {speedup_repair:.2f}x")

    # --- Pymoo _evaluate benchmark ---
    print(f"\n--- Pymoo _evaluate (N={N_EVAL}) ---")

    total_pymoo, per_pymoo = bench_pymoo_evaluate(data, X_eval)
    results["pymoo_evaluate"] = {
        "total_ms": round(total_pymoo, 3),
        "per_ind_ms": round(per_pymoo, 3),
    }
    print(f"  Total: {total_pymoo:.1f} ms  per_ind: {per_pymoo:.3f} ms/ind")

    # --- Batch repair timing ---
    print(f"\n--- Batch Repair (N={N_REPAIR}) ---")
    from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair, repair_batch

    rep_engine = BitsetSchedulingRepair(PKL_PATH)
    t0 = time.perf_counter()
    repair_batch(X_repair, rep_engine)
    t1 = time.perf_counter()
    batch_repair_total = (t1 - t0) * 1000
    results["batch_repair"] = {
        "total_ms": round(batch_repair_total, 3),
        "per_ind_ms": round(batch_repair_total / N_REPAIR, 3),
    }
    print(
        f"  Total: {batch_repair_total:.1f} ms  "
        f"per_ind: {batch_repair_total / N_REPAIR:.1f} ms/ind"
    )

    # --- Save results ---
    out_path = RESULTS_DIR / "bench_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"  Evaluator speedup: {speedup_eval:.2f}x  "
        f"({s_orig['mean_ms']:.3f} -> {per_batch:.3f} ms/ind)"
    )
    print(
        f"  Repair speedup:    {speedup_repair:.2f}x  "
        f"({s_repair_orig['mean_ms']:.1f} -> {s_repair_bs['mean_ms']:.1f} ms/ind)"
    )


if __name__ == "__main__":
    main()
