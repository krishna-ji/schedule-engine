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
import logging
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.utils.logging_config import quick_setup

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

PKL_PATH = ".cache/events_with_domains.pkl"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load pickled event domain data from the cache."""
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def make_random_population(data, N, rng):
    """Generate a random population of N chromosomes from domain data."""
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
    """Return the p-th percentile of arr as a float."""
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
    """Compute summary statistics (mean, median, p95, min, max) in milliseconds."""
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(percentile(arr, 95), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def main():
    """Run all evaluator and repair benchmarks and export results."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Evaluator & Repair Acceleration")
    logger.info("=" * 60)

    data = load_data()
    E = len(data["events"])
    rng = np.random.default_rng(42)

    N_EVAL = 200
    N_REPAIR = 20

    X_eval = make_random_population(data, N_EVAL, rng)
    X_repair = make_random_population(data, N_REPAIR, rng)

    results = {"n_events": E, "n_eval": N_EVAL, "n_repair": N_REPAIR}

    # --- Evaluator benchmarks ---
    logger.info("--- Evaluator (N=%d) ---", N_EVAL)

    t_orig = bench_evaluator_original(data, X_eval)
    s_orig = stats(t_orig)
    results["evaluator_original"] = s_orig
    logger.info(
        "  Original:  mean=%.3f ms/ind  median=%.3f  p95=%.3f",
        s_orig["mean_ms"],
        s_orig["median_ms"],
        s_orig["p95_ms"],
    )

    total_batch, per_batch = bench_evaluator_batch(data, X_eval)
    results["evaluator_batch"] = {
        "total_ms": round(total_batch, 3),
        "per_ind_ms": round(per_batch, 3),
    }
    logger.info(
        "  Batch:     total=%.1f ms  per_ind=%.3f ms/ind", total_batch, per_batch
    )

    speedup_eval = s_orig["mean_ms"] / per_batch if per_batch > 0 else float("inf")
    results["evaluator_speedup"] = round(speedup_eval, 2)
    logger.info("  Speedup:   %.2fx", speedup_eval)

    # --- Repair benchmarks ---
    logger.info("--- Repair (N=%d) ---", N_REPAIR)

    t_repair_orig = bench_repair_original(data, X_repair, N_REPAIR)
    s_repair_orig = stats(t_repair_orig)
    results["repair_original"] = s_repair_orig
    logger.info(
        "  Original:  mean=%.1f ms/ind  median=%.1f  p95=%.1f",
        s_repair_orig["mean_ms"],
        s_repair_orig["median_ms"],
        s_repair_orig["p95_ms"],
    )

    t_repair_bs = bench_repair_bitset(data, X_repair, N_REPAIR)
    s_repair_bs = stats(t_repair_bs)
    results["repair_bitset"] = s_repair_bs
    logger.info(
        "  Bitset:    mean=%.1f ms/ind  median=%.1f  p95=%.1f",
        s_repair_bs["mean_ms"],
        s_repair_bs["median_ms"],
        s_repair_bs["p95_ms"],
    )

    speedup_repair = (
        s_repair_orig["mean_ms"] / s_repair_bs["mean_ms"]
        if s_repair_bs["mean_ms"] > 0
        else float("inf")
    )
    results["repair_speedup"] = round(speedup_repair, 2)
    logger.info("  Speedup:   %.2fx", speedup_repair)

    # --- Pymoo _evaluate benchmark ---
    logger.info("--- Pymoo _evaluate (N=%d) ---", N_EVAL)

    total_pymoo, per_pymoo = bench_pymoo_evaluate(data, X_eval)
    results["pymoo_evaluate"] = {
        "total_ms": round(total_pymoo, 3),
        "per_ind_ms": round(per_pymoo, 3),
    }
    logger.info("  Total: %.1f ms  per_ind: %.3f ms/ind", total_pymoo, per_pymoo)

    # --- Batch repair timing ---
    logger.info("--- Batch Repair (N=%d) ---", N_REPAIR)
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
    logger.info(
        "  Total: %.1f ms  per_ind: %.1f ms/ind",
        batch_repair_total,
        batch_repair_total / N_REPAIR,
    )

    # --- Save results ---
    out_path = RESULTS_DIR / "bench_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "  Evaluator speedup: %.2fx  (%.3f -> %.3f ms/ind)",
        speedup_eval,
        s_orig["mean_ms"],
        per_batch,
    )
    logger.info(
        "  Repair speedup:    %.2fx  (%.1f -> %.1f ms/ind)",
        speedup_repair,
        s_repair_orig["mean_ms"],
        s_repair_bs["mean_ms"],
    )


if __name__ == "__main__":
    quick_setup()
    main()
