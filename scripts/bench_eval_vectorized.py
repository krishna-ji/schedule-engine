#!/usr/bin/env python3
"""Benchmark: population-level vectorized evaluator vs per-individual batch.

Measures fast_evaluate_hard_vectorized vs fast_evaluate_hard_batch
across N=50, 100, 200, 400, 800 individuals.

Reports mean / median / p95 in ms, per-individual cost, and speedup.
Saves results to results/bench_eval_vectorized.json.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.utils.logging_config import quick_setup

logger = logging.getLogger(__name__)

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


def bench(fn, X, n_reps=5):
    """Run fn(X) n_reps times, return list of total-ms."""
    fn(X)  # warm-up
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn(X)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def stats(times_ms, N):
    arr = np.array(times_ms)
    return {
        "total_mean_ms": round(float(np.mean(arr)), 3),
        "total_median_ms": round(float(np.median(arr)), 3),
        "total_p95_ms": round(float(np.percentile(arr, 95)), 3),
        "per_ind_mean_ms": round(float(np.mean(arr)) / N, 4),
        "per_ind_median_ms": round(float(np.median(arr)) / N, 4),
        "per_ind_p95_ms": round(float(np.percentile(arr, 95)) / N, 4),
    }


def main():
    logger.info("=" * 70)
    logger.info("BENCHMARK: Vectorized vs Batch Hard-Constraint Evaluator")
    logger.info("=" * 70)

    data = load_data()
    E = len(data["events"])
    rng = np.random.default_rng(42)

    from src.pipeline.fast_evaluator_batch import (
        fast_evaluate_hard_batch,
        prepare_batch_data,
    )
    from src.pipeline.fast_evaluator_vectorized import (
        fast_evaluate_hard_vectorized,
        prepare_vectorized_data,
    )

    bdata = prepare_batch_data(data)
    vdata = prepare_vectorized_data(data)

    population_sizes = [50, 100, 200, 400, 800]
    results = {"n_events": E, "T": 42, "benchmarks": []}

    for N in population_sizes:
        X = make_random_population(data, N, rng)

        # Correctness check
        G_batch = fast_evaluate_hard_batch(X, bdata)
        G_vec = fast_evaluate_hard_vectorized(X, vdata)
        assert np.array_equal(G_batch, G_vec), f"Mismatch at N={N}!"

        # Benchmark
        t_batch = bench(lambda x: fast_evaluate_hard_batch(x, bdata), X, n_reps=7)
        t_vec = bench(lambda x: fast_evaluate_hard_vectorized(x, vdata), X, n_reps=7)

        s_batch = stats(t_batch, N)
        s_vec = stats(t_vec, N)
        speedup = s_batch["total_mean_ms"] / s_vec["total_mean_ms"]

        entry = {
            "N": N,
            "batch": s_batch,
            "vectorized": s_vec,
            "speedup": round(speedup, 2),
        }
        results["benchmarks"].append(entry)

        logger.info("--- N = %d ---", N)
        logger.info(
            "  Batch (per-ind loop):  mean=%8.1f ms  (%.4f ms/ind)  median=%.1f  p95=%.1f",
            s_batch["total_mean_ms"],
            s_batch["per_ind_mean_ms"],
            s_batch["total_median_ms"],
            s_batch["total_p95_ms"],
        )
        logger.info(
            "  Vectorized:            mean=%8.1f ms  (%.4f ms/ind)  median=%.1f  p95=%.1f",
            s_vec["total_mean_ms"],
            s_vec["per_ind_mean_ms"],
            s_vec["total_median_ms"],
            s_vec["total_p95_ms"],
        )
        logger.info("  Speedup: %.2fx", speedup)

    # Save
    out_path = RESULTS_DIR / "bench_eval_vectorized.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Summary table
    logger.info("=" * 70)
    logger.info("%6s  %14s  %12s  %8s", "N", "Batch ms/ind", "Vec ms/ind", "Speedup")
    logger.info("-" * 50)
    for entry in results["benchmarks"]:
        N = entry["N"]
        b = entry["batch"]["per_ind_mean_ms"]
        v = entry["vectorized"]["per_ind_mean_ms"]
        s = entry["speedup"]
        logger.info("%6d  %14.4f  %12.4f  %7.2fx", N, b, v, s)
    logger.info("=" * 70)


if __name__ == "__main__":
    quick_setup()
    main()
