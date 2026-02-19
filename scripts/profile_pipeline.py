#!/usr/bin/env python3
"""Profile the scheduling pipeline to find vectorization targets.

Usage:
    python scripts/profile_pipeline.py [--pop 200] [--gens 50]

Produces:
    results/profile_cprofile.txt    – cProfile cumulative report
    results/profile_pyinstrument.html – pyinstrument flame chart
    results/profile_summary.json    – machine-readable hotspot summary
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pop", type=int, default=200)
parser.add_argument("--gens", type=int, default=50)
args = parser.parse_args()

POP = args.pop
NGEN = args.gens
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────
def build_pkl_if_needed():
    """Ensure events_with_domains.pkl exists."""
    pkl = Path("events_with_domains.pkl")
    if pkl.exists():
        return str(pkl)
    from src.pipeline.build_events import build_events_with_domains

    build_events_with_domains()
    return str(pkl)


def make_problem_and_algo(pkl_path: str, pop_size: int):
    from src.pipeline.pymoo_operators import create_algorithm
    from src.pipeline.scheduling_problem import create_problem

    problem = create_problem(pkl_path)
    algo = create_algorithm(pkl_path, pop_size=pop_size)
    return problem, algo


def run_minimization(problem, algo, n_gen: int):
    """Run pymoo minimize — the main thing we profile."""
    from pymoo.optimize import minimize

    res = minimize(problem, algo, ("n_gen", n_gen), seed=42, verbose=False)
    return res


# ── SECTION 1: Micro-benchmarks (individual functions) ──────────────
def micro_benchmarks(pkl_path: str):
    """Time individual pipeline steps on synthetic populations."""
    import pickle

    from src.pipeline.fast_evaluator_batch import (
        fast_evaluate_hard_batch,
        prepare_batch_data,
    )
    from src.pipeline.fast_evaluator_vectorized import (
        fast_evaluate_hard_vectorized,
        prepare_vectorized_data,
    )
    from src.pipeline.pymoo_operators import ConstructiveSampling
    from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair
    from src.pipeline.scheduling_problem import SchedulingProblem

    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    batch_data = prepare_batch_data(pkl_data)
    vec_data = prepare_vectorized_data(pkl_data)
    prob = SchedulingProblem(pkl_path)

    # Build a population via constructive sampling
    sampling = ConstructiveSampling(pkl_path)
    t0 = time.perf_counter()
    X = sampling._do(prob, POP)
    t_sampling = time.perf_counter() - t0

    results = {"constructive_sampling": {"pop": POP, "time_s": t_sampling}}

    # Hard eval — batch
    t0 = time.perf_counter()
    for _ in range(5):
        G_batch = fast_evaluate_hard_batch(X, batch_data)
    t_batch = (time.perf_counter() - t0) / 5
    results["hard_eval_batch"] = {"pop": POP, "time_s": t_batch}

    # Hard eval — vectorized
    t0 = time.perf_counter()
    for _ in range(5):
        G_vec = fast_evaluate_hard_vectorized(X, vec_data)
    t_vec = (time.perf_counter() - t0) / 5
    results["hard_eval_vectorized"] = {"pop": POP, "time_s": t_vec}
    results["hard_eval_speedup_vec_vs_batch"] = round(t_batch / max(t_vec, 1e-9), 2)

    # Equivalence check
    if not np.array_equal(G_batch, G_vec):
        diffs = np.sum(G_batch != G_vec)
        results["hard_eval_equiv"] = f"MISMATCH: {diffs} cells differ"
    else:
        results["hard_eval_equiv"] = "EXACT_MATCH"

    # Repair — per individual
    repairer = BitsetSchedulingRepair(pkl_path)
    n_repair = min(POP, 30)  # repair is slow, sample
    t0 = time.perf_counter()
    for i in range(n_repair):
        repairer.repair(X[i].copy())
    t_repair = (time.perf_counter() - t0) / n_repair
    results["repair_per_individual"] = {
        "n_sampled": n_repair,
        "time_per_ind_s": t_repair,
        "est_full_pop_s": t_repair * POP,
    }

    # Soft eval — per individual (if evaluator available)
    try:
        t0 = time.perf_counter()
        for i in range(min(10, POP)):
            prob._evaluate_soft(X[i].astype(int))
        t_soft = (time.perf_counter() - t0) / min(10, POP)
        results["soft_eval_per_individual"] = {
            "n_sampled": min(10, POP),
            "time_per_ind_s": t_soft,
            "est_full_pop_s": t_soft * POP,
        }
    except Exception as e:
        results["soft_eval_per_individual"] = {"error": str(e)}

    # Full _evaluate call
    out = {}
    t0 = time.perf_counter()
    prob._evaluate(X, out)
    t_full = time.perf_counter() - t0
    results["full_evaluate"] = {"pop": POP, "time_s": t_full}

    return results


# ── SECTION 2: Full run with cProfile ───────────────────────────────
def profile_full_run(pkl_path: str):
    problem, algo = make_problem_and_algo(pkl_path, POP)

    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.perf_counter()
    res = run_minimization(problem, algo, NGEN)
    wall_time = time.perf_counter() - t0
    profiler.disable()

    # Save cProfile text
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(80)
    cprof_text = s.getvalue()
    (OUT_DIR / "profile_cprofile.txt").write_text(cprof_text, encoding="utf-8")

    # Also save sorted by tottime
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats("tottime")
    ps2.print_stats(80)
    cprof_tottime = s2.getvalue()
    (OUT_DIR / "profile_cprofile_tottime.txt").write_text(
        cprof_tottime, encoding="utf-8"
    )

    # Extract top hotspots programmatically
    stats_list = []
    for key, val in profiler.stats.items():
        filename, lineno, func = key
        cc, nc, tt, ct, callers = val
        stats_list.append(
            {
                "file": os.path.basename(filename),
                "lineno": lineno,
                "function": func,
                "ncalls": nc,
                "tottime": round(tt, 4),
                "cumtime": round(ct, 4),
            }
        )
    stats_list.sort(key=lambda x: x["cumtime"], reverse=True)
    top_hotspots = stats_list[:30]

    return {
        "wall_time_s": round(wall_time, 2),
        "pop": POP,
        "gens": NGEN,
        "top_hotspots": top_hotspots,
        "best_F": res.F.tolist() if res.F is not None else None,
    }


# ── SECTION 3: pyinstrument flame chart ────────────────────────────
def profile_pyinstrument(pkl_path: str):
    import pyinstrument

    problem, algo = make_problem_and_algo(pkl_path, POP)

    profiler = pyinstrument.Profiler()
    profiler.start()
    run_minimization(problem, algo, NGEN)
    profiler.stop()

    # Save HTML
    html = profiler.output_html()
    (OUT_DIR / "profile_pyinstrument.html").write_text(html, encoding="utf-8")

    # Save text summary
    text = profiler.output_text(unicode=False)
    (OUT_DIR / "profile_pyinstrument.txt").write_text(text, encoding="utf-8")

    return {"saved": "results/profile_pyinstrument.html"}


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"=== Profiling pipeline: pop={POP}, gens={NGEN} ===")
    print("Problem: 549 events, 75 rooms, 189 instructors, 42 quanta")
    print()

    pkl_path = build_pkl_if_needed()

    # 1) Micro-benchmarks
    print("── Micro-benchmarks ──")
    micro = micro_benchmarks(pkl_path)
    for k, v in micro.items():
        print(f"  {k}: {v}")
    print()

    # 2) cProfile full run
    print(f"── cProfile full run (pop={POP}, gens={NGEN}) ──")
    cprof = profile_full_run(pkl_path)
    print(f"  Wall time: {cprof['wall_time_s']}s")
    print("  Top 10 hotspots (by cumtime):")
    for h in cprof["top_hotspots"][:10]:
        print(
            f"    {h['function']:40s} {h['tottime']:8.3f}s tot  {h['cumtime']:8.3f}s cum  ({h['ncalls']} calls)  [{h['file']}:{h['lineno']}]"
        )
    print()

    # 3) pyinstrument
    print(f"── pyinstrument (pop={POP}, gens={NGEN}) ──")
    pi_result = profile_pyinstrument(pkl_path)
    print(f"  {pi_result}")
    print()

    # 4) Save combined summary
    summary = {
        "problem": {
            "n_events": 549,
            "n_rooms": 75,
            "n_instructors": 189,
            "quanta": 42,
        },
        "params": {"pop": POP, "gens": NGEN},
        "micro_benchmarks": micro,
        "cprofile": {
            "wall_time_s": cprof["wall_time_s"],
            "top_hotspots": cprof["top_hotspots"][:20],
        },
    }
    (OUT_DIR / "profile_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=== Profiling complete ===")
    print("  results/profile_cprofile.txt")
    print("  results/profile_cprofile_tottime.txt")
    print("  results/profile_pyinstrument.html")
    print("  results/profile_pyinstrument.txt")
    print("  results/profile_summary.json")
