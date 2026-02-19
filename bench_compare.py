#!/usr/bin/env python3
"""Head-to-head benchmark: DEAP vs pymoo on identical seeds.

Logs per-generation metrics for both solvers:
  - best_hard, best_soft, mean_hard, mean_soft
  - time_per_gen
  - cv_min, cv_mean (pymoo only — sum-of-G constraint violations)

Output:
  results/bench_compare/runs.jsonl    — one JSON object per (solver, seed, gen)
  results/bench_compare/summary.json  — aggregate summary across seeds

Usage:
    python bench_compare.py --gens 50 --pop 50 --seeds 3
    python bench_compare.py --pymoo-only --gens 100 --pop 100 --seeds 5
    python bench_compare.py --deap-only  --gens 50  --pop 50  --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "bench_compare"


# =====================================================================
#  pymoo runner
# =====================================================================


def run_pymoo(pop_size: int, n_gen: int, seed: int) -> dict:
    """Run pymoo NSGA-II and return per-generation metrics."""
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize

    from pymoo_operators import create_algorithm
    from scheduling_problem import create_problem

    class TrackingCallback(Callback):
        def __init__(self):
            super().__init__()
            self.rows: list[dict] = []
            self._gen_t0: float = time.perf_counter()

        def notify(self, algorithm):
            now = time.perf_counter()
            dt = now - self._gen_t0
            self._gen_t0 = now

            pop = algorithm.pop
            F = pop.get("F")
            G = pop.get("G")
            cv = G.sum(axis=1).clip(0)

            best_cv_idx = int(np.argmin(cv))
            hard_vals = F[:, 0]
            soft_vals = F[:, 1]

            self.rows.append(
                {
                    "solver": "pymoo",
                    "seed": seed,
                    "gen": int(algorithm.n_gen),
                    "best_hard": float(hard_vals[best_cv_idx]),
                    "best_soft": float(soft_vals[best_cv_idx]),
                    "mean_hard": float(hard_vals.mean()),
                    "mean_soft": float(soft_vals.mean()),
                    "cv_min": float(cv.min()),
                    "cv_mean": float(cv.mean()),
                    "n_feasible": int((cv == 0).sum()),
                    "time_per_gen": float(dt),
                }
            )
            # Progress line so user can see it's not stuck
            gen = algorithm.n_gen
            print(
                f"    gen {gen:3d}  hard={hard_vals[best_cv_idx]:.0f}  "
                f"soft={soft_vals[best_cv_idx]:.0f}  "
                f"cv_min={cv.min():.0f}  "
                f"feasible={int((cv == 0).sum())}  "
                f"({dt:.1f}s)",
                flush=True,
            )

    callback = TrackingCallback()
    pkl_path = str(PROJECT_ROOT / "events_with_domains.pkl")
    if not Path(pkl_path).exists():
        from build_events import build_events_with_domains

        build_events_with_domains()

    prob = create_problem(pkl_path)
    algo = create_algorithm(
        pkl_path=pkl_path, pop_size=pop_size, algorithm="nsga2", seed=seed
    )

    t0 = time.perf_counter()
    res = minimize(
        prob, algo, ("n_gen", n_gen), seed=seed, verbose=False, callback=callback
    )
    elapsed = time.perf_counter() - t0

    # Final snapshot
    pop = res.pop
    F = pop.get("F")
    G = pop.get("G")
    cv = G.sum(axis=1).clip(0)
    best_idx = int(np.argmin(cv))

    summary = {
        "solver": "pymoo",
        "seed": seed,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "elapsed_s": float(elapsed),
        "sec_per_gen": float(elapsed / n_gen) if n_gen else 0.0,
        "final_best_hard": float(F[best_idx, 0]),
        "final_best_soft": float(F[best_idx, 1]),
        "final_cv_min": float(cv.min()),
        "final_n_feasible": int((cv == 0).sum()),
    }
    return {"rows": callback.rows, "summary": summary}


# =====================================================================
#  DEAP runner
# =====================================================================


def run_deap(pop_size: int, n_gen: int, seed: int) -> dict:
    """Run DEAP-style NSGA-II and return per-generation metrics.

    Reimplements a lean evolution loop using the existing DEAP
    infrastructure (evaluator, crossover, mutation) so we capture
    per-generation timing without depending on the full
    BaselineExperiment logging.
    """
    import random as stdlib_random

    from src.constraints.evaluator import Evaluator
    from src.ga.core.population import generate_course_group_aware_population
    from src.ga.operators.constraint_guided_mutation import constraint_guided_mutation
    from src.ga.operators.crossover import crossover_course_group_aware
    from src.io.data_store import DataStore
    from src.io.time_system import QuantumTimeSystem

    store = DataStore.from_json("data")
    ctx = store.to_context()
    qts = QuantumTimeSystem()
    evaluator = Evaluator()

    np.random.seed(seed)
    stdlib_random.seed(seed)

    # ---------- helpers -------------------------------------------------
    def _eval_pop(population: list) -> list[tuple[float, float]]:
        return [evaluator.fitness(g, ctx, qts) for g in population]

    def _record(gen: int, scores: list[tuple[float, float]], dt: float) -> dict:
        hards = np.array([s[0] for s in scores])
        softs = np.array([s[1] for s in scores])
        best_idx = int(np.argmin(hards))
        return {
            "solver": "deap",
            "seed": seed,
            "gen": gen,
            "best_hard": float(hards[best_idx]),
            "best_soft": float(softs[best_idx]),
            "mean_hard": float(hards.mean()),
            "mean_soft": float(softs.mean()),
            "cv_min": None,  # DEAP has no G-vector
            "cv_mean": None,
            "n_feasible": int((hards == 0).sum()),
            "time_per_gen": float(dt),
        }

    # ---------- init ----------------------------------------------------
    t_gen0 = time.perf_counter()
    print("  DEAP init: generating population...", flush=True)
    pop = generate_course_group_aware_population(pop_size, ctx, parallel=False)
    print("  DEAP init: evaluating...", flush=True)
    scores = _eval_pop(pop)
    dt_init = time.perf_counter() - t_gen0
    print(f"  DEAP init done ({dt_init:.1f}s)", flush=True)
    rows: list[dict] = [_record(0, scores, dt_init)]

    t_total0 = time.perf_counter()

    # ---------- evolution loop ------------------------------------------
    for gen in range(1, n_gen + 1):
        t_gen = time.perf_counter()

        # Create offspring
        offspring: list = []
        for _ in range(pop_size):
            i, j = stdlib_random.sample(range(len(pop)), 2)
            p1 = pop[i] if scores[i][0] <= scores[j][0] else pop[j]
            i, j = stdlib_random.sample(range(len(pop)), 2)
            p2 = pop[i] if scores[i][0] <= scores[j][0] else pop[j]

            if stdlib_random.random() < 0.9:
                try:
                    c1, _c2 = crossover_course_group_aware(
                        list(p1), list(p2), cx_prob=0.5, validate=False
                    )
                    child = c1
                except Exception:
                    child = list(p1)
            else:
                child = list(p1)

            if stdlib_random.random() < 0.3:
                try:
                    child, _stats = constraint_guided_mutation(child, ctx)
                except Exception:
                    pass

            offspring.append(child)

        off_scores = _eval_pop(offspring)

        # Merge & truncation select (simple elitist)
        combined = list(zip(pop + offspring, scores + off_scores))
        combined.sort(key=lambda x: (x[1][0], x[1][1]))
        combined = combined[:pop_size]
        pop = [p for p, _ in combined]
        scores = [s for _, s in combined]

        dt = time.perf_counter() - t_gen
        rows.append(_record(gen, scores, dt))
        # Progress line
        best_h = scores[0][0]
        best_s = scores[0][1]
        n_feas = sum(1 for s in scores if s[0] == 0)
        print(
            f"    gen {gen:3d}  hard={best_h:.0f}  soft={best_s:.0f}  "
            f"feasible={n_feas}  ({dt:.1f}s)",
            flush=True,
        )

    elapsed = time.perf_counter() - t_total0 + dt_init

    hards_final = np.array([s[0] for s in scores])
    softs_final = np.array([s[1] for s in scores])
    best_idx = int(np.argmin(hards_final))

    summary = {
        "solver": "deap",
        "seed": seed,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "elapsed_s": float(elapsed),
        "sec_per_gen": float(elapsed / n_gen) if n_gen else 0.0,
        "final_best_hard": float(hards_final[best_idx]),
        "final_best_soft": float(softs_final[best_idx]),
        "final_cv_min": None,
        "final_n_feasible": int((hards_final == 0).sum()),
    }
    return {"rows": rows, "summary": summary}


# =====================================================================
#  CLI entry point
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="DEAP vs pymoo head-to-head benchmark")
    parser.add_argument("--gens", type=int, default=50, help="Generations per run")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to run")
    parser.add_argument(
        "--seed-start", type=int, default=42, help="First seed value (default: 42)"
    )
    parser.add_argument("--pymoo-only", action="store_true", help="Skip DEAP runs")
    parser.add_argument("--deap-only", action="store_true", help="Skip pymoo runs")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = RESULTS_DIR / "runs.jsonl"
    summary_path = RESULTS_DIR / "summary.json"

    all_rows: list[dict] = []
    all_summaries: list[dict] = []

    seeds = [args.seed_start + i for i in range(args.seeds)]

    # Truncate JSONL at start so we get a clean file
    jsonl_path.write_text("")

    for seed in seeds:
        # ---- pymoo ----
        if not args.deap_only:
            print(f"\n{'=' * 60}")
            print(f"PYMOO  seed={seed}  pop={args.pop}  gens={args.gens}")
            print(f"{'=' * 60}")
            result = run_pymoo(args.pop, args.gens, seed)
            all_rows.extend(result["rows"])
            all_summaries.append(result["summary"])
            s = result["summary"]
            print(
                f"  Done {s['elapsed_s']:.1f}s  ({s['sec_per_gen']:.2f}s/gen)  "
                f"best_hard={s['final_best_hard']:.0f}  "
                f"best_soft={s['final_best_soft']:.0f}  "
                f"feasible={s['final_n_feasible']}"
            )
            # Incremental save so crashes don't lose data
            with open(jsonl_path, "a") as f:
                for row in result["rows"]:
                    f.write(json.dumps(row, default=str) + "\n")

        # ---- DEAP ----
        if not args.pymoo_only:
            print(f"\n{'=' * 60}")
            print(f"DEAP   seed={seed}  pop={args.pop}  gens={args.gens}")
            print(f"{'=' * 60}")
            result = run_deap(args.pop, args.gens, seed)
            all_rows.extend(result["rows"])
            all_summaries.append(result["summary"])
            s = result["summary"]
            print(
                f"  Done {s['elapsed_s']:.1f}s  ({s['sec_per_gen']:.2f}s/gen)  "
                f"best_hard={s['final_best_hard']:.0f}  "
                f"best_soft={s['final_best_soft']:.0f}  "
                f"feasible={s['final_n_feasible']}"
            )
            # Incremental save
            with open(jsonl_path, "a") as f:
                for row in result["rows"]:
                    f.write(json.dumps(row, default=str) + "\n")

    # ---- JSONL already written incrementally ----
    print(f"\nWrote {len(all_rows)} rows -> {jsonl_path}")

    # ---- Write summary JSON ----
    # Also compute cross-seed aggregates
    agg: dict[str, dict] = {}
    for solver in ("pymoo", "deap"):
        runs = [s for s in all_summaries if s["solver"] == solver]
        if not runs:
            continue
        hards = [r["final_best_hard"] for r in runs]
        softs = [r["final_best_soft"] for r in runs]
        times = [r["elapsed_s"] for r in runs]
        spg = [r["sec_per_gen"] for r in runs]
        agg[solver] = {
            "n_runs": len(runs),
            "median_best_hard": float(np.median(hards)),
            "median_best_soft": float(np.median(softs)),
            "mean_best_hard": float(np.mean(hards)),
            "mean_best_soft": float(np.mean(softs)),
            "median_sec_per_gen": float(np.median(spg)),
            "mean_elapsed_s": float(np.mean(times)),
        }

    out = {
        "config": {
            "gens": args.gens,
            "pop": args.pop,
            "seeds": seeds,
        },
        "per_run": all_summaries,
        "aggregate": agg,
    }
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote summary -> {summary_path}")

    # ---- Print quick comparison ----
    print(f"\n{'=' * 60}")
    print("AGGREGATE COMPARISON")
    print(f"{'=' * 60}")
    header = f"{'Metric':<28} {'pymoo':>12} {'deap':>12}"
    print(header)
    print("-" * len(header))
    for metric in (
        "median_best_hard",
        "median_best_soft",
        "mean_best_hard",
        "mean_best_soft",
        "median_sec_per_gen",
        "mean_elapsed_s",
    ):
        pv = agg.get("pymoo", {}).get(metric, "-")
        dv = agg.get("deap", {}).get(metric, "-")
        pv_s = f"{pv:.2f}" if isinstance(pv, float) else str(pv)
        dv_s = f"{dv:.2f}" if isinstance(dv, float) else str(dv)
        print(f"  {metric:<26} {pv_s:>12} {dv_s:>12}")


if __name__ == "__main__":
    main()
