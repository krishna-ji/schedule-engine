#!/usr/bin/env python3
"""Unified scheduling solver CLI.

Solver: **pymoo** (NSGA-II with vectorized evaluator).

Usage:
    python solve.py --gens 100 --pop 50 --seed 42

The pymoo solver uses the numeric encoding pipeline
(build_events.py -> fast_evaluator.py -> scheduling_problem.py).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

__version__ = "2.0.0"  # pymoo-default release


def solve_pymoo(args) -> dict:
    """Run pymoo NSGA-II solver."""
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize

    from pymoo_operators import create_algorithm
    from scheduling_problem import create_problem

    pkl_path = str(PROJECT_ROOT / "events_with_domains.pkl")
    if not Path(pkl_path).exists():
        print("Building events_with_domains.pkl...")
        from build_events import build_events_with_domains

        build_events_with_domains()

    class ProgressCallback(Callback):
        def __init__(self, log_interval: int = 10):
            super().__init__()
            self.log_interval = log_interval
            self.best_hards: list[float] = []

        def notify(self, algorithm):
            pop = algorithm.pop
            F = pop.get("F")
            G = pop.get("G")
            cv = G.sum(axis=1).clip(0)
            best_idx = np.argmin(cv)

            if algorithm.n_gen == 1 or algorithm.n_gen % self.log_interval == 0:
                print(
                    f"  Gen {algorithm.n_gen:4d}: "
                    f"best_hard={F[best_idx, 0]:.0f} "
                    f"best_soft={F[best_idx, 1]:.0f} "
                    f"min_cv={cv.min():.0f} "
                    f"feasible={int((cv == 0).sum())}/{len(pop)}"
                )
            self.best_hards.append(float(F[best_idx, 0]))

    print(f"Pymoo NSGA-II: pop={args.pop}, gens={args.gens}, seed={args.seed}")
    prob = create_problem(pkl_path)
    algo = create_algorithm(
        pkl_path=pkl_path,
        pop_size=args.pop,
        algorithm="nsga2",
        seed=args.seed,
    )

    callback = ProgressCallback(log_interval=max(1, args.gens // 20))
    t0 = time.time()
    res = minimize(
        prob,
        algo,
        ("n_gen", args.gens),
        seed=args.seed,
        verbose=False,
        callback=callback,
    )
    elapsed = time.time() - t0

    # Extract best solution
    pop = res.pop
    F = pop.get("F")
    G = pop.get("G")
    cv = G.sum(axis=1).clip(0)
    best_idx = np.argmin(cv)

    print(f"\nDone in {elapsed:.1f}s ({elapsed/args.gens:.2f}s/gen)")
    print(
        f"Best: hard={F[best_idx, 0]:.0f} soft={F[best_idx, 1]:.0f} cv={cv[best_idx]:.0f}"
    )

    return {
        **_solver_metadata(pkl_path, "pymoo", args),
        "best_hard": float(F[best_idx, 0]),
        "best_soft": float(F[best_idx, 1]),
        "best_cv": float(cv[best_idx]),
        "n_feasible": int((cv == 0).sum()),
        "elapsed_s": elapsed,
        "sec_per_gen": elapsed / args.gens if args.gens else 0,
        "best_chromosome": res.pop[best_idx].get("X").tolist(),
    }


# -----------------------------------------------------------------
#  Metadata helper
# -----------------------------------------------------------------


def _solver_metadata(pkl_path: str, solver_name: str, args) -> dict:
    """Return a dict of metadata fields for the result JSON."""
    meta: dict = {
        "solver": solver_name,
        "solve_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "gens": args.gens,
            "pop": args.pop,
            "seed": args.seed,
        },
    }
    try:
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        meta["data_hash"] = pkl.get("data_hash", "unknown")
        meta["schema_version"] = pkl.get("schema_version", "unknown")
        meta["n_events"] = len(pkl.get("events", []))
    except Exception:
        meta["data_hash"] = "unavailable"
        meta["schema_version"] = "unavailable"
    return meta


# -----------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="University timetable scheduling solver"
    )
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for results",
    )
    args = parser.parse_args()

    print(f"Solver: pymoo")
    print(f"Config: pop={args.pop}, gens={args.gens}, seed={args.seed}")
    print()

    result = solve_pymoo(args)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Don't save full chromosome in JSON output by default (it's huge)
        save_result = {k: v for k, v in result.items() if k != "best_chromosome"}
        with open(out_path, "w") as f:
            json.dump(save_result, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
