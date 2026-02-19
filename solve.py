#!/usr/bin/env python3
"""Unified scheduling solver CLI.

Default solver: **pymoo** (NSGA-II with vectorized evaluator).
DEAP is kept as a fallback for regression checks.

Usage:
    python solve.py --gens 100 --pop 50 --seed 42            # pymoo (default)
    python solve.py --solver deap --gens 2000 --pop 50        # DEAP fallback

Environment override:
    SCHED_SOLVER=deap python solve.py --gens 100 --pop 50     # env kill-switch

The pymoo solver uses the numeric encoding pipeline
(build_events.py -> fast_evaluator.py -> scheduling_problem.py).

The DEAP solver uses the existing BaselineExperiment from
src/experiments.  It is deprecated and will be removed in a future
release once pymoo has been validated in production.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from datetime import UTC, datetime, timezone
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


def solve_deap(args) -> dict:
    """Run DEAP baseline via BaselineExperiment.

    .. deprecated::
        DEAP is kept as a fallback.  Use ``--solver pymoo`` (the default)
        for production runs.
    """
    warnings.warn(
        "DEAP solver is deprecated. Use --solver pymoo (the default) for "
        "production runs. DEAP will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.experiments import BaselineExperiment

    print(
        f"DEAP NSGA-II (deprecated): pop={args.pop}, gens={args.gens}, seed={args.seed}"
    )
    t0 = time.time()

    output_dir = PROJECT_ROOT / "output" / "solve_deap" / f"seed{args.seed}"
    exp = BaselineExperiment(
        seed=args.seed,
        pop_size=args.pop,
        ngen=args.gens,
        cxpb=0.9,
        mutpb=0.3,
        fitness_weights=(-1.0, -1.0),
        data_dir=PROJECT_ROOT / "data",
        output_dir=output_dir,
        opening_time="10:00",
        closing_time="17:00",
        closed_days=["Saturday"],
        init_strategy="smart",
        log_interval=max(1, args.gens // 20),
        verbose=True,
    )
    metadata = exp.run()
    elapsed = time.time() - t0

    # Extract best fitness from DEAP individual
    best = exp.best_individual
    best_hard = best.fitness.values[0] if hasattr(best, "fitness") else float("nan")
    best_soft = best.fitness.values[1] if hasattr(best, "fitness") else float("nan")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Best: hard={best_hard:.0f} soft={best_soft:.0f}")

    pkl_path = str(PROJECT_ROOT / "events_with_domains.pkl")
    return {
        **_solver_metadata(pkl_path, "deap", args),
        "best_hard": float(best_hard),
        "best_soft": float(best_soft),
        "elapsed_s": elapsed,
        "sec_per_gen": elapsed / args.gens if args.gens else 0,
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
    # Solver precedence: CLI arg > env var (SCHED_SOLVER) > instance_config > "pymoo"
    env_solver = os.environ.get("SCHED_SOLVER", "").lower().strip()
    if env_solver in ("pymoo", "deap"):
        default_solver = env_solver
    else:
        try:
            from instance_config import DEFAULT_SOLVER

            default_solver = (
                DEFAULT_SOLVER if DEFAULT_SOLVER in ("pymoo", "deap") else "pymoo"
            )
        except ImportError:
            default_solver = "pymoo"

    parser = argparse.ArgumentParser(
        description="University timetable scheduling solver"
    )
    parser.add_argument(
        "--solver",
        choices=["pymoo", "deap"],
        default=default_solver,
        help=f"Solver backend (default: {default_solver}; override with SCHED_SOLVER env var)",
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

    print(f"Solver: {args.solver}")
    print(f"Config: pop={args.pop}, gens={args.gens}, seed={args.seed}")
    print()

    if args.solver == "pymoo":
        result = solve_pymoo(args)
    else:
        result = solve_deap(args)

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
