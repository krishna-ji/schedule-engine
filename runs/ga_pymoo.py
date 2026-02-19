#!/usr/bin/env python3
"""Pymoo GA run script with selectable modes.

Modes
-----
  baseline  – Standard NSGA-II   (default, moderate mutation)
  memetic   – NSGA-II + elite local-search repair each generation

Usage
-----
    python runs/ga_pymoo.py --mode baseline --gens 200 --pop 100 --seed 42
    python runs/ga_pymoo.py --mode memetic  --gens 150 --pop 80  --output output/memetic_run.json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

__version__ = "1.0.0"

# ── mode presets ────────────────────────────────────────────────────
MODE_PRESETS: dict[str, dict] = {
    "baseline": {
        "algorithm": "nsga2",
        "crossover_prob": 0.5,
        "mutation_event_prob": 0.05,
        "elite_repair": False,
    },
    "memetic": {
        "algorithm": "nsga2",
        "crossover_prob": 0.6,
        "mutation_event_prob": 0.10,
        "elite_repair": True,
        "elite_pct": 0.05,
        "repair_iterations": 5,
    },
}


# ── progress callback ──────────────────────────────────────────────
def _make_callback(log_interval: int):
    from pymoo.core.callback import Callback

    class ProgressCallback(Callback):
        def __init__(self):
            super().__init__()
            self.best_hards: list[float] = []
            self.best_softs: list[float] = []

        def notify(self, algorithm):
            F = algorithm.pop.get("F")
            G = algorithm.pop.get("G")
            cv = G.sum(axis=1).clip(0)
            best_idx = int(np.argmin(cv))

            if algorithm.n_gen == 1 or algorithm.n_gen % log_interval == 0:
                print(
                    f"  Gen {algorithm.n_gen:4d}: "
                    f"hard={F[best_idx, 0]:.0f}  "
                    f"soft={F[best_idx, 1]:.0f}  "
                    f"cv={cv.min():.0f}  "
                    f"feasible={int((cv == 0).sum())}/{len(algorithm.pop)}"
                )
            self.best_hards.append(float(F[best_idx, 0]))
            self.best_softs.append(float(F[best_idx, 1]))

    return ProgressCallback()


# ── memetic local-search callback ──────────────────────────────────
def _make_memetic_callback(
    pkl_path: str,
    elite_pct: float,
    repair_iters: int,
    log_interval: int,
):
    """Wrap the progress callback with per-generation elite repair.

    After each generation the top *elite_pct* individuals (by constraint
    violation) are passed through the full SchedulingRepair pipeline
    *repair_iters* times, intensifying local search on the best solutions.
    """
    from pymoo.core.callback import Callback

    from src.pipeline.repair_operator import SchedulingRepair

    base_cb = _make_callback(log_interval)
    repairer = SchedulingRepair(pkl_path)

    class MemeticCallback(Callback):
        def __init__(self):
            super().__init__()

        def notify(self, algorithm):
            # Regular progress logging first
            base_cb.notify(algorithm)

            # Elite local-search repair
            pop = algorithm.pop
            G = pop.get("G")
            cv = G.sum(axis=1).clip(0)
            n_elite = max(1, int(len(pop) * elite_pct))

            # Pick the n_elite individuals with lowest constraint violation
            elite_idxs = np.argsort(cv)[:n_elite]

            for idx in elite_idxs:
                X = pop[idx].get("X").copy()
                for _ in range(repair_iters):
                    X = repairer.repair(X)
                pop[idx].set("X", X)

        @property
        def best_hards(self):
            return base_cb.best_hards

        @property
        def best_softs(self):
            return base_cb.best_softs

    return MemeticCallback()


# ── solver ─────────────────────────────────────────────────────────
def run(args) -> dict:
    """Execute a GA run with the chosen mode preset."""
    from pymoo.optimize import minimize

    from src.pipeline.build_events import build_events_with_domains
    from src.pipeline.pymoo_operators import create_algorithm
    from src.pipeline.scheduling_problem import create_problem

    preset = MODE_PRESETS[args.mode]

    pkl_path = str(PROJECT_ROOT / "events_with_domains.pkl")
    if not Path(pkl_path).exists():
        print("Building events_with_domains.pkl ...")
        build_events_with_domains()

    print(f"Mode: {args.mode}  |  algo={preset['algorithm']}  "
          f"pop={args.pop}  gens={args.gens}  seed={args.seed}")

    prob = create_problem(pkl_path)
    algo = create_algorithm(
        pkl_path=pkl_path,
        pop_size=args.pop,
        crossover_prob=preset["crossover_prob"],
        mutation_event_prob=preset["mutation_event_prob"],
        algorithm=preset["algorithm"],
        seed=args.seed,
    )

    log_interval = max(1, args.gens // 20)

    if preset.get("elite_repair"):
        callback = _make_memetic_callback(
            pkl_path,
            elite_pct=preset.get("elite_pct", 0.05),
            repair_iters=preset.get("repair_iterations", 5),
            log_interval=log_interval,
        )
    else:
        callback = _make_callback(log_interval)

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

    # Best solution
    F = res.pop.get("F")
    G = res.pop.get("G")
    cv = G.sum(axis=1).clip(0)
    best_idx = int(np.argmin(cv))

    print(f"\nDone in {elapsed:.1f}s  ({elapsed / args.gens:.2f}s/gen)")
    print(f"Best: hard={F[best_idx, 0]:.0f}  "
          f"soft={F[best_idx, 1]:.0f}  cv={cv[best_idx]:.0f}")

    return {
        "solver": "pymoo",
        "mode": args.mode,
        "version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "gens": args.gens,
            "pop": args.pop,
            "seed": args.seed,
            **{k: v for k, v in preset.items() if k != "algorithm"},
        },
        "best_hard": float(F[best_idx, 0]),
        "best_soft": float(F[best_idx, 1]),
        "best_cv": float(cv[best_idx]),
        "n_feasible": int((cv == 0).sum()),
        "elapsed_s": round(elapsed, 2),
        "sec_per_gen": round(elapsed / args.gens, 3) if args.gens else 0,
        "convergence_hard": callback.best_hards,
        "convergence_soft": callback.best_softs,
    }


# ── CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pymoo GA run — baseline & memetic modes"
    )
    parser.add_argument(
        "--mode",
        choices=list(MODE_PRESETS),
        default="baseline",
        help="Run mode (default: baseline)",
    )
    parser.add_argument("--gens", type=int, default=200, help="Generations")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Save results JSON to this path"
    )
    args = parser.parse_args()

    result = run(args)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
