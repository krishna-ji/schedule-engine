#!/usr/bin/env python3
"""Build an approximate reference Pareto front from multiple GA runs.

The script runs ``BaselineExperiment`` *N* times with different seeds,
collects all feasible objective vectors, applies non-dominated sorting,
and saves the resulting front to ``reference_front.npy`` (numpy) and
``reference_front.csv`` (human-readable) in the project root.

Usage:
    python scripts/build_reference_front.py --runs 5 --gens 200 --pop 100
    python scripts/build_reference_front.py --runs 3 --gens 50 --pop 30  # quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reference Pareto front from multiple GA runs."
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of seeded runs")
    parser.add_argument("--gens", type=int, default=200, help="Generations per run")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument(
        "--base-seed", type=int, default=42, help="Base seed (each run = base+i)"
    )
    args = parser.parse_args()

    from src.experiments.ga_experiment import BaselineExperiment, GAExperiment

    all_F: list[np.ndarray] = []

    for i in range(args.runs):
        seed = args.base_seed + i
        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{args.runs}  seed={seed}  pop={args.pop}  gens={args.gens}")
        print(f"{'='*60}")
        t0 = time.time()

        exp = BaselineExperiment(
            pop_size=args.pop,
            ngen=args.gens,
            seed=seed,
            export_pdf=False,
            verbose=False,
        )
        result = exp.run()

        # Collect feasible rows from the final population
        # Re-load the result's raw arrays from the experiment
        # The experiment object stores the pymoo result internally—
        # but the public API returns a dict.  We re-run minimise quickly?
        # Actually, the _execute already ran.  Extract from result dict.
        best_hard = result.get("best_hard", float("inf"))
        best_cv = result.get("best_cv", float("inf"))
        elapsed = time.time() - t0
        print(
            f"  Run {i+1} done in {elapsed:.1f}s  "
            f"best_hard={best_hard:.0f}  best_cv={best_cv:.0f}"
        )

        # To get the full population F/G, we need to capture it.
        # The simplest approach: re-run the problem evaluation on the
        # last population.  But that's wasteful.  Instead, we'll run
        # minimize ourselves and capture the result object.
        # ---
        # Actually let's just do the run manually:
        import pickle

        from pymoo.optimize import minimize

        from src.pipeline.pymoo_operators import create_algorithm
        from src.pipeline.scheduling_problem import create_problem

        pkl_path = str(PROJECT_ROOT / "events_with_domains.pkl")
        from src.io.data_store import DataStore
        from src.io.time_system import QuantumTimeSystem

        store = DataStore.from_json(str(PROJECT_ROOT / "data"))
        ctx = store.to_context()
        qts = QuantumTimeSystem()

        prob = create_problem(pkl_path, ctx=ctx, qts=qts)
        algo = create_algorithm(
            pkl_path=pkl_path,
            pop_size=args.pop,
            n_offsprings=args.pop,
            crossover_prob=0.5,
            mutation_event_prob=0.05,
            algorithm="nsga2",
            seed=seed,
        )

        res = minimize(prob, algo, ("n_gen", args.gens), seed=seed, verbose=False)

        F = res.pop.get("F")
        G = res.pop.get("G")
        cv = G.sum(axis=1).clip(0)
        feasible = cv <= 0
        if feasible.any():
            all_F.append(F[feasible])
            print(f"  Collected {int(feasible.sum())} feasible points")
        else:
            print("  No feasible solutions in this run")

    if not all_F:
        print("\nNo feasible solutions across all runs — cannot build reference front.")
        sys.exit(1)

    # Combine and apply non-dominated sorting
    combined = np.vstack(all_F)
    print(f"\nCombined pool: {combined.shape[0]} feasible objective vectors")

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    nds = NonDominatedSorting()
    fronts = nds.do(combined)
    pf = combined[fronts[0]]
    print(f"Reference front size: {pf.shape[0]}")

    # Save
    npy_path = PROJECT_ROOT / "reference_front.npy"
    csv_path = PROJECT_ROOT / "reference_front.csv"
    np.save(npy_path, pf)
    np.savetxt(csv_path, pf, delimiter=",", header="hard,soft", comments="")
    print(f"Saved: {npy_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
