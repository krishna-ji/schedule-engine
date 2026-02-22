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
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import quick_setup

logger = logging.getLogger(__name__)


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

    from src.experiments.ga_experiment import BaselineExperiment

    all_F: list[np.ndarray] = []

    for i in range(args.runs):
        seed = args.base_seed + i
        logger.info("=" * 60)
        logger.info(
            "  Run %d/%d  seed=%d  pop=%d  gens=%d",
            i + 1,
            args.runs,
            seed,
            args.pop,
            args.gens,
        )
        logger.info("=" * 60)
        t0 = time.time()

        exp = BaselineExperiment(
            pop_size=args.pop,
            ngen=args.gens,
            seed=seed,
            export_pdf=False,
            verbose=False,
        )
        result = exp.run()
        elapsed = time.time() - t0

        best_hard = result.get("best_hard", float("inf"))
        best_cv = result.get("best_cv", float("inf"))
        logger.info(
            "  Run %d done in %.1fs  best_hard=%.0f  best_cv=%.0f",
            i + 1,
            elapsed,
            best_hard,
            best_cv,
        )

        # Extract feasible solutions from final population
        F = np.array(result["final_F"])
        G = np.array(result["final_G"])
        cv = G.sum(axis=1).clip(0)
        feasible = cv <= 0
        if feasible.any():
            all_F.append(F[feasible])
            logger.info("  Collected %d feasible points", int(feasible.sum()))
        else:
            logger.info("  No feasible solutions in this run")

    if not all_F:
        logger.error(
            "No feasible solutions across all runs — cannot build reference front."
        )
        sys.exit(1)

    # Combine and apply non-dominated sorting
    combined = np.vstack(all_F)
    logger.info("Combined pool: %d feasible objective vectors", combined.shape[0])

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    nds = NonDominatedSorting()
    fronts = nds.do(combined)
    pf = combined[fronts[0]]
    logger.info("Reference front size: %d", pf.shape[0])

    # Save
    npy_path = PROJECT_ROOT / "reference_front.npy"
    csv_path = PROJECT_ROOT / "reference_front.csv"
    np.save(npy_path, pf)
    np.savetxt(csv_path, pf, delimiter=",", header="hard,soft", comments="")
    logger.info("Saved: %s", npy_path)
    logger.info("Saved: %s", csv_path)


if __name__ == "__main__":
    quick_setup()
    main()
