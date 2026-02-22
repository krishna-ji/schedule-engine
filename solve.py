#!/usr/bin/env python3
"""Unified scheduling solver CLI.

Solver: **pymoo** (NSGA-II with vectorized evaluator).

Usage:
    python solve.py --gens 100 --pop 50 --seed 42

Automatically generates ALL outputs:
    - Convergence & MOEA metric plots (17 PDFs)
    - Schedule PDFs (calendar, instructor, room)
    - CSV exports (pareto front, population fitness)
    - results.json with full metrics
    - Violation report

Output is written to ``output/ga_baseline/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="University timetable scheduling solver"
    )
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip schedule PDF generation (faster)",
    )
    parser.add_argument(
        "--force-pdf",
        action="store_true",
        help="Generate schedule PDFs even when no feasible solution exists",
    )
    args = parser.parse_args()

    from src.experiments import BaselineExperiment

    exp = BaselineExperiment(
        pop_size=args.pop,
        ngen=args.gens,
        seed=args.seed,
        export_pdf=not args.no_pdf,
        force_pdf=args.force_pdf,
        verbose=True,
    )
    exp.run()


if __name__ == "__main__":
    main()
