#!/usr/bin/env python3
"""
GA Ultimate: ILS + Full Repair Arsenal (Mode F) — PRODUCTION

Combines ALL repair operators into an Iterated Local Search pipeline:
  Phase 1: Multi-start init + iterative (deterministic repair + gene-level LS)
  Phase 2: ILS loop — perturb → repair → gene-LS → RepairEngine → accept
           Periodic group & instructor rescheduling on stagnation.
           Warm + fresh diversification restart on prolonged stagnation.

Generates thesis-ready ILS diagnostic plots in ``output/<run>/plots/ils/``:
  - Hard & soft convergence with improvement / restart markers
  - Per-constraint breakdown (stacked area + line)
  - Repair operator efficacy (det-repair, gene-LS, RepairEngine)
  - Improvement waterfall chart
  - Search dynamics (candidate vs best)
  - Perturbation size over iterations
  - Rescheduling event impact (before / after)
  - Wall-time profiling (per-iter + cumulative)
  - 6-panel diagnostic dashboard

Best result so far: Hard=75 (seed=42, 300 ILS iters, ~53 min)

Usage:
    python runs/ga_06_ultimate.py
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import UltimateExperiment

# ── PRODUCTION CONFIGURATION ─────────────────────────────────────────

# Reproducibility
SEED = 42

# GA Core Parameters (used by BaseExperiment for toolbox setup)
POP_SIZE = 1  # ILS is single-solution; pop_size=1 for init
NGEN = 300  # Used as upper bound for ILS iterations tracking
CXPB = 0.15  # Not used by ILS, but required by BaseExperiment
MUTPB = 0.4  # Not used by ILS, but required by BaseExperiment
FITNESS_WEIGHTS = (-1.0, -1.0)  # (hard, soft) minimize both

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated: output/ga_06_ultimate/<timestamp>

# Time Configuration
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

# Logging
LOG_INTERVAL = 1  # Log every ILS iteration (thesis granularity)
VERBOSE = True

# ── Mode F Specific: ILS Pipeline ────────────────────────────────────

# Phase 1: Multi-start initialisation
N_STARTS = 5  # Number of random starts
REPAIR_LS_ROUNDS = 5  # det-repair+gene-LS rounds per start

# Phase 2: Iterated Local Search
ILS_ITERATIONS = 300  # Main ILS iterations
PERTURB_FRAC = 0.15  # Perturb 15% of best Hard as n_perturb
PERTURB_MIN = 10  # Min genes to perturb

# Diversification
STAGNATION_RESTART = 30  # Restart after this many stale iterations

# RepairEngine (used in each ILS iteration)
ENGINE_MAX_STEPS = 20
ENGINE_BUDGET_MS = 500.0
ENGINE_MAX_CANDIDATES = 50
ENGINE_POLICY = "epsilon_greedy"
ENGINE_EPSILON = 0.15

# Deterministic repair
DETERMINISTIC_MAX_ITERS = 3

# Gene-level local search per ILS iteration
LS_MAX_ITERS = 12


def main() -> None:
    """Run Mode F: Ultimate, full-arsenal ILS experiment."""
    # Configure root logger for full console output at INFO level
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    exp = UltimateExperiment(
        # Core (for BaseExperiment compatibility)
        seed=SEED,
        pop_size=POP_SIZE,
        ngen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        fitness_weights=FITNESS_WEIGHTS,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        opening_time=OPENING_TIME,
        closing_time=CLOSING_TIME,
        closed_days=CLOSED_DAYS,
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
        # Phase 1: Multi-start
        n_starts=N_STARTS,
        repair_ls_rounds=REPAIR_LS_ROUNDS,
        # Phase 2: ILS
        ils_iterations=ILS_ITERATIONS,
        perturb_frac=PERTURB_FRAC,
        perturb_min=PERTURB_MIN,
        # Diversification
        stagnation_restart=STAGNATION_RESTART,
        # RepairEngine
        engine_max_steps=ENGINE_MAX_STEPS,
        engine_budget_ms=ENGINE_BUDGET_MS,
        engine_max_candidates=ENGINE_MAX_CANDIDATES,
        engine_policy=ENGINE_POLICY,
        engine_epsilon=ENGINE_EPSILON,
        # Deterministic repair
        deterministic_max_iters=DETERMINISTIC_MAX_ITERS,
        # Gene-level LS
        ls_max_iters=LS_MAX_ITERS,
    )
    result = exp.run()

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Hard: {result['results']['final_min_hard']:.0f}")
    print(f"  Soft: {result['results']['final_min_soft']:.0f}")
    print(f"  Time: {result['results']['elapsed_time']:.1f}s")
    print(f"  ILS improvements: {result['results']['ils_improvements']}")
    print(f"  Restarts: {result['results']['restarts']}")
    print(f"  Output: {exp.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
