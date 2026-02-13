#!/usr/bin/env python3
"""
GA Ultimate: ILS + Full Repair Arsenal (Mode F)

Combines ALL repair operators into an Iterated Local Search pipeline:
  Phase 1: Multi-start init + iterative (deterministic repair + gene-level LS)
  Phase 2: ILS loop — perturb → repair → gene-LS → RepairEngine → accept

Usage:
    python runs/ga_06_ultimate.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schedule_engine.experiments import UltimateExperiment

# ── PRODUCTION CONFIGURATION ─────────────────────────────────────────

# Reproducibility
SEED = 42

# GA Core Parameters (used by BaseExperiment for toolbox setup)
POP_SIZE = 1  # ILS is single-solution; pop_size=1 for init
NGEN = 500  # Used as upper bound for ILS iterations tracking
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
LOG_INTERVAL = 10  # Log every 10 ILS iterations
VERBOSE = True

# ── Mode F Specific: ILS Pipeline ────────────────────────────────────

# Phase 1: Multi-start initialisation
N_STARTS = 5  # Number of random starts
REPAIR_LS_ROUNDS = 5  # det-repair+gene-LS rounds per start

# Phase 2: Iterated Local Search
ILS_ITERATIONS = 200  # Main ILS iterations
PERTURB_FRAC = 0.15  # Perturb 15% of best Hard as n_perturb
PERTURB_MIN = 10  # Min genes to perturb

# Simulated Annealing acceptance
SA_START_TEMP = 8.0  # Starting temperature
SA_END_TEMP = 0.3  # Final temperature

# Diversification
STAGNATION_RESTART = 50  # Restart after this many stale iterations

# RepairEngine (used in each ILS iteration)
ENGINE_MAX_STEPS = 20
ENGINE_BUDGET_MS = 500.0
ENGINE_MAX_CANDIDATES = 40
ENGINE_POLICY = "epsilon_greedy"
ENGINE_EPSILON = 0.15

# Deterministic repair
DETERMINISTIC_MAX_ITERS = 2

# Gene-level local search per ILS iteration
LS_MAX_ITERS = 10


def main() -> None:
    """Run Mode F: Ultimate, full-arsenal ILS experiment."""
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
        # Simulated Annealing
        sa_start_temp=SA_START_TEMP,
        sa_end_temp=SA_END_TEMP,
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
    exp.run()


if __name__ == "__main__":
    main()
