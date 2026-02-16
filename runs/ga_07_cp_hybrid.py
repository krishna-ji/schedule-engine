#!/usr/bin/env python3
"""
GA CP-Hybrid: Pure GA + CP-SAT Distributed Constraint Repair (Mode G)

A proper genetic algorithm where the ONLY repair mechanism is
OR-Tools CP-SAT with a distributed constraint programming approach.

Pipeline:
  Phase 1: Create initial population + CP-SAT repair each individual
  Phase 2: GA loop -- tournament select -> crossover -> mutate ->
           CP-SAT repair on violated genes -> evaluate -> NSGA-II select
           Every cp_full_interval generations, run full decomposed CP
           pipeline (bridge global -> per-cluster) on best individual.
  Phase 3: Final full CP-SAT polish on best individual

No heuristic repairs, no deterministic repair, no gene-level local
search, no RepairEngine.  CP-SAT handles both hard constraints
(as hard CP constraints) and soft constraints (compactness objectives).

Target: 0 hard violations + minimised soft penalties

Usage:
    python runs/ga_07_cp_hybrid.py
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.modes.cp_hybrid import CPHybridExperiment

# -- CONFIGURATION --------------------------------------------------------

# Reproducibility
SEED = 42

# GA Core
POP_SIZE = 20  # Real population-based GA
NGEN = 50  # Generations (each includes CP repair)
CXPB = 0.7  # Crossover probability
MUTPB = 0.3  # Mutation probability
FITNESS_WEIGHTS = (-1.0, -1.0)

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = None  # Auto-generated: output/ga_07_cp_hybrid/<timestamp>

# Time Configuration
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]

# Logging
LOG_INTERVAL = 1
VERBOSE = True

# -- CP-SAT Parameters ---------------------------------------------------

CP_TIMEOUT = 10  # seconds per quick repair (violated genes only)
CP_TIMEOUT_FULL = 60  # seconds for full pipeline (global + cluster)
CP_NUM_WORKERS = 8  # CP-SAT threads per model
CP_MIN_SHARED_COURSES = 2  # clustering threshold
CP_FULL_INTERVAL = 10  # full pipeline every N generations
CP_SOFT_OBJECTIVE = True  # include soft constraints in CP objective

# GA Selection
TOURNAMENT_SIZE = 3


def main() -> None:
    """Run Mode G: GA + CP-SAT hybrid experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    exp = CPHybridExperiment(
        # Core GA
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
        # CP-SAT
        cp_timeout=CP_TIMEOUT,
        cp_timeout_full=CP_TIMEOUT_FULL,
        cp_num_workers=CP_NUM_WORKERS,
        cp_min_shared_courses=CP_MIN_SHARED_COURSES,
        cp_full_interval=CP_FULL_INTERVAL,
        cp_soft_objective=CP_SOFT_OBJECTIVE,
        tournament_size=TOURNAMENT_SIZE,
    )
    result = exp.run()

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULT -- GA + CP-SAT HYBRID")
    print("=" * 60)
    print(f"  Hard: {result['results']['final_min_hard']:.0f}")
    print(f"  Soft: {result['results']['final_min_soft']:.0f}")
    print(f"  Time: {result['results']['elapsed_time']:.1f}s")
    print(
        f"  CP quick repairs: {result['results']['cp_quick_repairs']}"
        f" ({result['results']['cp_quick_success']} successful)"
    )
    print(
        f"  CP full repairs:  {result['results']['cp_full_repairs']}"
        f" ({result['results']['cp_full_success']} successful)"
    )
    print(f"  Output: {exp.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
