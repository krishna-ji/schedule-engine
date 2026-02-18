#!/usr/bin/env python3
"""
Decomposed GA + CP-SAT Scheduler (Mode H)

Supergroup-decomposed architecture:
  1. Clusters programmes by shared courses (ARCH, CIVIL, IT, MECH, MASTERS)
  2. Runs GA with full population over all clusters
  3. Every N generations, CP-SAT fixes hard constraint violations per-cluster
  4. Final CP-SAT optimization pass with soft objectives

No RL, no heuristic repair, no LNS — just GA diversity + CP-SAT precision.

Usage:
    python runs/ga_08_decomposed_cp.py
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# -- CONFIGURATION --------------------------------------------------------

SEED = 42

# GA Core
POP_SIZE = 50
NGEN = 200
CXPB = 0.7
MUTPB = 0.4

# CP-SAT
CP_INTERVAL = 5  # CP-SAT polish every N generations
CP_ELITE_COUNT = 3  # Number of elites to CP-optimize per interval
CP_TIMEOUT_BRIDGE = 30  # seconds for bridge gene solving
CP_TIMEOUT_CLUSTER = 20  # seconds per cluster solving
CP_TIMEOUT_CHUNK = 20  # seconds per chunk within large cluster
CP_NUM_WORKERS = 4  # CP-SAT internal parallelism
CP_SOFT_OBJECTIVE = False  # Include soft constraints in CP objective
CP_FULL_INTERVAL = 20  # Full (non-fix-only) CP every N gens (0=disable)
CP_MAX_CHUNK_SIZE = 60  # Max genes per CP-SAT chunk

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    """Run Mode H: Decomposed GA + CP-SAT."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy loggers
    logging.getLogger("src.ga.core.population").setLevel(logging.WARNING)

    # Load data
    from src.ga.run_helpers import load_data

    print("=" * 60)
    print("DECOMPOSED GA + CP-SAT SCHEDULER (Mode H)")
    print("=" * 60)
    print(f"  Pop size:    {POP_SIZE}")
    print(f"  Generations: {NGEN}")
    print(f"  CP interval: every {CP_INTERVAL} gens")
    print(f"  CP full opt: every {CP_FULL_INTERVAL} gens")
    print(f"  CP chunk:    max {CP_MAX_CHUNK_SIZE} genes")
    print(f"  Seed:        {SEED}")
    print()

    data = load_data(data_dir=DATA_DIR)
    ctx = data.to_context()

    print(f"  Courses:     {len(ctx.courses)}")
    print(f"  Groups:      {len(ctx.groups)}")
    print(f"  Instructors: {len(ctx.instructors)}")
    print(f"  Rooms:       {len(ctx.rooms)}")
    print(f"  Timeslots:   {len(ctx.available_quanta)}")
    print()

    # Build and run decomposed scheduler
    from src.ga.decomposed import DecomposedScheduler
    from src.ga.decomposed.coordinator import DecomposedConfig

    config = DecomposedConfig(
        pop_size=POP_SIZE,
        generations=NGEN,
        crossover_prob=CXPB,
        mutation_prob=MUTPB,
        cp_interval=CP_INTERVAL,
        cp_elite_count=CP_ELITE_COUNT,
        cp_timeout_bridge=CP_TIMEOUT_BRIDGE,
        cp_timeout_cluster=CP_TIMEOUT_CLUSTER,
        cp_timeout_chunk=CP_TIMEOUT_CHUNK,
        cp_num_workers=CP_NUM_WORKERS,
        cp_soft_objective=CP_SOFT_OBJECTIVE,
        cp_full_interval=CP_FULL_INTERVAL,
        cp_max_chunk_size=CP_MAX_CHUNK_SIZE,
        seed=SEED,
    )

    scheduler = DecomposedScheduler(ctx, config=config)
    result = scheduler.run()

    # Final output
    print("\n" + "=" * 60)
    print("FINAL RESULT — DECOMPOSED GA + CP-SAT")
    print("=" * 60)
    print(f"  Hard violations: {result.best_hard:.0f}")
    print(f"  Soft penalty:    {result.best_soft:.1f}")
    print(f"  Generations:     {result.generations_run}")
    print(f"  Total time:      {result.total_time:.1f}s")
    print(f"  Clusters:        {len(result.cluster_info)}")
    for cid, info in sorted(result.cluster_info.items()):
        print(
            f"    {cid}: {', '.join(info['programmes'])} "
            f"({info['groups']} groups, {info['courses']} courses)"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
