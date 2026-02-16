#!/usr/bin/env python3
"""Production CP-SAT test with relaxed config."""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.modes.cp_hybrid import CPHybridExperiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)

# RELAXED CONFIG for initial test
exp = CPHybridExperiment(
    seed=42,
    pop_size=10,
    ngen=10,  # 10 generations
    cxpb=0.7,
    mutpb=0.3,
    fitness_weights=(-1.0, -1.0),
    data_dir=PROJECT_ROOT / "data",
    output_dir=None,
    opening_time="10:00",
    closing_time="17:00",
    closed_days=["Saturday"],
    log_interval=1,
    verbose=True,
    cp_timeout=15,  # Increased to 15s (was 5s)
    cp_timeout_full=90,  # Increased to 90s (was 30s)
    cp_num_workers=8,
    cp_min_shared_courses=2,
    cp_full_interval=5,  # Run full pipeline at gen 5
    cp_soft_objective=True,
    tournament_size=3,
)

print("\n" + "=" * 60)
print("RELAXED TEST: 10 individuals × 10 generations")
print("CP timeout: 15s quick, 90s full")
print("=" * 60 + "\n")

result = exp.run()

print("\n" + "=" * 60)
print("TEST RESULT")
print("=" * 60)
print(f"  Hard: {result['results']['final_min_hard']:.0f}")
print(f"  Soft: {result['results']['final_min_soft']:.0f}")
print(f"  Time: {result['results']['elapsed_time']:.1f}s")
print(
    f"  CP quick: {result['results']['cp_quick_repairs']} ({result['results']['cp_quick_success']} ok)"
)
print(
    f"  CP full:  {result['results']['cp_full_repairs']} ({result['results']['cp_full_success']} ok)"
)
print("=" * 60)
