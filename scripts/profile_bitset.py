#!/usr/bin/env python3
"""Micro-benchmark: profile BitsetSchedulingRepair.repair() on a single individual.

Usage:
    python scripts/profile_bitset.py
"""
from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

PKL_PATH = PROJECT_ROOT / ".cache" / "events_with_domains.pkl"


def main() -> None:
    if not PKL_PATH.exists():
        print(f"ERROR: {PKL_PATH} not found. Run the pipeline first.")
        sys.exit(1)

    from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

    repairer = BitsetSchedulingRepair(str(PKL_PATH))
    rng = np.random.default_rng(42)

    # Build a random chromosome (3*E interleaved: inst, room, time)
    E = repairer.n_events
    X = np.zeros(3 * E, dtype=int)
    inst = X[0::3]
    room = X[1::3]
    time = X[2::3]
    for e in range(E):
        ai = repairer.allowed_instructors[e]
        ar = repairer.allowed_rooms[e]
        at = repairer.allowed_starts[e]
        inst[e] = rng.choice(ai) if ai else 0
        room[e] = rng.choice(ar) if ar else 0
        time[e] = rng.choice(at) if at else 0

    print(f"Events: {E}")
    print(f"Chromosome length: {len(X)}")
    print(
        f"Rooms: {repairer.n_rooms}, Instructors: {repairer.n_instructors}, Groups: {repairer.n_groups}"
    )
    print()

    # --- Single repair call profiled ---
    print("=" * 70)
    print("PROFILE: single repairer.repair(X) call")
    print("=" * 70)

    profiler = cProfile.Profile()
    profiler.enable()
    X_repaired = repairer.repair(X, rng=np.random.default_rng(99))
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    print("\n--- Top 15 by tottime ---")
    stats.sort_stats("tottime")
    stats.print_stats(15)

    print("\n--- Top 15 by cumtime ---")
    stats.sort_stats("cumtime")
    stats.print_stats(15)

    # --- Multi-pass (simulating repair_iters=8) ---
    print("=" * 70)
    print("PROFILE: 8 repair passes (simulating memetic elite repair)")
    print("=" * 70)

    X2 = X.copy()
    profiler2 = cProfile.Profile()
    profiler2.enable()
    for p in range(8):
        r = np.random.default_rng([0, 0, p]) if p % 2 == 0 else None
        X_new = repairer.repair(X2, rng=r)
        if np.array_equal(X_new, X2):
            print(f"  Converged at pass {p}")
            break
        X2 = X_new
    profiler2.disable()

    stats2 = pstats.Stats(profiler2)
    stats2.strip_dirs()

    print("\n--- Top 15 by tottime (8 passes) ---")
    stats2.sort_stats("tottime")
    stats2.print_stats(15)


if __name__ == "__main__":
    main()
