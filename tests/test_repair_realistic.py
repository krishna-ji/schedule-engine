#!/usr/bin/env python3
"""Realistic repair test: post-crossover perturbation, not pure random noise.

Simulates GA offspring by:
  A) Building a "base" individual using the existing DEAP generator
  B) Applying local perturbations (5–20 gene mutations) K=200 times
  C) Repairing each offspring and measuring hard constraint violations

Pass criteria:
  1. >=95% of offspring reach the structural floor (<=24 violations --
     the 24 events with 0 suitable rooms that cannot be fixed).
  2. Median hard-violation reduction >=90%.
"""

from __future__ import annotations

import pickle
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.ga.core.population import generate_pure_random_population
from src.io.data_store import DataStore
from src.pipeline.build_events import _make_event_key
from src.pipeline.fast_evaluator import fast_evaluate_hard
from src.pipeline.repair_operator import SchedulingRepair

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_OFFSPRING = 200
PERTURBATION_RANGE = (5, 20)  # mutate 5–20 genes per offspring
FLOOR_THRESHOLD = 0.95  # >=95% reach floor
REDUCTION_THRESHOLD = 0.90  # median reduction >=90%
PKL_PATH = ".cache/events_with_domains.pkl"

random.seed(42)
np.random.seed(42)


def genes_to_chromosome(genes, events, instructor_to_idx, room_to_idx) -> np.ndarray:
    """Convert SessionGene list -> 3xE interleaved chromosome."""
    sorted_genes = sorted(genes, key=_make_event_key)
    n = len(events)
    chrom = np.zeros(3 * n, dtype=int)
    for i, g in enumerate(sorted_genes):
        chrom[3 * i] = instructor_to_idx[g.instructor_id]
        chrom[3 * i + 1] = room_to_idx[g.room_id]
        chrom[3 * i + 2] = g.start_quanta
    return chrom


def perturb_chromosome(
    base: np.ndarray,
    n_events: int,
    allowed_instructors: list[list[int]],
    allowed_rooms: list[list[int]],
    allowed_starts: list[list[int]],
    n_mutations: int,
) -> np.ndarray:
    """Create an offspring by mutating n_mutations random genes."""
    child = base.copy()
    events_to_mutate = random.sample(range(n_events), min(n_mutations, n_events))

    for e in events_to_mutate:
        # Randomly choose what to mutate: instructor, room, or time
        what = random.choice(["inst", "room", "time", "inst+time", "room+time"])

        ai = allowed_instructors[e]
        ar = allowed_rooms[e]
        at = allowed_starts[e]

        if "inst" in what and ai:
            child[3 * e] = random.choice(ai)
        if "room" in what and ar:
            child[3 * e + 1] = random.choice(ar)
        if "time" in what and at:
            child[3 * e + 2] = random.choice(at)

    return child


def evaluate_chromosome(
    chrom: np.ndarray,
    events,
    allowed_instructors,
    allowed_rooms,
    inst_avail,
    room_avail,
) -> int:
    """Evaluate total hard violations for a chromosome."""
    inst = chrom[0::3]
    room = chrom[1::3]
    time_ = chrom[2::3]
    result = fast_evaluate_hard(
        events,
        inst,
        room,
        time_,
        allowed_instructors,
        allowed_rooms,
        inst_avail,
        room_avail,
    )
    return sum(result.values())


def main():
    t0 = time.time()

    if not Path(PKL_PATH).exists():
        print(f"ERROR: {PKL_PATH} not found. Run 'python build_events.py' first.")
        sys.exit(1)

    with open(PKL_PATH, "rb") as f:
        pkl = pickle.load(f)

    events = pkl["events"]
    n_events = len(events)
    allowed_instructors = pkl["allowed_instructors"]
    allowed_rooms = pkl["allowed_rooms"]
    allowed_starts = pkl["allowed_starts"]
    instructor_to_idx = pkl["instructor_to_idx"]
    room_to_idx = pkl["room_to_idx"]
    inst_avail = pkl["instructor_available_quanta"]
    room_avail = pkl["room_available_quanta"]

    structural_floor = sum(1 for ar in allowed_rooms if len(ar) == 0)

    print("=" * 70)
    print("REALISTIC REPAIR TEST")
    print(f"  Events: {n_events}")
    print(f"  Offspring: {N_OFFSPRING}")
    print(
        f"  Perturbation: {PERTURBATION_RANGE[0]}–{PERTURBATION_RANGE[1]} genes per offspring"
    )
    print(f"  Structural floor: {structural_floor} (events with 0 suitable rooms)")
    print("  Pass criteria:")
    print(f"    1. >={FLOOR_THRESHOLD * 100:.0f}% reach hard<={structural_floor}")
    print(f"    2. Median hard reduction >={REDUCTION_THRESHOLD * 100:.0f}%")
    print("=" * 70)

    # Step A: Generate base individual and repair it to get a "decent" starting point
    print("\nGenerating base individual...")
    store = DataStore.from_json("data")
    ctx = store.to_context()

    pop = generate_pure_random_population(3, ctx, parallel=False)
    best_base: np.ndarray | None = None
    best_base_score = float("inf")

    repairer = SchedulingRepair(PKL_PATH)

    for genes in pop:
        chrom = genes_to_chromosome(genes, events, instructor_to_idx, room_to_idx)
        repaired = repairer.repair(chrom)
        score = evaluate_chromosome(
            repaired,
            events,
            allowed_instructors,
            allowed_rooms,
            inst_avail,
            room_avail,
        )
        if score < best_base_score:
            best_base_score = score
            best_base = repaired
        print(f"  Base candidate: hard={score}")

    print(f"  Selected base with hard={best_base_score}")
    assert best_base is not None, "No base individual found"

    # Step B: Generate offspring via local perturbation
    print(f"\nGenerating {N_OFFSPRING} offspring via perturbation...")
    pre_scores: list[int] = []
    post_scores: list[int] = []
    repair_times: list[float] = []
    floor_feasible = 0

    for i in range(N_OFFSPRING):
        n_mut = random.randint(*PERTURBATION_RANGE)
        offspring = perturb_chromosome(
            best_base,
            n_events,
            allowed_instructors,
            allowed_rooms,
            allowed_starts,
            n_mut,
        )

        pre = evaluate_chromosome(
            offspring,
            events,
            allowed_instructors,
            allowed_rooms,
            inst_avail,
            room_avail,
        )
        pre_scores.append(pre)

        t_start = time.time()
        repaired = repairer.repair(offspring)
        t_end = time.time()
        repair_times.append(t_end - t_start)

        post = evaluate_chromosome(
            repaired,
            events,
            allowed_instructors,
            allowed_rooms,
            inst_avail,
            room_avail,
        )
        post_scores.append(post)

        if post <= structural_floor:
            floor_feasible += 1

        if i % 50 == 0:
            print(
                f"  [{i:3d}/{N_OFFSPRING}] mutations={n_mut} pre={pre} -> post={post} "
                f"floor_ok={floor_feasible}/{i + 1} ({(t_end - t_start) * 1000:.0f}ms)"
            )

    # Step C: Compute statistics
    pre_arr = np.array(pre_scores)
    post_arr = np.array(post_scores)
    reductions = np.where(pre_arr > 0, 1.0 - post_arr / pre_arr, 0.0)

    median_reduction = float(np.median(reductions))
    floor_rate = floor_feasible / N_OFFSPRING
    mean_repair_ms = np.mean(repair_times) * 1000

    print()
    print("-" * 70)
    print("DISTRIBUTION STATISTICS")
    print("-" * 70)
    print("Pre-repair  hard violations:")
    print(
        f"  min={int(pre_arr.min())}  median={int(np.median(pre_arr))}  "
        f"p95={int(np.percentile(pre_arr, 95))}  max={int(pre_arr.max())}  "
        f"mean={pre_arr.mean():.1f}"
    )
    print("Post-repair hard violations:")
    print(
        f"  min={int(post_arr.min())}  median={int(np.median(post_arr))}  "
        f"p95={int(np.percentile(post_arr, 95))}  max={int(post_arr.max())}  "
        f"mean={post_arr.mean():.1f}"
    )
    print("Reduction %:")
    print(
        f"  min={reductions.min() * 100:.1f}%  median={median_reduction * 100:.1f}%  "
        f"p95={np.percentile(reductions, 95) * 100:.1f}%  max={reductions.max() * 100:.1f}%"
    )
    print("Repair timing:")
    print(
        f"  mean={mean_repair_ms:.1f}ms  max={max(repair_times) * 1000:.1f}ms  "
        f"total={sum(repair_times):.1f}s for {N_OFFSPRING} individuals"
    )
    print()
    print(
        f"Floor-feasible (hard<={structural_floor}): "
        f"{floor_feasible}/{N_OFFSPRING} = {floor_rate * 100:.1f}%"
    )
    print(f"Median reduction: {median_reduction * 100:.1f}%")

    # Verdicts
    print()
    print("=" * 70)
    pass_floor = floor_rate >= FLOOR_THRESHOLD
    pass_reduction = median_reduction >= REDUCTION_THRESHOLD

    print(
        f"Criterion 1 (>={FLOOR_THRESHOLD * 100:.0f}% floor-feasible): "
        f"{'PASS' if pass_floor else 'FAIL'} ({floor_rate * 100:.1f}%)"
    )
    print(
        f"Criterion 2 (median reduction >={REDUCTION_THRESHOLD * 100:.0f}%): "
        f"{'PASS' if pass_reduction else 'FAIL'} ({median_reduction * 100:.1f}%)"
    )

    overall = pass_floor and pass_reduction
    print(f"RESULT: {'PASS' if overall else 'FAIL'}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print("=" * 70)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
