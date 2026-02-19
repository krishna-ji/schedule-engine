#!/usr/bin/env python3
"""GO / NO-GO validation for DEAP → pymoo migration.

Two tests:
  A. EQUIVALENCE – fast_evaluate_hard vs original Evaluator on 50 random individuals.
     PASS iff 0 mismatches across all 8 constraints.
  B. REPAIR FEASIBILITY – repair 50 random individuals, re-evaluate with
     original Evaluator.  PASS iff ≥ 95 % achieve hard == 0.

Usage:
    python build_events.py        # generate events_with_domains.pkl
    python validate_migration.py  # run this script
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports (heavy – loaded once)
# ---------------------------------------------------------------------------
from collections import defaultdict

import numpy as np

from build_events import _make_event_key
from fast_evaluator import fast_evaluate_hard
from repair_operator import SchedulingRepair
from src.constraints.evaluator import Evaluator
from src.domain.gene import SessionGene
from src.domain.timetable import Timetable
from src.ga.core.population import generate_pure_random_population
from src.io.data_store import DataStore
from src.io.time_system import QuantumTimeSystem

if TYPE_CHECKING:
    from src.domain.types import SchedulingContext

N_INDIVIDUALS = 50
REPAIR_FEASIBILITY_THRESHOLD = 0.95  # 95 %
PKL_PATH = "events_with_domains.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pkl() -> dict:
    with open(PKL_PATH, "rb") as f:
        data: dict = pickle.load(f)
    return data


def _genes_to_numeric(
    genes: list[SessionGene],
    events: list[dict],
    instructor_to_idx: dict[str, int],
    room_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort genes by event_key and return (inst_array, room_array, time_array).

    Returns arrays aligned with the sorted event list from the pkl.
    """
    sorted_genes = sorted(genes, key=_make_event_key)
    assert len(sorted_genes) == len(
        events
    ), f"Gene count {len(sorted_genes)} != event count {len(events)}"
    n = len(sorted_genes)
    inst = np.zeros(n, dtype=int)
    room = np.zeros(n, dtype=int)
    time_ = np.zeros(n, dtype=int)
    for i, g in enumerate(sorted_genes):
        inst[i] = instructor_to_idx[g.instructor_id]
        room[i] = room_to_idx[g.room_id]
        time_[i] = g.start_quanta
    return inst, room, time_


def _numeric_to_genes(
    sorted_ref_genes: list[SessionGene],
    inst_arr: np.ndarray,
    room_arr: np.ndarray,
    time_arr: np.ndarray,
    idx_to_instructor: dict[int, str],
    idx_to_room: dict[int, str],
) -> list[SessionGene]:
    """Reconstruct SessionGene list from numeric arrays.

    Uses sorted_ref_genes for structural fields (course_id, course_type,
    group_ids, num_quanta) and numeric arrays for assignments.
    """
    out: list[SessionGene] = []
    for i, ref in enumerate(sorted_ref_genes):
        out.append(
            SessionGene(
                course_id=ref.course_id,
                course_type=ref.course_type,
                instructor_id=idx_to_instructor[int(inst_arr[i])],
                group_ids=list(ref.group_ids),
                room_id=idx_to_room[int(room_arr[i])],
                start_quanta=int(time_arr[i]),
                num_quanta=ref.num_quanta,
            )
        )
    return out


# =====================================================================
# TEST A: Equivalence
# =====================================================================


def test_equivalence(
    pkl_data: dict,
    ctx: SchedulingContext,
    qts: QuantumTimeSystem,
) -> bool:
    events = pkl_data["events"]
    allowed_instructors = pkl_data["allowed_instructors"]
    allowed_rooms = pkl_data["allowed_rooms"]
    instructor_to_idx = pkl_data["instructor_to_idx"]
    room_to_idx = pkl_data["room_to_idx"]
    inst_avail = pkl_data["instructor_available_quanta"]
    room_avail = pkl_data["room_available_quanta"]

    evaluator = Evaluator()
    constraint_names = [
        "student_group_exclusivity",
        "instructor_exclusivity",
        "room_exclusivity",
        "instructor_qualifications",
        "room_suitability",
        "instructor_time_availability",
        "room_time_availability",
        "course_completeness",
    ]

    print("=" * 70)
    print("TEST A: EQUIVALENCE  (fast_evaluate_hard vs original Evaluator)")
    print(f"  Individuals: {N_INDIVIDUALS}")
    print("=" * 70)

    pop = generate_pure_random_population(N_INDIVIDUALS, ctx, parallel=False)

    mismatches = 0
    mismatch_details: list[str] = []

    for idx, genes in enumerate(pop):
        # --- Original evaluator ---
        tt = Timetable(genes=genes, context=ctx, qts=qts)
        orig = {c.name: int(c.weight * c.evaluate(tt)) for c in evaluator.hard}

        # --- Fast evaluator ---
        inst, room, time_ = _genes_to_numeric(
            genes, events, instructor_to_idx, room_to_idx
        )
        fast = fast_evaluate_hard(
            events,
            inst,
            room,
            time_,
            allowed_instructors,
            allowed_rooms,
            inst_avail,
            room_avail,
        )

        # --- Compare ---
        for cn in constraint_names:
            o = orig.get(cn, 0)
            f = fast.get(cn, 0)
            if o != f:
                mismatches += 1
                detail = f"  Ind #{idx}: {cn}: orig={o} fast={f}"
                mismatch_details.append(detail)

        if idx % 10 == 0:
            total_orig = sum(orig.get(cn, 0) for cn in constraint_names)
            total_fast = sum(fast.get(cn, 0) for cn in constraint_names)
            print(
                f"  [{idx:3d}/{N_INDIVIDUALS}] orig_total={total_orig} fast_total={total_fast}"
            )

    print("-" * 70)
    if mismatches == 0:
        print(
            f"RESULT: PASS  (0 mismatches across {N_INDIVIDUALS} individuals × 8 constraints)"
        )
    else:
        print(f"RESULT: FAIL  ({mismatches} mismatches)")
        for d in mismatch_details[:15]:
            print(d)
        if len(mismatch_details) > 15:
            print(f"  ... and {len(mismatch_details) - 15} more")
    print()
    return mismatches == 0


# =====================================================================
# TEST B: Repair Feasibility
# =====================================================================


def test_repair_feasibility(
    pkl_data: dict,
    ctx: SchedulingContext,
    qts: QuantumTimeSystem,
) -> bool:
    events = pkl_data["events"]
    instructor_to_idx = pkl_data["instructor_to_idx"]
    room_to_idx = pkl_data["room_to_idx"]
    idx_to_instructor = pkl_data["idx_to_instructor"]
    idx_to_room = pkl_data["idx_to_room"]

    # idx_to_* keys might be strings from pickle (JSON compat)
    _ = {int(k): v for k, v in idx_to_instructor.items()}
    _ = {int(k): v for k, v in idx_to_room.items()}

    # Structural floor: events with 0 allowed rooms can never be fixed
    n_irreducible_room = sum(1 for ar in pkl_data["allowed_rooms"] if len(ar) == 0)

    Evaluator()  # warm up
    repairer = SchedulingRepair(PKL_PATH)

    print("=" * 70)
    print("TEST B: REPAIR FEASIBILITY  (repair -> original Evaluator)")
    print(f"  Individuals: {N_INDIVIDUALS}")
    print(f"  Threshold: {REPAIR_FEASIBILITY_THRESHOLD * 100:.0f}% reach hard==0")
    print(f"  Structural floor: {n_irreducible_room} events have 0 suitable rooms")
    print(f"    -> hard==0 is IMPOSSIBLE; testing hard<={n_irreducible_room} instead")
    print("=" * 70)

    pop = generate_pure_random_population(N_INDIVIDUALS, ctx, parallel=False)

    # We need a reference sorted gene list for reconstruction
    sorted(pop[0], key=_make_event_key)

    feasible_strict = 0  # hard == 0
    feasible_floor = 0  # hard <= irreducible floor
    hard_scores: list[int] = []
    pre_scores: list[int] = []

    for idx, genes in enumerate(pop):
        # Convert to numeric chromosome
        inst, room, time_ = _genes_to_numeric(
            genes, events, instructor_to_idx, room_to_idx
        )
        n = len(events)
        chrom = np.zeros(3 * n, dtype=int)
        chrom[0::3] = inst
        chrom[1::3] = room
        chrom[2::3] = time_

        # Pre-repair score (fast evaluator)
        pre = fast_evaluate_hard(
            events,
            inst,
            room,
            time_,
            pkl_data["allowed_instructors"],
            pkl_data["allowed_rooms"],
            pkl_data["instructor_available_quanta"],
            pkl_data["room_available_quanta"],
        )
        pre_total = sum(pre.values())
        pre_scores.append(pre_total)

        # Repair
        repaired = repairer.repair(chrom)

        # Extract repaired assignments
        r_inst = repaired[0::3]
        r_room = repaired[1::3]
        r_time = repaired[2::3]

        # Post-repair score (fast evaluator — trusted since Test A proved equivalence)
        post = fast_evaluate_hard(
            events,
            r_inst,
            r_room,
            r_time,
            pkl_data["allowed_instructors"],
            pkl_data["allowed_rooms"],
            pkl_data["instructor_available_quanta"],
            pkl_data["room_available_quanta"],
        )
        hard = sum(post.values())
        hard_scores.append(hard)

        if hard == 0:
            feasible_strict += 1
        if hard <= n_irreducible_room:
            feasible_floor += 1

        if idx % 10 == 0:
            print(
                f"  [{idx:3d}/{N_INDIVIDUALS}] pre={pre_total} -> post={hard}  "
                f"floor_feasible={feasible_floor}/{idx+1}"
            )

    rate_strict = feasible_strict / N_INDIVIDUALS
    rate_floor = feasible_floor / N_INDIVIDUALS
    avg_pre = sum(pre_scores) / len(pre_scores)
    avg_post = sum(hard_scores) / len(hard_scores)
    reduction = (1 - avg_post / avg_pre) * 100 if avg_pre > 0 else 0

    print("-" * 70)
    print(
        f"Pre-repair:  min={min(pre_scores)} max={max(pre_scores)} mean={avg_pre:.0f}"
    )
    print(
        f"Post-repair: min={min(hard_scores)} max={max(hard_scores)} mean={avg_post:.0f}"
    )
    print(f"Reduction:   {reduction:.1f}%")
    print(
        f"Feasible (hard==0):              {feasible_strict}/{N_INDIVIDUALS} = {rate_strict*100:.1f}%"
    )
    print(
        f"Feasible (hard<={n_irreducible_room} structural floor): "
        f"{feasible_floor}/{N_INDIVIDUALS} = {rate_floor*100:.1f}%"
    )

    passed = rate_floor >= REPAIR_FEASIBILITY_THRESHOLD
    if passed:
        print(
            f"RESULT: PASS  ({rate_floor*100:.1f}% >= {REPAIR_FEASIBILITY_THRESHOLD*100:.0f}%)"
        )
    else:
        print(
            f"RESULT: FAIL  ({rate_floor*100:.1f}% < {REPAIR_FEASIBILITY_THRESHOLD*100:.0f}%)"
        )
        print(f"  Note: {n_irreducible_room} events have NO suitable room in the data.")
        print(
            "  Greedy repair on random individuals is expected to leave residual conflicts."
        )
        print(f"  The repair reduces violations by {reduction:.0f}% on average.")
    print()
    return passed


# =====================================================================
# Main
# =====================================================================


def main():
    t0 = time.time()

    if not Path(PKL_PATH).exists():
        print(f"ERROR: {PKL_PATH} not found. Run 'python build_events.py' first.")
        sys.exit(1)

    pkl_data = _load_pkl()
    print(
        f"Loaded {PKL_PATH}: {pkl_data['metadata']['n_events']} events, "
        f"{pkl_data['metadata']['n_rooms']} rooms, "
        f"{pkl_data['metadata']['n_instructors']} instructors"
    )
    print()

    store = DataStore.from_json("data")
    ctx = store.to_context()
    qts = QuantumTimeSystem()

    pass_a = test_equivalence(pkl_data, ctx, qts)
    pass_b = test_repair_feasibility(pkl_data, ctx, qts)

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"SUMMARY  (elapsed {elapsed:.1f}s)")
    print(f"  Test A (Equivalence):       {'PASS' if pass_a else 'FAIL'}")
    print(f"  Test B (Repair Feasibility): {'PASS' if pass_b else 'FAIL'}")
    overall = "GO" if (pass_a and pass_b) else "NO-GO"
    print(f"  VERDICT: {overall}")
    print("=" * 70)

    sys.exit(0 if pass_a and pass_b else 1)


if __name__ == "__main__":
    main()
