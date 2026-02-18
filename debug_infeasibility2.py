#!/usr/bin/env python3
"""Diagnose WHY certain programme-semester chunks are INFEASIBLE even with 0 frozen genes.

This script:
1. Loads data + creates context
2. Creates a heuristic initial individual
3. Groups genes by (programme, semester) — same as cp_optimizer
4. For each INFEASIBLE chunk, checks:
   a. Per-gene: course_id, group_ids, num_quanta, #qualified instructors, #suitable rooms
   b. Per-group: total quanta needed vs 42 available
   c. Per-instructor bottleneck: whether part-time instructors block all options
   d. HC6 availability conflicts
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ga.core.population import create_individual
from src.ga.run_helpers import load_data
from src.utils.room_compatibility import (
    is_room_suitable_for_course,
    is_room_type_compatible,
)

DATA_DIR = PROJECT_ROOT / "data"

_SEM_RE = re.compile(r"^[A-Z]{2,5}(\d+)")


def _semester_key(gid: str) -> str:
    m = _SEM_RE.match(gid)
    return m.group(1) if m else "0"


def _programme_key(gid: str) -> str:
    m = re.match(r"^([A-Z]{2,5})", gid)
    return m.group(1) if m else gid


def main():
    data = load_data(data_dir=DATA_DIR)
    ctx = data.to_context()

    from src.io.time_system import QuantumTimeSystem

    qts = QuantumTimeSystem()  # uses default operating hours

    print(f"Available quanta: {len(ctx.available_quanta)}")
    print("Day ranges:")
    for d_name in qts.DAY_NAMES:
        d_off = qts.day_quanta_offset.get(d_name)
        d_cnt = qts.day_quanta_count.get(d_name, 0)
        if d_off is not None:
            print(
                f"  {d_name}: offset={d_off}, count={d_cnt}, range=[{d_off}, {d_off+d_cnt})"
            )
    print()

    # Create an individual to get the gene structure
    from src.ga.core.population import generate_hybrid_population
    from src.ga.run_helpers import setup_deap

    setup_deap()
    pop = generate_hybrid_population(1, ctx)
    genes = list(pop[0])

    # Pre-compute suitable rooms + qualified instructors (same as solver)
    suitable_rooms = {}
    type_compat_rooms = {}
    qual_instrs = {}

    for key, course in ctx.courses.items():
        req = str(getattr(course, "required_room_features", "lecture")).lower().strip()
        lab = getattr(course, "specific_lab_features", None)
        suitable = []
        compat = []
        for room in ctx.rooms.values():
            rt = str(getattr(room, "room_features", "lecture")).lower().strip()
            rf = getattr(room, "specific_features", None)
            if is_room_suitable_for_course(req, rt, lab, rf):
                suitable.append(room.room_id)
            elif is_room_type_compatible(req, rt):
                compat.append(room.room_id)
        suitable_rooms[key] = suitable
        type_compat_rooms[key] = suitable + compat
        qual_instrs[key] = list(course.qualified_instructor_ids)

    # Compute valid starts per duration (same as solver)
    avail_set = set(ctx.available_quanta)
    day_ranges = []
    for d_name in qts.DAY_NAMES:
        d_off = qts.day_quanta_offset.get(d_name)
        d_cnt = qts.day_quanta_count.get(d_name, 0)
        if d_off is not None and d_cnt > 0:
            day_ranges.append((d_off, d_off + d_cnt))

    def valid_starts(dur):
        result = []
        for sq in sorted(avail_set):
            if not all((sq + d) in avail_set for d in range(dur)):
                continue
            ok = False
            for ds, de in day_ranges:
                if ds <= sq < de and sq + dur <= de:
                    ok = True
                    break
            if not ok:
                continue
            result.append(sq)
        return result

    # Group genes by (programme, semester)
    chunks = defaultdict(list)
    for gi, g in enumerate(genes):
        if g.group_ids:
            gid = g.group_ids[0]
            prog = _programme_key(gid)
            sem = _semester_key(gid)
        else:
            prog, sem = "UNK", "0"
        chunks[(prog, sem)].append(gi)

    # Known failing chunks from the output
    failing_chunks = [
        ("BCE", "1"),
        ("BCT", "3"),
        ("BEI", "3"),
        ("BIE", "3"),
        ("BCT", "5"),
        ("BEI", "5"),
        ("BIE", "5"),
        ("BME", "5"),
        ("BCT", "8"),
        ("BEI", "8"),
    ]

    for prog, sem in sorted(chunks.keys()):
        chunk_indices = chunks[(prog, sem)]
        is_failing = (prog, sem) in failing_chunks
        label = f"{prog}-sem{sem}"

        if not is_failing:
            continue

        print(f"\n{'='*60}")
        print(f"CHUNK: {label} ({len(chunk_indices)} genes) — KNOWN INFEASIBLE")
        print(f"{'='*60}")

        # Analyze each gene
        group_quanta: dict[str, int] = defaultdict(
            int
        )  # group_id -> total quanta needed
        group_genes = defaultdict(list)  # group_id -> [(gi, course_id)]
        instr_demand = defaultdict(list)  # instructor_id -> [gi]

        for gi in chunk_indices:
            g = genes[gi]
            ckey = (g.course_id, g.course_type)
            dur = g.num_quanta
            sr = type_compat_rooms.get(ckey, []) or list(ctx.rooms.keys())
            qi = qual_instrs.get(ckey, []) or list(ctx.instructors.keys())
            vs = valid_starts(dur) or sorted(ctx.available_quanta)

            # Check HC6: how many qualified instructors have valid starts?
            full_time_instrs = 0
            part_time_ok = 0
            part_time_blocked = 0
            for iid in qi:
                instr = ctx.instructors.get(iid)
                if not instr:
                    continue
                if instr.is_full_time:
                    full_time_instrs += 1
                    instr_demand[iid].append(gi)
                else:
                    ok_starts = [
                        sq
                        for sq in vs
                        if all((sq + d) in instr.available_quanta for d in range(dur))
                    ]
                    if ok_starts:
                        part_time_ok += 1
                        instr_demand[iid].append(gi)
                    else:
                        part_time_blocked += 1

            usable_instrs = full_time_instrs + part_time_ok

            for gid in g.group_ids:
                group_quanta[gid] += dur
                group_genes[gid].append((gi, g.course_id))

            if usable_instrs == 0:
                print(
                    f"  *** GENE {gi}: {g.course_id} ({g.course_type}) — "
                    f"NO USABLE INSTRUCTORS! "
                    f"({len(qi)} qualified: {full_time_instrs} FT, "
                    f"{part_time_ok} PT-ok, {part_time_blocked} PT-blocked)"
                )
                print(
                    f"      groups={g.group_ids}, dur={dur}q, "
                    f"rooms={len(sr)}, valid_starts={len(vs)}"
                )
                for iid in qi:
                    instr = ctx.instructors.get(iid)
                    if instr and not instr.is_full_time:
                        ok_starts = [
                            sq
                            for sq in vs
                            if all(
                                (sq + d) in instr.available_quanta for d in range(dur)
                            )
                        ]
                        print(
                            f"      instr {iid}: avail_quanta={len(instr.available_quanta)}, "
                            f"valid_starts_for_course={len(ok_starts)}"
                        )
            elif usable_instrs <= 2:
                print(
                    f"  !! GENE {gi}: {g.course_id} ({g.course_type}) — "
                    f"only {usable_instrs} usable instructors"
                )
                print(
                    f"      groups={g.group_ids}, dur={dur}q, "
                    f"rooms={len(sr)}, valid_starts={len(vs)}"
                )

        # Check group capacity
        print("\n  Group capacity analysis (42 quanta available):")
        for gid in sorted(group_quanta.keys()):
            total = group_quanta[gid]
            count = len(group_genes[gid])
            pct = total / 42 * 100
            flag = (
                " *** OVERLOADED!"
                if total > 42
                else (" !! TIGHT" if total > 35 else "")
            )
            if total > 30 or flag:
                print(f"    {gid}: {count} genes, {total}q / 42q ({pct:.0f}%){flag}")

        # Check instructor bottleneck
        print("\n  Instructor demand:")
        for iid in sorted(instr_demand.keys()):
            demand = len(instr_demand[iid])
            instr = ctx.instructors.get(iid)
            if demand >= 3 or (instr and not instr.is_full_time and demand >= 2):
                ft = (
                    "FT"
                    if instr and instr.is_full_time
                    else f"PT({len(instr.available_quanta) if instr else 0}q)"
                )
                print(f"    {iid}: {demand} genes, {ft}")

    # Also report working chunks for comparison
    print(f"\n\n{'='*60}")
    print("WORKING CHUNKS (for comparison)")
    print(f"{'='*60}")
    working_chunks = [
        ("BAM", "1"),
        ("BAM", "3"),
        ("BME", "1"),
        ("BME", "3"),
        ("BCT", "1"),
        ("BEI", "1"),
        ("BIE", "1"),
    ]
    for prog, sem in working_chunks:
        if (prog, sem) not in chunks:
            continue
        chunk_indices = chunks[(prog, sem)]
        label = f"{prog}-sem{sem}"

        group_quanta_w: dict[str, int] = defaultdict(int)
        zero_instr_count = 0
        for gi in chunk_indices:
            g = genes[gi]
            ckey = (g.course_id, g.course_type)
            dur = g.num_quanta
            qi = qual_instrs.get(ckey, []) or list(ctx.instructors.keys())
            vs = valid_starts(dur)

            usable = 0
            for iid in qi:
                instr = ctx.instructors.get(iid)
                if not instr:
                    continue
                if instr.is_full_time:
                    usable += 1
                else:
                    ok_starts = [
                        sq
                        for sq in vs
                        if all((sq + d) in instr.available_quanta for d in range(dur))
                    ]
                    if ok_starts:
                        usable += 1
            if usable == 0:
                zero_instr_count += 1
            for gid in g.group_ids:
                group_quanta_w[gid] += dur

        max_load = max(group_quanta_w.values()) if group_quanta_w else 0
        print(
            f"  {label}: {len(chunk_indices)} genes, "
            f"max_group_load={max_load}/42q, "
            f"zero_instr_genes={zero_instr_count}"
        )


if __name__ == "__main__":
    main()
