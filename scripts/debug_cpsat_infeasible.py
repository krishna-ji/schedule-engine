#!/usr/bin/env python3
"""Debug why CP-SAT returns INFEASIBLE on full 549-gene solve.

Tests:
1. Domain analysis: any gene with empty valid-start / room / instructor domains?
2. Incremental: solve progressively larger subsets to find where infeasibility starts
3. Constraint ablation: which HC causes infeasibility?
"""

import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from src.ga.core.population import get_family_map_from_json
from src.ga.core.population_factory import PopulationFactory
from src.ga.repair.cp.solver import CPSATSolver
from src.io.data_store import DataStore

# Load data
data_dir = PROJECT_ROOT / "data"
store = DataStore.from_json(data_dir)
ctx = store.to_context()
family_map = get_family_map_from_json(str(data_dir / "Groups.json"))

# Create one individual
pf = PopulationFactory(ctx)
ind = pf.random_individual(conflict_aware=True)
print(f"Total genes: {len(ind)}")
print(f"Available quanta: {sorted(ctx.available_quanta)}")
print(f"Rooms: {len(ctx.rooms)}, Instructors: {len(ctx.instructors)}")
print()

# ======================================================================
# 1. Domain analysis
# ======================================================================
print("=" * 60)
print("1. DOMAIN ANALYSIS")
print("=" * 60)

solver = CPSATSolver(
    ctx, family_map, timeout_seconds=15, num_workers=8, soft_objective=False
)

empty_start_genes = []
empty_room_genes = []
empty_instr_genes = []
narrow_room_genes = []  # <= 2 rooms

for gi, g in enumerate(ind):
    ckey = (g.course_id, g.course_type)
    dur = g.num_quanta

    vs = solver._valid_starts(dur)
    sr = solver._suitable_rooms.get(ckey, [])
    tc = solver._type_compat_rooms.get(ckey, [])
    qi = solver._qual_instrs.get(ckey, [])

    if not vs:
        empty_start_genes.append((gi, g.course_id, g.course_type, dur))
    if not sr and not tc:
        empty_room_genes.append((gi, g.course_id, g.course_type))
    if not sr:
        narrow_room_genes.append((gi, g.course_id, g.course_type, len(tc)))
    if not qi:
        empty_instr_genes.append((gi, g.course_id, g.course_type))

print(f"Genes with empty valid_starts: {len(empty_start_genes)}")
for x in empty_start_genes[:5]:
    print(f"  gene {x[0]}: {x[1]}-{x[2]}, dur={x[3]}")

print(f"Genes with NO suitable/type-compat rooms: {len(empty_room_genes)}")
for x_room in empty_room_genes[:5]:
    print(f"  gene {x_room[0]}: {x_room[1]}-{x_room[2]}")

print(
    f"Genes with 0 suitable rooms (using type-compat fallback): {len(narrow_room_genes)}"
)
for x_narrow in narrow_room_genes[:10]:
    print(
        f"  gene {x_narrow[0]}: {x_narrow[1]}-{x_narrow[2]}, fallback_rooms={x_narrow[3]}"
    )

print(f"Genes with NO qualified instructors: {len(empty_instr_genes)}")
for x_instr in empty_instr_genes[:5]:
    print(f"  gene {x_instr[0]}: {x_instr[1]}-{x_instr[2]}")

# Room demand analysis
print()
room_demand: dict[str, int] = defaultdict(int)  # room -> total quanta demanded
for gi, g in enumerate(ind):
    ckey = (g.course_id, g.course_type)
    sr = solver._suitable_rooms.get(ckey, [])
    if not sr:
        sr = solver._type_compat_rooms.get(ckey, [])
    for r in sr:
        room_demand[r] += g.num_quanta

most_demanded = sorted(room_demand.items(), key=lambda x: -x[1])[:10]
print("Most demanded rooms (total quanta if all genes went there):")
for r, q in most_demanded:
    print(f"  {r}: {q} quanta (capacity: {len(ctx.available_quanta)} per slot)")

# Course room bottleneck
print()
course_room_count: dict[tuple[str, str], dict[str, int]] = defaultdict(
    lambda: {"suitable": 0, "genes": 0, "total_quanta": 0}
)
for gi, g in enumerate(ind):
    ckey = (g.course_id, g.course_type)
    sr = solver._suitable_rooms.get(ckey, [])
    d = course_room_count[ckey]
    d["suitable"] = len(sr)
    d["genes"] += 1
    d["total_quanta"] += g.num_quanta

bottlenecks = sorted(course_room_count.items(), key=lambda x: x[1]["suitable"])
print("Courses with fewest suitable rooms:")
for ckey, d in bottlenecks[:15]:
    avail_room_quanta = d["suitable"] * len(ctx.available_quanta)
    util = (
        d["total_quanta"] / avail_room_quanta * 100
        if avail_room_quanta > 0
        else float("inf")
    )
    print(
        f"  {ckey[0]}-{ckey[1]}: {d['suitable']} rooms, {d['genes']} genes, "
        f"{d['total_quanta']}q needed, {avail_room_quanta}q avail, util={util:.0f}%"
    )

# Group schedule density
print()
group_quanta: dict[str, int] = defaultdict(int)
for g in ind:
    for gid in g.group_ids:
        group_quanta[gid] += g.num_quanta

busiest = sorted(group_quanta.items(), key=lambda x: -x[1])[:10]
print("Busiest groups (quanta needed):")
for gid, q in busiest:
    util = q / len(ctx.available_quanta) * 100
    print(f"  {gid}: {q}/{len(ctx.available_quanta)} quanta ({util:.1f}%)")

# Part-time instructor analysis
print()
pt_genes = 0
pt_no_valid = 0
for gi, g in enumerate(ind):
    ckey = (g.course_id, g.course_type)
    qi = solver._qual_instrs.get(ckey, [])
    # Check if ALL qualified instructors are part-time with no valid starts
    all_blocked = True
    for iid in qi:
        instr = ctx.instructors.get(iid)
        if not instr:
            continue
        if instr.is_full_time:
            all_blocked = False
            break
        dur = g.num_quanta
        ok_starts = [
            sq
            for sq in solver._valid_starts(dur)
            if all((sq + d) in instr.available_quanta for d in range(dur))
        ]
        if ok_starts:
            all_blocked = False
            break
    if all_blocked and qi:
        pt_no_valid += 1

print(
    f"Genes where ALL qualified instructors are blocked (no valid starts): {pt_no_valid}"
)

# ======================================================================
# 2. Incremental solve (find where feasibility breaks)
# ======================================================================
print()
print("=" * 60)
print("2. INCREMENTAL SOLVE")
print("=" * 60)

# Group genes by group_id clusters
group_to_genes = defaultdict(list)
for gi, g in enumerate(ind):
    for gid in g.group_ids:
        group_to_genes[gid].append(gi)

# Sort groups by size
sorted_groups = sorted(group_to_genes.keys())

# Test: solve genes for each programme individually
from src.ga.repair.cp.partitioner import partition_genes

partition = partition_genes(ind, ctx, min_shared_courses=2)

print(f"Clusters: {len(partition.clusters)}")
for cl in partition.clusters:
    cid = cl.cluster_id
    cl_indices = partition.cluster_gene_indices.get(cid, [])
    print(f"  {cid}: {len(cl_indices)} genes, {len(cl.group_ids)} groups")

    if cl_indices:
        s = CPSATSolver(
            ctx, family_map, timeout_seconds=15, num_workers=8, soft_objective=False
        )
        t0 = time.time()
        r_result = s.solve(ind, cl_indices, frozen=None, warm_start=True)
        dt = time.time() - t0
        print(f"    -> {r_result.status} in {dt:.1f}s")

# ======================================================================
# 3. Constraint ablation (which HC causes infeasibility?)
# ======================================================================
print()
print("=" * 60)
print("3. CONSTRAINT ABLATION (largest cluster)")
print("=" * 60)

# Find largest cluster
largest_cl = max(
    partition.clusters,
    key=lambda c: len(partition.cluster_gene_indices.get(c.cluster_id, [])),
)
largest_indices = partition.cluster_gene_indices.get(largest_cl.cluster_id, [])
print(f"Testing cluster {largest_cl.cluster_id}: {len(largest_indices)} genes")

# Build a stripped-down model to test each constraint
from ortools.sat.python import cp_model

all_rooms = list(ctx.rooms.keys())
all_instrs = list(ctx.instructors.keys())


def build_and_solve(
    gene_indices,
    *,
    enable_group_nooverlap=True,
    enable_instr_nooverlap=True,
    enable_room_nooverlap=True,
    enable_instr_avail=True,
    label="",
):
    """Build a minimal CP model with selectable constraints."""
    model = cp_model.CpModel()

    starts = {}
    room_idxs = {}
    instr_idxs = {}
    intervals = {}
    dur_map = {}
    rooms_for = {}
    instrs_for = {}

    for gi in gene_indices:
        g = ind[gi]
        ckey = (g.course_id, g.course_type)
        dur = g.num_quanta
        dur_map[gi] = dur

        sr = solver._type_compat_rooms.get(ckey, []) or all_rooms
        rooms_for[gi] = sr
        qi = solver._qual_instrs.get(ckey, []) or all_instrs
        instrs_for[gi] = qi

        vs = solver._valid_starts(dur) or sorted(ctx.available_quanta)
        starts[gi] = model.new_int_var_from_domain(
            cp_model.Domain.from_values(vs), f"s{gi}"
        )
        room_idxs[gi] = model.new_int_var(0, len(sr) - 1, f"r{gi}")
        instr_idxs[gi] = model.new_int_var(0, len(qi) - 1, f"i{gi}")
        intervals[gi] = model.new_fixed_size_interval_var(starts[gi], dur, f"iv{gi}")

        model.add_hint(starts[gi], g.start_quanta)
        if g.room_id in sr:
            model.add_hint(room_idxs[gi], sr.index(g.room_id))
        if g.instructor_id in qi:
            model.add_hint(instr_idxs[gi], qi.index(g.instructor_id))

    # HC1: Group exclusivity
    if enable_group_nooverlap:
        group_ivs = defaultdict(list)
        for gi in gene_indices:
            for gid in ind[gi].group_ids:
                group_ivs[gid].append(intervals[gi])
        for ivs in group_ivs.values():
            if len(ivs) >= 2:
                model.add_no_overlap(ivs)

    # HC2: Instructor exclusivity
    if enable_instr_nooverlap:
        instr_opt_ivs = defaultdict(list)
        for gi in gene_indices:
            for li, iid in enumerate(instrs_for[gi]):
                b = model.new_bool_var(f"is{gi}i{li}")
                model.add(instr_idxs[gi] == li).only_enforce_if(b)
                model.add(instr_idxs[gi] != li).only_enforce_if(b.negated())
                oiv = model.new_optional_fixed_size_interval_var(
                    starts[gi], dur_map[gi], b, f"oi{gi}i{li}"
                )
                instr_opt_ivs[iid].append(oiv)
        for ivs in instr_opt_ivs.values():
            if len(ivs) >= 2:
                model.add_no_overlap(ivs)

    # HC3: Room exclusivity
    if enable_room_nooverlap:
        room_opt_ivs = defaultdict(list)
        for gi in gene_indices:
            for li, rid in enumerate(rooms_for[gi]):
                b = model.new_bool_var(f"rs{gi}r{li}")
                model.add(room_idxs[gi] == li).only_enforce_if(b)
                model.add(room_idxs[gi] != li).only_enforce_if(b.negated())
                oiv = model.new_optional_fixed_size_interval_var(
                    starts[gi], dur_map[gi], b, f"ro{gi}r{li}"
                )
                room_opt_ivs[rid].append(oiv)
        for ivs in room_opt_ivs.values():
            if len(ivs) >= 2:
                model.add_no_overlap(ivs)

    # HC6: Instructor availability
    if enable_instr_avail:
        for gi in gene_indices:
            dur = dur_map[gi]
            for li, iid in enumerate(instrs_for[gi]):
                instr = ctx.instructors.get(iid)
                if not instr or instr.is_full_time:
                    continue
                ok_starts = [
                    sq
                    for sq in solver._valid_starts(dur)
                    if all((sq + d) in instr.available_quanta for d in range(dur))
                ]
                if not ok_starts:
                    model.add(instr_idxs[gi] != li)
                else:
                    b = model.new_bool_var(f"av{gi}i{li}")
                    model.add(instr_idxs[gi] == li).only_enforce_if(b)
                    model.add(instr_idxs[gi] != li).only_enforce_if(b.negated())
                    model.add_linear_expression_in_domain(
                        starts[gi], cp_model.Domain.from_values(ok_starts)
                    ).only_enforce_if(b)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 15
    s.parameters.num_workers = 8
    s.parameters.log_search_progress = False

    t0 = time.time()
    status = s.solve(model)
    dt = time.time() - t0
    sname = s.status_name(status)
    print(f"  {label:40s} -> {sname:12s} ({dt:.1f}s)")
    return sname


# Test each constraint in isolation
build_and_solve(
    largest_indices,
    enable_group_nooverlap=True,
    enable_instr_nooverlap=False,
    enable_room_nooverlap=False,
    enable_instr_avail=False,
    label="HC1 only (group NoOverlap)",
)

build_and_solve(
    largest_indices,
    enable_group_nooverlap=False,
    enable_instr_nooverlap=True,
    enable_room_nooverlap=False,
    enable_instr_avail=False,
    label="HC2 only (instr NoOverlap)",
)

build_and_solve(
    largest_indices,
    enable_group_nooverlap=False,
    enable_instr_nooverlap=False,
    enable_room_nooverlap=True,
    enable_instr_avail=False,
    label="HC3 only (room NoOverlap)",
)

build_and_solve(
    largest_indices,
    enable_group_nooverlap=False,
    enable_instr_nooverlap=False,
    enable_room_nooverlap=False,
    enable_instr_avail=True,
    label="HC6 only (instr avail)",
)

build_and_solve(
    largest_indices,
    enable_group_nooverlap=True,
    enable_instr_nooverlap=True,
    enable_room_nooverlap=False,
    enable_instr_avail=False,
    label="HC1+HC2 (group+instr)",
)

build_and_solve(
    largest_indices,
    enable_group_nooverlap=True,
    enable_instr_nooverlap=False,
    enable_room_nooverlap=True,
    enable_instr_avail=False,
    label="HC1+HC3 (group+room)",
)

build_and_solve(
    largest_indices,
    enable_group_nooverlap=True,
    enable_instr_nooverlap=True,
    enable_room_nooverlap=True,
    enable_instr_avail=False,
    label="HC1+HC2+HC3 (no avail)",
)

build_and_solve(
    largest_indices,
    enable_group_nooverlap=True,
    enable_instr_nooverlap=True,
    enable_room_nooverlap=True,
    enable_instr_avail=True,
    label="ALL constraints",
)
