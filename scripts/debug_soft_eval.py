"""Debug script to isolate soft eval per-constraint discrepancy."""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from src.utils.logging_config import quick_setup

logger = quick_setup()


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constraints.evaluator import Evaluator
from src.domain.timetable import Timetable
from src.io.data_store import DataStore
from src.io.time_system import QuantumTimeSystem
from src.pipeline.encoding import chromosome_views
from src.pipeline.pymoo_operators import ConstructiveSampling
from src.pipeline.scheduling_problem import SchedulingProblem
from src.pipeline.soft_evaluator_vectorized import (
    SoftVectorizedData,
    eval_soft_vectorized,
    prepare_soft_vectorized_data,
)

PKL_PATH = "events_with_domains.pkl"
with open(PKL_PATH, "rb") as f:
    pkl_data = pickle.load(f)

# Generate population
prob = SchedulingProblem(PKL_PATH)
sampling = ConstructiveSampling(PKL_PATH)
X = sampling._do(prob, 5)
logger.info("Pop shape: %s", X.shape)

events = pkl_data["events"]
idx_to_instructor = {int(k): v for k, v in pkl_data["idx_to_instructor"].items()}
idx_to_room = {int(k): v for k, v in pkl_data["idx_to_room"].items()}
E = len(events)

store = DataStore.from_json("data")
ctx = store.to_context()
qts = QuantumTimeSystem()
evaluator = Evaluator()

sdata = prepare_soft_vectorized_data(pkl_data)


# Vectorized: compute SEPARATELY student compactness, instructor compactness, lunch
def _eval_student_compactness_only(X, sdata):
    """Run only student compactness from vectorized code."""
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]
    n_groups = sdata.n_groups
    n_days = sdata.n_days
    qpd = sdata.quanta_per_day

    time_assign = X[:, 2::3]
    GQ = sdata.GQ
    grp_starts = time_assign[:, sdata.grp_exp_event]
    grp_quanta = grp_starts + sdata.grp_exp_offset[np.newaxis, :]
    grp_days = grp_quanta // qpd
    grp_within = grp_quanta % qpd
    grp_days = np.clip(grp_days, 0, n_days - 1)

    stride = n_groups * n_days * qpd
    n_idx = np.repeat(np.arange(N, dtype=np.int64), GQ)
    g_flat = np.tile(sdata.grp_exp_group, N)
    d_flat = grp_days.ravel()
    w_flat = grp_within.ravel()
    flat_idx = n_idx * stride + g_flat * (n_days * qpd) + d_flat * qpd + w_flat
    occ_flat = np.bincount(flat_idx.astype(np.int64), minlength=N * stride)
    occ = occ_flat.reshape(N, n_groups, n_days, qpd) > 0

    any_occ = occ.any(axis=3)
    occ_count = occ.sum(axis=3)
    qrange = np.arange(qpd, dtype=np.int32)
    qr4 = qrange[np.newaxis, np.newaxis, np.newaxis, :]
    occ_masked_min = np.where(occ, qr4, qpd)
    occ_masked_max = np.where(occ, qr4, -1)
    min_q = occ_masked_min.min(axis=3)
    max_q = occ_masked_max.max(axis=3)
    break_mask = sdata.break_within_day
    in_span = (qr4 >= min_q[:, :, :, np.newaxis]) & (qr4 <= max_q[:, :, :, np.newaxis])
    gap_mask = in_span & ~occ & ~break_mask[np.newaxis, np.newaxis, np.newaxis, :]
    gap = gap_mask.sum(axis=3).astype(np.int32)
    gap = np.where(any_occ & (occ_count >= 2), gap, 0)
    return gap.sum(axis=(1, 2)).astype(np.float64)


def _eval_instructor_compactness_only(X, sdata):
    """Run only instructor compactness from vectorized code."""
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]
    n_inst = sdata.n_instructors
    n_days = sdata.n_days
    qpd = sdata.quanta_per_day

    inst_assign = X[:, 0::3]
    time_assign = X[:, 2::3]
    Q = sdata.Q
    inst_starts = time_assign[:, sdata.exp_event]
    inst_quanta = inst_starts + sdata.exp_offset[np.newaxis, :]
    inst_ids = inst_assign[:, sdata.exp_event]
    inst_days = inst_quanta // qpd
    inst_within = inst_quanta % qpd
    inst_days = np.clip(inst_days, 0, n_days - 1)

    stride_i = n_inst * n_days * qpd
    n_idx_i = np.repeat(np.arange(N, dtype=np.int64), Q)
    i_flat = inst_ids.ravel()
    d_flat_i = inst_days.ravel()
    w_flat_i = inst_within.ravel()
    flat_idx_i = (
        n_idx_i * stride_i + i_flat * (n_days * qpd) + d_flat_i * qpd + w_flat_i
    )
    occ_i_flat = np.bincount(flat_idx_i.astype(np.int64), minlength=N * stride_i)
    occ_i = occ_i_flat.reshape(N, n_inst, n_days, qpd) > 0

    any_occ_i = occ_i.any(axis=3)
    occ_count_i = occ_i.sum(axis=3)
    qrange = np.arange(qpd, dtype=np.int32)
    qr4 = qrange[np.newaxis, np.newaxis, np.newaxis, :]
    occ_masked_min_i = np.where(occ_i, qr4, qpd)
    occ_masked_max_i = np.where(occ_i, qr4, -1)
    min_q_i = occ_masked_min_i.min(axis=3)
    max_q_i = occ_masked_max_i.max(axis=3)
    break_mask = sdata.break_within_day
    in_span_i = (qr4 >= min_q_i[:, :, :, np.newaxis]) & (
        qr4 <= max_q_i[:, :, :, np.newaxis]
    )
    gap_mask_i = in_span_i & ~occ_i & ~break_mask[np.newaxis, np.newaxis, np.newaxis, :]
    gap_i = gap_mask_i.sum(axis=3).astype(np.int32)
    gap_i = np.where(any_occ_i & (occ_count_i >= 2), gap_i, 0)
    return gap_i.sum(axis=(1, 2)).astype(np.float64)


def _eval_lunch_only(X, sdata):
    """Run only lunch from vectorized code."""
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]
    n_groups = sdata.n_groups
    n_days = sdata.n_days
    qpd = sdata.quanta_per_day

    time_assign = X[:, 2::3]
    GQ = sdata.GQ
    grp_starts = time_assign[:, sdata.grp_exp_event]
    grp_quanta = grp_starts + sdata.grp_exp_offset[np.newaxis, :]
    grp_days = grp_quanta // qpd
    grp_within = grp_quanta % qpd
    grp_days = np.clip(grp_days, 0, n_days - 1)

    stride = n_groups * n_days * qpd
    n_idx = np.repeat(np.arange(N, dtype=np.int64), GQ)
    g_flat = np.tile(sdata.grp_exp_group, N)
    d_flat = grp_days.ravel()
    w_flat = grp_within.ravel()
    flat_idx = n_idx * stride + g_flat * (n_days * qpd) + d_flat * qpd + w_flat
    occ_flat = np.bincount(flat_idx.astype(np.int64), minlength=N * stride)
    occ = occ_flat.reshape(N, n_groups, n_days, qpd) > 0
    any_occ = occ.any(axis=3)

    lunch_mask = sdata.lunch_window
    lunch_window_size = int(lunch_mask.sum())
    occ_in_lunch = (occ & lunch_mask[np.newaxis, np.newaxis, np.newaxis, :]).sum(axis=3)
    free_lunch = lunch_window_size - occ_in_lunch
    lunch_deficit = np.maximum(sdata.lunch_min_quanta - free_lunch, 0)
    lunch_deficit = np.where(any_occ, lunch_deficit, 0)
    return lunch_deficit.sum(axis=(1, 2)).astype(np.float64)


# ---- Compute vectorized per-constraint ----
vec_student = _eval_student_compactness_only(X, sdata)
vec_instructor = _eval_instructor_compactness_only(X, sdata)
vec_lunch = _eval_lunch_only(X, sdata)
vec_total = vec_student + vec_instructor + vec_lunch

# ---- Compute OOP per-constraint ----
from src.domain.gene import SessionGene

oop_student = []
oop_instructor = []
oop_lunch = []

for ind_idx in range(X.shape[0]):
    xi = X[ind_idx].astype(int)
    inst, room, time_ = chromosome_views(xi)
    genes = []
    for e in range(E):
        ev = events[e]
        genes.append(
            SessionGene(
                course_id=ev["course_id"],
                course_type=ev["course_type"],
                instructor_id=idx_to_instructor[int(inst[e])],
                group_ids=list(ev["group_ids"]),
                room_id=idx_to_room[int(room[e])],
                start_quanta=int(time_[e]),
                num_quanta=ev["num_quanta"],
            )
        )
    tt = Timetable(genes, ctx, qts)
    bd = evaluator.soft_breakdown(tt)
    oop_student.append(bd.get("student_schedule_compactness", 0.0))
    oop_instructor.append(bd.get("instructor_schedule_compactness", 0.0))
    oop_lunch.append(bd.get("student_lunch_break", 0.0))

oop_student = np.array(oop_student)
oop_instructor = np.array(oop_instructor)
oop_lunch = np.array(oop_lunch)
oop_total = oop_student + oop_instructor + oop_lunch

logger.info("\n=== PER-CONSTRAINT COMPARISON ===")
logger.info("\nStudent Compactness:")
logger.info("  Vec: %s", vec_student)
logger.info("  OOP: %s", oop_student)
logger.info("  Diff: %s", vec_student - oop_student)

logger.info("\nInstructor Compactness:")
logger.info("  Vec: %s", vec_instructor)
logger.info("  OOP: %s", oop_instructor)
logger.info("  Diff: %s", vec_instructor - oop_instructor)

logger.info("\nLunch Break:")
logger.info("  Vec: %s", vec_lunch)
logger.info("  OOP: %s", oop_lunch)
logger.info("  Diff: %s", vec_lunch - oop_lunch)

logger.info("\nTotal:")
logger.info("  Vec: %s", vec_total)
logger.info("  OOP: %s", oop_total)
logger.info("  Diff: %s", vec_total - oop_total)

# Deep dive: for individual 0, show per-group per-day student gaps
logger.info("\n=== DEEP DIVE: Individual 0 ===")
xi = X[0].astype(int)
inst, room, time_ = chromosome_views(xi)
genes = []
for e in range(E):
    ev = events[e]
    genes.append(
        SessionGene(
            course_id=ev["course_id"],
            course_type=ev["course_type"],
            instructor_id=idx_to_instructor[int(inst[e])],
            group_ids=list(ev["group_ids"]),
            room_id=idx_to_room[int(room[e])],
            start_quanta=int(time_[e]),
            num_quanta=ev["num_quanta"],
        )
    )
tt = Timetable(genes, ctx, qts)

# OOP group daily map
from src.constraints.constraints import _group_daily_map

group_daily = _group_daily_map(tt, qts)
break_quanta = qts.get_midday_break_quanta()

# Sort groups for comparison
all_gids = set()
for ev in events:
    all_gids.update(ev["group_ids"])
group_to_idx = {gid: i for i, gid in enumerate(sorted(all_gids))}

# Show first few groups with gaps
logger.info("  Total groups in OOP: %d", len(group_daily))
logger.info("  Total groups in Vec: %d", sdata.n_groups)

gap_examples = 0
for gid in sorted(group_daily.keys()):
    days = group_daily[gid]
    for day_name, quanta in days.items():
        if len(quanta) < 2:
            continue
        sorted_q = sorted(quanta)
        min_q, max_q = sorted_q[0], sorted_q[-1]
        break_q = break_quanta.get(day_name, set())
        oop_gap = sum(
            1 for q in range(min_q, max_q + 1) if q not in quanta and q not in break_q
        )
        if oop_gap > 0 and gap_examples < 10:
            gidx = group_to_idx.get(gid, -1)
            logger.debug(
                "  Group %s (idx=%d), %s: occ=%s, break=%s, oop_gap=%d",
                gid,
                gidx,
                day_name,
                sorted_q,
                sorted(break_q),
                oop_gap,
            )
            gap_examples += 1

# Compare occupancy tensors for individual 0
logger.info("\n=== OCCUPANCY COMPARISON: Individual 0 ===")

# Build vec occupancy tensor for ind 0
N_test = 1
X_test = X[0:1].astype(np.int64)
n_groups_v = sdata.n_groups
n_days_v = sdata.n_days
qpd_v = sdata.quanta_per_day
time_assign = X_test[:, 2::3]
GQ = sdata.GQ
grp_starts = time_assign[:, sdata.grp_exp_event]
grp_quanta = grp_starts + sdata.grp_exp_offset[np.newaxis, :]
grp_days = grp_quanta // qpd_v
grp_within = grp_quanta % qpd_v
grp_days = np.clip(grp_days, 0, n_days_v - 1)
stride = n_groups_v * n_days_v * qpd_v
n_idx = np.repeat(np.arange(N_test, dtype=np.int64), GQ)
g_flat = np.tile(sdata.grp_exp_group, N_test)
d_flat = grp_days.ravel()
w_flat = grp_within.ravel()
flat_idx = n_idx * stride + g_flat * (n_days_v * qpd_v) + d_flat * qpd_v + w_flat
occ_flat_v = np.bincount(flat_idx.astype(np.int64), minlength=N_test * stride)
occ_v = occ_flat_v.reshape(N_test, n_groups_v, n_days_v, qpd_v) > 0

DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
idx_to_group = {v: k for k, v in group_to_idx.items()}

# Compare a few groups
mismatch_count = 0
for gidx in range(min(n_groups_v, 10)):
    gid = idx_to_group.get(gidx, "??")
    oop_days = group_daily.get(gid, {})
    for d_idx in range(n_days_v):
        day_name = DAY_NAMES[d_idx]
        vec_occ = set(np.where(occ_v[0, gidx, d_idx])[0])
        oop_occ = oop_days.get(day_name, set())
        if vec_occ != oop_occ:
            logger.warning(
                "  MISMATCH %s (idx=%d), %s: vec=%s, oop=%s",
                gid,
                gidx,
                day_name,
                sorted(vec_occ),
                sorted(oop_occ),
            )
            mismatch_count += 1
    if mismatch_count > 20:
        logger.info("  ... (truncated)")
        break

if mismatch_count == 0:
    logger.info("  All group-day occupancies MATCH!")
else:
    # Detailed trace for first mismatch
    logger.info("\n  Total mismatches: %d", mismatch_count)
    # Trace BAM1A (idx=0) Wednesday (day 3) — vec has extra [0,1], oop has [2,3,5,6]
    target_gid = "BAM1A"
    target_gidx = group_to_idx[target_gid]
    target_day = 3  # Wednesday
    target_day_name = "Wednesday"
    logger.debug("\n  TRACE: Events mapped to %s (idx=%d)", target_gid, target_gidx)

    # Find all events with BAM1A in their groups
    bam1a_events = [
        (e, ev) for e, ev in enumerate(events) if target_gid in ev["group_ids"]
    ]
    logger.debug("  Total events for %s: %d", target_gid, len(bam1a_events))

    xi = X[0].astype(int)
    inst_v, room_v, time_v = chromosome_views(xi)

    for e, ev in bam1a_events:
        start = int(time_v[e])
        dur = ev["num_quanta"]
        quanta = list(range(start, start + dur))
        days_within = [(q // 7, q % 7) for q in quanta]
        on_target = [(d, w) for d, w in days_within if d == target_day]
        all_days = [(d, w) for d, w in days_within]
        if on_target or any(d == target_day for d, _ in all_days):
            logger.debug(
                "    Event %d (%s): start=%d, dur=%d, quanta=%s, day_within=%s",
                e,
                ev["course_id"],
                start,
                dur,
                quanta,
                days_within,
            )

    # Now show what OOP has for BAM1A Wednesday
    oop_bam1a_wed = group_daily.get(target_gid, {}).get(target_day_name, set())
    logger.debug(
        "\n  OOP %s %s: %s", target_gid, target_day_name, sorted(oop_bam1a_wed)
    )
    logger.debug(
        "  Vec %s %s: %s",
        target_gid,
        target_day_name,
        sorted(set(np.where(occ_v[0, target_gidx, target_day])[0])),
    )

    # Also trace events at continuous quanta 21, 22 (Wednesday q0, q1) for any group
    logger.debug(
        "\n  Events yielding continuous quanta 21-22 (Wed q0-q1) for %s:", target_gid
    )
    for e, ev in bam1a_events:
        start = int(time_v[e])
        dur = ev["num_quanta"]
        quanta = list(range(start, start + dur))
        if any(q in [21, 22] for q in quanta):
            logger.debug(
                "    Event %d (%s): start=%d, dur=%d, quanta=%s",
                e,
                ev["course_id"],
                start,
                dur,
                quanta,
            )

    # Now compare: what events does the expansion array map to Wednesday for BAM1A?
    logger.debug("\n  Expansion entries for group %d (BAM1A):", target_gidx)
    exp_mask = sdata.grp_exp_group == target_gidx
    exp_events_for_group = sdata.grp_exp_event[exp_mask]
    exp_offsets_for_group = sdata.grp_exp_offset[exp_mask]
    for e_idx in sorted(set(exp_events_for_group)):
        sub_mask = (sdata.grp_exp_event == e_idx) & exp_mask
        offsets = sdata.grp_exp_offset[sub_mask]
        start = int(time_v[e_idx])
        quanta = [start + int(o) for o in offsets]
        days_within = [(q // 7, q % 7) for q in quanta]
        on_wed = [(d, w) for d, w in days_within if d == target_day]
        if on_wed:
            ev = events[e_idx]
            logger.debug(
                "    Event %d (%s): start=%d, offsets=%s, quanta=%s, days_within=%s",
                e_idx,
                ev["course_id"],
                start,
                list(offsets),
                quanta,
                days_within,
            )

# Also compare instructor occupancy for first few instructors
logger.info("\n=== INSTRUCTOR OCCUPANCY COMPARISON: Individual 0 ===")
from src.constraints.constraints import _instructor_daily_map

inst_daily = _instructor_daily_map(tt, qts)

# Build vec instructor occupancy for ind 0
inst_assign_t = X_test[:, 0::3]
Q_v = sdata.Q
inst_starts = time_assign[:, sdata.exp_event]
inst_quanta = inst_starts + sdata.exp_offset[np.newaxis, :]
inst_ids_v = inst_assign_t[:, sdata.exp_event]
inst_days_v = inst_quanta // qpd_v
inst_within_v = inst_quanta % qpd_v
inst_days_v = np.clip(inst_days_v, 0, n_days_v - 1)
n_inst_v = sdata.n_instructors
stride_i = n_inst_v * n_days_v * qpd_v
n_idx_i = np.repeat(np.arange(N_test, dtype=np.int64), Q_v)
i_flat_v = inst_ids_v.ravel()
d_flat_i = inst_days_v.ravel()
w_flat_i = inst_within_v.ravel()
flat_idx_i = (
    n_idx_i * stride_i + i_flat_v * (n_days_v * qpd_v) + d_flat_i * qpd_v + w_flat_i
)
occ_i_flat_v = np.bincount(flat_idx_i.astype(np.int64), minlength=N_test * stride_i)
occ_i_v = occ_i_flat_v.reshape(N_test, n_inst_v, n_days_v, qpd_v) > 0

# Build instructor index from idx_to_instructor
idx_to_inst_name = {int(k): v for k, v in pkl_data["idx_to_instructor"].items()}

mismatch_inst = 0
for iid in range(min(n_inst_v, 200)):
    iname = idx_to_inst_name.get(iid)
    if iname is None:
        continue
    oop_i_days = inst_daily.get(iname, {})
    for d_idx in range(n_days_v):
        day_name = DAY_NAMES[d_idx]
        vec_occ_i = set(np.where(occ_i_v[0, iid, d_idx])[0])
        oop_occ_i = oop_i_days.get(day_name, set())
        if vec_occ_i != oop_occ_i:
            logger.warning(
                "  MISMATCH inst %s (idx=%d), %s: vec=%s, oop=%s",
                iname,
                iid,
                day_name,
                sorted(vec_occ_i),
                sorted(oop_occ_i),
            )
            mismatch_inst += 1
            if mismatch_inst > 20:
                break
    if mismatch_inst > 20:
        logger.info("  ... (truncated)")
        break

if mismatch_inst == 0:
    logger.info("  All instructor-day occupancies MATCH!")
