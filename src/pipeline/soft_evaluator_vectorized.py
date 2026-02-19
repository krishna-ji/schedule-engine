"""Vectorized soft constraint evaluator — operates on full population matrices.

Replaces the per-individual OOP Timetable->Evaluator pipeline for the top 3
soft constraints:
  1. StudentScheduleCompactness  — gap penalty per group per day
  2. InstructorScheduleCompactness — gap penalty per instructor per day
  3. StudentLunchBreak — free quanta in lunch window per group per day

API
---
    prepare_soft_vectorized_data(pkl_data, qts=None) -> SoftVectorizedData
    eval_soft_vectorized(X, sdata) -> S    # shape (N,) float64

All computation is done with numpy over the full population (N individuals)
without per-individual Python loops.

Algorithm
---------
For gap penalty (compactness):
  1. Expand events into (event, quantum) tuples (like hard evaluator).
  2. For each (individual, group/instructor, day), find occupied within-day
     quanta using scatter-into-bins (np.add.at on a boolean tensor).
  3. For each occupied entity-day, compute gap = (max_q - min_q + 1) - count
     - break_quanta_in_span, i.e. total range minus occupied minus breaks.

For lunch break:
  1. Build group-day occupancy tensor.
  2. For each group-day, count occupied quanta in the break window.
  3. Penalty = max(0, break_min_quanta - (window_size - occupied_in_window)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default time system constants (6 days, 7 quanta/day)
_DEFAULT_N_DAYS = 6
_DEFAULT_QUANTA_PER_DAY = 7
_DEFAULT_BREAK_WITHIN_DAY = {2}  # midday break at within-day quantum 2
_DEFAULT_BREAK_WINDOW = {2, 3}  # break window: within-day quanta 2-3


@dataclass(frozen=True, slots=True)
class SoftVectorizedData:
    """Precomputed arrays for vectorized soft evaluation."""

    n_events: int
    n_groups: int
    n_instructors: int
    n_days: int
    quanta_per_day: int

    # Per-event durations (E,)
    durations: np.ndarray  # int32

    # Group expansion: each event × duration × group_count entries
    # GQ = sum(dur_e * n_groups_e)
    GQ: int
    grp_exp_event: np.ndarray  # int32 (GQ,) — event index
    grp_exp_offset: np.ndarray  # int32 (GQ,) — quantum offset within event
    grp_exp_group: np.ndarray  # int32 (GQ,) — group index

    # Instructor expansion: Q = sum(durations)
    Q: int
    exp_event: np.ndarray  # int32 (Q,)
    exp_offset: np.ndarray  # int32 (Q,)

    # Event-to-instructor mapping (E,) — which instructor index for each event
    # (not used in precompute — populated at eval time from X)

    # Day boundary arrays
    day_offsets: np.ndarray  # int32 (n_days,) — start quantum of each day
    day_sizes: np.ndarray  # int32 (n_days,) — quanta count per day

    # Break quanta (midday break, for gap exclusion)
    # Shape: (n_days,) of within-day quantum (set as int for simple cases)
    break_within_day: np.ndarray  # bool (quanta_per_day,) — True if break quantum

    # Lunch window
    lunch_window: np.ndarray  # bool (quanta_per_day,) — True if in lunch window
    lunch_min_quanta: int  # minimum free quanta required

    # Weights
    gap_penalty_per_quantum: float
    lunch_penalty_per_missing: float


def prepare_soft_vectorized_data(
    pkl_data: dict,
    *,
    n_days: int = _DEFAULT_N_DAYS,
    quanta_per_day: int = _DEFAULT_QUANTA_PER_DAY,
    break_within_day_quanta: set[int] | None = None,
    lunch_window_quanta: set[int] | None = None,
    lunch_min_quanta: int = 1,
    gap_penalty_per_quantum: float = 1.0,
    lunch_penalty_per_missing: float = 1.0,
) -> SoftVectorizedData:
    """Build SoftVectorizedData from pkl dict."""
    events = pkl_data["events"]
    E = len(events)

    if break_within_day_quanta is None:
        break_within_day_quanta = _DEFAULT_BREAK_WITHIN_DAY
    if lunch_window_quanta is None:
        lunch_window_quanta = _DEFAULT_BREAK_WINDOW

    # Group index mapping
    all_gids: set[str] = set()
    for ev in events:
        all_gids.update(ev["group_ids"])
    group_to_idx = {gid: i for i, gid in enumerate(sorted(all_gids))}
    n_groups = len(group_to_idx)

    # Dimensions
    n_instructors = (
        max((max(ai) for ai in pkl_data["allowed_instructors"] if ai), default=0) + 1
    )

    durations = np.array([ev["num_quanta"] for ev in events], dtype=np.int32)
    event_group_indices = [
        [group_to_idx[gid] for gid in ev["group_ids"]] for ev in events
    ]

    # Instructor expansion (Q entries)
    Q = int(durations.sum())
    exp_event = np.empty(Q, dtype=np.int32)
    exp_offset = np.empty(Q, dtype=np.int32)
    pos = 0
    for e in range(E):
        d = int(durations[e])
        exp_event[pos : pos + d] = e
        exp_offset[pos : pos + d] = np.arange(d, dtype=np.int32)
        pos += d

    # Group expansion (GQ entries)
    GQ = sum(int(durations[e]) * len(event_group_indices[e]) for e in range(E))
    grp_exp_event = np.empty(GQ, dtype=np.int32)
    grp_exp_offset = np.empty(GQ, dtype=np.int32)
    grp_exp_group = np.empty(GQ, dtype=np.int32)
    pos = 0
    for e in range(E):
        d = int(durations[e])
        for gidx in event_group_indices[e]:
            grp_exp_event[pos : pos + d] = e
            grp_exp_offset[pos : pos + d] = np.arange(d, dtype=np.int32)
            grp_exp_group[pos : pos + d] = gidx
            pos += d

    # Day boundaries
    day_offsets = np.arange(n_days, dtype=np.int32) * quanta_per_day
    day_sizes = np.full(n_days, quanta_per_day, dtype=np.int32)

    # Break mask
    break_mask = np.zeros(quanta_per_day, dtype=np.bool_)
    for q in break_within_day_quanta:
        if 0 <= q < quanta_per_day:
            break_mask[q] = True

    # Lunch window mask
    lunch_mask = np.zeros(quanta_per_day, dtype=np.bool_)
    for q in lunch_window_quanta:
        if 0 <= q < quanta_per_day:
            lunch_mask[q] = True

    return SoftVectorizedData(
        n_events=E,
        n_groups=n_groups,
        n_instructors=n_instructors,
        n_days=n_days,
        quanta_per_day=quanta_per_day,
        durations=durations,
        GQ=GQ,
        grp_exp_event=grp_exp_event,
        grp_exp_offset=grp_exp_offset,
        grp_exp_group=grp_exp_group,
        Q=Q,
        exp_event=exp_event,
        exp_offset=exp_offset,
        day_offsets=day_offsets,
        day_sizes=day_sizes,
        break_within_day=break_mask,
        lunch_window=lunch_mask,
        lunch_min_quanta=lunch_min_quanta,
        gap_penalty_per_quantum=gap_penalty_per_quantum,
        lunch_penalty_per_missing=lunch_penalty_per_missing,
    )


# ------------------------------------------------------------------
# Vectorized soft evaluation kernel
# ------------------------------------------------------------------


def eval_soft_vectorized(
    X: np.ndarray,
    sdata: SoftVectorizedData,
) -> np.ndarray:
    """Evaluate soft constraints for the full population.

    Parameters
    ----------
    X : ndarray, shape (N, 3*E), int
        Population matrix.
    sdata : SoftVectorizedData

    Returns
    -------
    S : ndarray, shape (N,), float64
        Total soft penalty per individual (sum of all soft constraints).
    """
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]
    n_groups = sdata.n_groups
    n_inst = sdata.n_instructors
    n_days = sdata.n_days
    qpd = sdata.quanta_per_day

    # Extract assignment views
    inst_assign = X[:, 0::3]  # (N, E)
    time_assign = X[:, 2::3].copy()  # (N, E) — copy because we may clamp

    # ------------------------------------------------------------------
    # Day-boundary clamping (match SessionGene.__post_init__ behaviour):
    # If an event's duration fits within a single day but the start would
    # cause it to spill past the end of the day, shift start backwards.
    # ------------------------------------------------------------------
    durations = sdata.durations  # (E,) int32
    day_offsets_e = (time_assign // qpd) * qpd  # (N, E) — start of starting day
    end_of_day_e = day_offsets_e + qpd  # (N, E) — exclusive end of day
    spills = (durations[np.newaxis, :] <= qpd) & (
        time_assign + durations[np.newaxis, :] > end_of_day_e
    )
    clamped_start = np.maximum(day_offsets_e, end_of_day_e - durations[np.newaxis, :])
    time_assign = np.where(spills, clamped_start, time_assign)

    S = np.zeros(N, dtype=np.float64)

    # ==================================================================
    # 1. Student Schedule Compactness (gap penalty per group per day)
    # ==================================================================
    # Build occupancy tensor: occ[n, g, day, within_day_q] = count
    # Using group expansion arrays

    GQ = sdata.GQ
    grp_starts = time_assign[:, sdata.grp_exp_event]  # (N, GQ)
    grp_quanta = (
        grp_starts + sdata.grp_exp_offset[np.newaxis, :]
    )  # (N, GQ) absolute quanta

    # Convert absolute quanta to (day, within_day)
    grp_days = grp_quanta // qpd  # (N, GQ)
    grp_within = grp_quanta % qpd  # (N, GQ) — within-day quantum index

    # Clamp days to valid range
    grp_days = np.clip(grp_days, 0, n_days - 1)

    # Build flat index: n * (n_groups * n_days * qpd) + g * (n_days * qpd) + day * qpd + within
    stride = n_groups * n_days * qpd
    n_idx = np.repeat(np.arange(N, dtype=np.int64), GQ)
    g_flat = np.tile(sdata.grp_exp_group, N)
    d_flat = grp_days.ravel()
    w_flat = grp_within.ravel()

    flat_idx = n_idx * stride + g_flat * (n_days * qpd) + d_flat * qpd + w_flat

    # Binary occupancy: 1 if any event occupies this (group, day, quantum)
    occ_flat = np.bincount(flat_idx.astype(np.int64), minlength=N * stride)
    occ = occ_flat.reshape(N, n_groups, n_days, qpd) > 0  # bool (N, G, D, QPD)

    # For each (n, g, d): compute gap penalty
    # gap = count of quanta in [min_q, max_q] that are NOT occupied AND NOT break
    # Direct computation using range mask (guaranteed correct).

    # any_occ[n,g,d] = True if group has any class on that day
    any_occ = occ.any(axis=3)  # (N, G, D)

    # Count occupied quanta per (n, g, d)
    occ_count = occ.sum(axis=3)  # (N, G, D)

    # Find min and max within-day quantum per (n, g, d)
    qrange = np.arange(qpd, dtype=np.int32)  # (QPD,)
    qr4 = qrange[np.newaxis, np.newaxis, np.newaxis, :]  # (1,1,1,QPD)

    occ_masked_min = np.where(occ, qr4, qpd)
    occ_masked_max = np.where(occ, qr4, -1)

    min_q = occ_masked_min.min(axis=3)  # (N, G, D)
    max_q = occ_masked_max.max(axis=3)  # (N, G, D)

    # Build per-(n,g,d) range mask: True for quanta in [min_q, max_q]
    break_mask = sdata.break_within_day  # (QPD,) bool
    in_span = (qr4 >= min_q[:, :, :, np.newaxis]) & (qr4 <= max_q[:, :, :, np.newaxis])
    # gap = quanta that are in_span AND NOT occupied AND NOT break
    gap_mask = in_span & ~occ & ~break_mask[np.newaxis, np.newaxis, np.newaxis, :]
    gap = gap_mask.sum(axis=3).astype(np.int32)  # (N, G, D)
    # Only count where entity has >= 2 occupied quanta on that day
    gap = np.where(any_occ & (occ_count >= 2), gap, 0)

    student_compactness = (
        gap.sum(axis=(1, 2)).astype(np.float64) * sdata.gap_penalty_per_quantum
    )
    S += student_compactness

    # ==================================================================
    # 2. Instructor Schedule Compactness (same pattern, over instructors)
    # ==================================================================
    Q = sdata.Q
    inst_starts = time_assign[:, sdata.exp_event]  # (N, Q)
    inst_quanta = inst_starts + sdata.exp_offset[np.newaxis, :]  # (N, Q)
    inst_ids = inst_assign[:, sdata.exp_event]  # (N, Q) — instructor index per entry

    inst_days = inst_quanta // qpd  # (N, Q)
    inst_within = inst_quanta % qpd  # (N, Q)
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
    occ_i = occ_i_flat.reshape(N, n_inst, n_days, qpd) > 0  # (N, I, D, QPD)

    any_occ_i = occ_i.any(axis=3)  # (N, I, D)
    occ_count_i = occ_i.sum(axis=3)  # (N, I, D)

    occ_masked_min_i = np.where(occ_i, qr4, qpd)
    occ_masked_max_i = np.where(occ_i, qr4, -1)
    min_q_i = occ_masked_min_i.min(axis=3)
    max_q_i = occ_masked_max_i.max(axis=3)

    # Direct gap computation for instructors (same pattern as students)
    in_span_i = (qr4 >= min_q_i[:, :, :, np.newaxis]) & (
        qr4 <= max_q_i[:, :, :, np.newaxis]
    )
    gap_mask_i = in_span_i & ~occ_i & ~break_mask[np.newaxis, np.newaxis, np.newaxis, :]
    gap_i = gap_mask_i.sum(axis=3).astype(np.int32)
    gap_i = np.where(any_occ_i & (occ_count_i >= 2), gap_i, 0)

    instructor_compactness = (
        gap_i.sum(axis=(1, 2)).astype(np.float64) * sdata.gap_penalty_per_quantum
    )
    S += instructor_compactness

    # ==================================================================
    # 3. Student Lunch Break
    # ==================================================================
    # For each (n, group, day): count occupied quanta in lunch window
    # Penalty if (window_size - occupied_in_window) < lunch_min_quanta
    lunch_mask = sdata.lunch_window  # (QPD,) bool
    lunch_window_size = int(lunch_mask.sum())

    # occ already has shape (N, G, D, QPD)
    # occupied_in_lunch[n, g, d] = sum of occ[n,g,d,q] for q in lunch window
    occ_in_lunch = (occ & lunch_mask[np.newaxis, np.newaxis, np.newaxis, :]).sum(
        axis=3
    )  # (N, G, D)

    # Free quanta in lunch window
    free_lunch = lunch_window_size - occ_in_lunch  # (N, G, D)

    # Penalty for insufficient free
    lunch_deficit = np.maximum(sdata.lunch_min_quanta - free_lunch, 0)  # (N, G, D)
    # Only penalize days where the group has classes
    lunch_deficit = np.where(any_occ, lunch_deficit, 0)

    lunch_penalty = (
        lunch_deficit.sum(axis=(1, 2)).astype(np.float64)
        * sdata.lunch_penalty_per_missing
    )
    S += lunch_penalty

    return S
