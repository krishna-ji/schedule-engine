r"""Vectorized soft constraint evaluator over full population tensors.

Replaces the per-individual OOP ``Timetable → Evaluator`` pipeline with
three population-level NumPy kernels that evaluate all $N$ individuals
in a single pass with **zero Python loops over $N$, $E$, or $T$**.

Soft constraints
^^^^^^^^^^^^^^^^

1. **CSC** (Cohort Schedule Contiguity) — gap penalty per group per day.

   For group $g$ on day $d$, let
   $q_{\min}^{g,d}$ and $q_{\max}^{g,d}$ be the first and last occupied
   within-day quanta.  The gap count is:

   .. math::

       \text{gap}_{n,g,d} = \sum_{q=q_{\min}}^{q_{\max}}
         \mathbb{1}[\lnot\text{occ}_{n,g,d,q}]
         \cdot \mathbb{1}[q \notin \mathcal{B}]

   where $\mathcal{B} = \{2, 3, 4\}$ is the floating lunch exclusion set.
   Gaps are weighted by a **density ratio** $\rho_g = L_g / (D \cdot Q_d)$
   so that heavily-loaded groups incur proportionally larger penalties.

2. **FSC** (Faculty Schedule Contiguity) — identical gap computation
   over the instructor dimension $i$ instead of group $g$.

3. **MIP** (Meridian Interval Preservation) — lunch break penalty.

   For each $(n, g, d)$ with any scheduled class, count free quanta in
   the lunch window $\mathcal{W} = \{2, 3, 4\}$:

   .. math::

       \text{lunch\_deficit}_{n,g,d} = \max\bigl(
         \ell_{\min} - (|\mathcal{W}| - |\text{occ} \cap \mathcal{W}|),\; 0
       \bigr)

HPC notes
---------
- Occupancy is built via ``np.bincount`` on linearised flat indices
  $k = n \cdot G \cdot D \cdot Q_d + g \cdot D \cdot Q_d + d \cdot Q_d + w$,
  then reshaped into a 4-D boolean tensor
  $\text{occ} \in \{0,1\}^{N \times G \times D \times Q_d}$.
- Min/max within-day quantum detection uses ``np.where`` masking
  with sentinel values ($Q_d$ for min, $-1$ for max) followed by
  axis-3 reduction — no Python loops.
- Total memory footprint: $O(N \cdot G \cdot D \cdot Q_d)$ for
  group occupancy + $O(N \cdot I \cdot D \cdot Q_d)$ for
  instructor occupancy.

API
---
prepare_soft_vectorized_data(pkl_data) -> SoftVectorizedData
eval_soft_vectorized(X, sdata) -> S  # shape ``(N,)`` float64
eval_soft_vectorized_breakdown(X, sdata) -> (S, {"CSC": ..., "FSC": ..., "MIP": ...})
evaluate_paired_cohorts_vectorized(X, lookups) -> penalty  # shape ``(N,)`` float64
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default time system constants (6 days, 7 quanta/day)
_DEFAULT_N_DAYS = 6
_DEFAULT_QUANTA_PER_DAY = 7
_DEFAULT_BREAK_WITHIN_DAY = {
    2,
    3,
    4,
}  # floating lunch exclusion: quanta 2-4 (12:00-15:00)
_DEFAULT_BREAK_WINDOW = {2, 3, 4}  # lunch window: within-day quanta 2-4 (12:00-15:00)


@dataclass(frozen=True, slots=True)
class SoftVectorizedData:
    r"""Immutable precomputed arrays for vectorized soft evaluation.

    Constructed once via ``prepare_soft_vectorized_data`` and reused
    across all generations.  All arrays are contiguous and typed for
    optimal NumPy dispatch.

    Expansion arrays
    ^^^^^^^^^^^^^^^^
    Events are expanded into quantum-level tuples, yielding two
    families:

    - **Instructor expansion** ($Q = \sum_e d_e$ entries):
      ``(exp_event[q'], exp_offset[q'])`` — maps expanded quantum
      $q'$ back to its owning event and within-event offset.

    - **Group expansion** ($GQ = \sum_e d_e \cdot |G_e|$ entries):
      ``(grp_exp_event, grp_exp_offset, grp_exp_group)`` — one entry
      per (event, quantum, group) triple.

    These arrays eliminate per-event Python loops during occupancy
    tensor construction.
    """

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
    r"""Evaluate soft constraints for the full population in one pass.

    Computes $S_n = \text{CSC}_n + \text{FSC}_n + \text{MIP}_n$ for
    each individual $n$ using population-level 4-D occupancy tensors.

    Day-boundary clamping is applied first: if $d_e \le Q_d$ but
    $t_e + d_e > \lceil t_e / Q_d \rceil \cdot Q_d$, the start is
    shifted backward to prevent cross-day spill.

    Parameters
    ----------
    X : ndarray, shape ``(N, 3*E)``, int
        Population matrix (interleaved ``[I, R, T]`` per event).
    sdata : SoftVectorizedData
        Precomputed expansion arrays and configuration.

    Returns
    -------
    S : ndarray, shape ``(N,)``, float64
        Total soft penalty per individual.

    Complexity
    ----------
    $O(N \cdot (GQ + Q + G \cdot D \cdot Q_d + I \cdot D \cdot Q_d))$.
    Dominated by the 4-D boolean operations on occupancy tensors.
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

    # ── Density-Aware Compactness: scale gap penalty by group load ──
    group_load = np.bincount(sdata.grp_exp_group, minlength=n_groups)
    density_ratio = group_load.astype(np.float64) / (n_days * qpd)
    density_scale = density_ratio[np.newaxis, :, np.newaxis]  # (1, G, 1)

    student_compactness = (gap * density_scale).sum(axis=(1, 2)).astype(
        np.float64
    ) * sdata.gap_penalty_per_quantum
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


# Short labels for soft constraint components (Academic Nomenclature)
SOFT_COMPONENT_NAMES: list[str] = [
    "CSC",  # Cohort Schedule Contiguity
    "FSC",  # Faculty Schedule Contiguity
    "MIP",  # Meridian Interval Preservation
]


def eval_soft_vectorized_breakdown(
    X: np.ndarray,
    sdata: SoftVectorizedData,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Like eval_soft_vectorized but also returns per-component arrays.

    Returns
    -------
    S : ndarray, shape (N,), float64  — total soft penalty
    breakdown : dict[str, ndarray shape (N,)]
        Keys: ``CSC``, ``FSC``, ``MIP``
    """
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]
    n_groups = sdata.n_groups
    n_inst = sdata.n_instructors
    n_days = sdata.n_days
    qpd = sdata.quanta_per_day

    inst_assign = X[:, 0::3]
    time_assign = X[:, 2::3].copy()

    durations = sdata.durations
    day_offsets_e = (time_assign // qpd) * qpd
    end_of_day_e = day_offsets_e + qpd
    spills = (durations[np.newaxis, :] <= qpd) & (
        time_assign + durations[np.newaxis, :] > end_of_day_e
    )
    clamped_start = np.maximum(day_offsets_e, end_of_day_e - durations[np.newaxis, :])
    time_assign = np.where(spills, clamped_start, time_assign)

    # ── 1. Student compactness ────────────────────────────────────
    GQ = sdata.GQ
    grp_starts = time_assign[:, sdata.grp_exp_event]
    grp_quanta = grp_starts + sdata.grp_exp_offset[np.newaxis, :]
    grp_days = np.clip(grp_quanta // qpd, 0, n_days - 1)
    grp_within = grp_quanta % qpd

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

    # ── Density-Aware Compactness: scale gap penalty by group load ──
    group_load = np.bincount(sdata.grp_exp_group, minlength=n_groups)
    density_ratio = group_load.astype(np.float64) / (n_days * qpd)
    density_scale = density_ratio[np.newaxis, :, np.newaxis]  # (1, G, 1)

    student_compactness = (gap * density_scale).sum(axis=(1, 2)).astype(
        np.float64
    ) * sdata.gap_penalty_per_quantum

    # ── 2. Instructor compactness ─────────────────────────────────
    Q = sdata.Q
    inst_starts = time_assign[:, sdata.exp_event]
    inst_quanta = inst_starts + sdata.exp_offset[np.newaxis, :]
    inst_ids = inst_assign[:, sdata.exp_event]

    inst_days = np.clip(inst_quanta // qpd, 0, n_days - 1)
    inst_within = inst_quanta % qpd

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
    occ_masked_min_i = np.where(occ_i, qr4, qpd)
    occ_masked_max_i = np.where(occ_i, qr4, -1)
    min_q_i = occ_masked_min_i.min(axis=3)
    max_q_i = occ_masked_max_i.max(axis=3)

    in_span_i = (qr4 >= min_q_i[:, :, :, np.newaxis]) & (
        qr4 <= max_q_i[:, :, :, np.newaxis]
    )
    gap_mask_i = in_span_i & ~occ_i & ~break_mask[np.newaxis, np.newaxis, np.newaxis, :]
    gap_i = gap_mask_i.sum(axis=3).astype(np.int32)
    gap_i = np.where(any_occ_i & (occ_count_i >= 2), gap_i, 0)

    instructor_compactness = (
        gap_i.sum(axis=(1, 2)).astype(np.float64) * sdata.gap_penalty_per_quantum
    )

    # ── 3. Student lunch break ────────────────────────────────────
    lunch_mask_arr = sdata.lunch_window
    lunch_window_size = int(lunch_mask_arr.sum())
    occ_in_lunch = (occ & lunch_mask_arr[np.newaxis, np.newaxis, np.newaxis, :]).sum(
        axis=3
    )
    free_lunch = lunch_window_size - occ_in_lunch
    lunch_deficit = np.maximum(sdata.lunch_min_quanta - free_lunch, 0)
    lunch_deficit = np.where(any_occ, lunch_deficit, 0)
    lunch_penalty = (
        lunch_deficit.sum(axis=(1, 2)).astype(np.float64)
        * sdata.lunch_penalty_per_missing
    )

    S = student_compactness + instructor_compactness + lunch_penalty

    breakdown = {
        "CSC": student_compactness,
        "FSC": instructor_compactness,
        "MIP": lunch_penalty,
    }
    return S, breakdown


# ------------------------------------------------------------------
# Paired Cohort Practical Alignment (group-level XOR penalty)
# ------------------------------------------------------------------


def evaluate_paired_cohorts_vectorized(
    X: np.ndarray,
    lookups: VectorizedLookups,
) -> np.ndarray:
    r"""Compute symmetric sub-cohort practical alignment penalty (SSCP).

    For every sibling subgroup pair $(A, B)$, builds a practical-event
    occupancy tensor $\mathbf{P} \in \{0,1\}^{N \times G \times T}$
    and penalises the symmetric difference (XOR) of their occupancy
    rows:

    .. math::

        \text{SSCP}_n = \sum_{(A,B) \in \mathcal{P}}
          \max\Bigl(
            \sum_{q=0}^{T-1} \mathbf{P}_{n,A,q} \oplus \mathbf{P}_{n,B,q}
            - |L_A - L_B|,\; 0
          \Bigr)

    where $L_A = \sum_q \mathbf{P}_{n,A,q}$ is the practical load of
    subgroup $A$, and $|L_A - L_B|$ is the **unavoidable** load
    difference subtracted as a floor.

    The occupancy tensor is built via ``np.bincount`` on linearised
    $(n, g, q)$ keys expanded from practical events only (filtered
    by ``is_practical`` mask).  No Python loops over $N$.

    Parameters
    ----------
    X : ndarray, shape ``(N, 3*E)``, int
        Population matrix.
    lookups : VectorizedLookups
        Must contain ``cohort_subgroup_pairs`` (S, 2),
        ``is_practical`` (E,), ``durations`` (E,), and group
        membership arrays.

    Returns
    -------
    penalty : ndarray, shape ``(N,)``, float64
        Per-individual XOR-based alignment penalty.

    Complexity
    ----------
    $O(N \cdot PGQ + N \cdot S \cdot T)$ where $PGQ$ is the practical
    group-quantum expansion size and $S$ is the number of subgroup pairs.
    """
    pairs = lookups.cohort_subgroup_pairs  # (S, 2) int32
    if pairs.shape[0] == 0:
        N = X.shape[0] if X.ndim == 2 else 1
        return np.zeros(N, dtype=np.float64)

    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]

    is_prac = lookups.is_practical  # (E,) bool
    durations = lookups.durations  # (E,) int32
    n_groups = lookups.n_groups
    T_total = int(lookups.durations.sum() * 0 + 42)  # T = 42 (6 days × 7 QPD)
    # Use the module-level T from bitset_time imported at top of this file
    from .bitset_time import T as T_val

    T_total = T_val

    # -- Filter to practical events only --
    prac_mask = np.flatnonzero(is_prac)  # indices of practical events
    if len(prac_mask) == 0:
        return np.zeros(N, dtype=np.float64)

    # Get group expansion arrays but only for practical events
    # We need: for each practical event, its groups, its start time, its duration
    # Build occupancy tensor is_prac_occ[N, n_groups, T_total] via scatter

    # Practical event starts: (N, n_prac)
    prac_starts = X[:, prac_mask * 3 + 2]  # (N, n_prac)
    prac_durs = durations[prac_mask]  # (n_prac,)

    # Group membership for practical events
    grp_mem_prac = lookups.group_membership[prac_mask]  # (n_prac, n_groups) bool

    # Expand: for each (prac_event, quantum_offset, group) → mark occupancy
    # Build expansion: prac_event × duration offsets
    n_prac = len(prac_mask)
    # Create offset arrays for each practical event
    max_dur = int(prac_durs.max()) if n_prac > 0 else 0

    # Expand practical events by their durations
    prac_rep = np.repeat(np.arange(n_prac, dtype=np.int32), prac_durs)  # (PQ,)
    _pcum = np.zeros(n_prac + 1, dtype=np.int64)
    np.cumsum(prac_durs, out=_pcum[1:])
    _pstarts = np.repeat(_pcum[:n_prac], prac_durs)
    prac_offsets = np.arange(int(prac_durs.sum()), dtype=np.int32) - _pstarts.astype(
        np.int32
    )  # (PQ,)

    # Groups for each expanded entry
    prac_groups_per_event = []
    prac_event_idx_per_group = []
    for pi in range(n_prac):
        gidxs = np.flatnonzero(grp_mem_prac[pi])  # groups this prac event belongs to
        for g in gidxs:
            prac_groups_per_event.append(g)
            prac_event_idx_per_group.append(pi)

    if not prac_groups_per_event:
        return np.zeros(N, dtype=np.float64)

    prac_g_arr = np.array(prac_groups_per_event, dtype=np.int32)  # (PG,)
    prac_ei_arr = np.array(prac_event_idx_per_group, dtype=np.int32)  # (PG,)
    prac_d_arr = prac_durs[prac_ei_arr]  # (PG,) durations

    # Now expand by duration: PG × dur → PGQ entries
    pgq_g = np.repeat(prac_g_arr, prac_d_arr)  # (PGQ,) group index
    pgq_ei = np.repeat(prac_ei_arr, prac_d_arr)  # (PGQ,) prac event index
    _pgcum = np.zeros(len(prac_g_arr) + 1, dtype=np.int64)
    np.cumsum(prac_d_arr, out=_pgcum[1:])
    _pgstarts = np.repeat(_pgcum[:-1], prac_d_arr)
    pgq_off = np.arange(int(prac_d_arr.sum()), dtype=np.int32) - _pgstarts.astype(
        np.int32
    )  # (PGQ,) offset

    # Absolute quanta for each individual: prac_starts[:, pgq_ei] + pgq_off
    # prac_starts is (N, n_prac), pgq_ei indexes into n_prac
    abs_quanta = prac_starts[:, pgq_ei] + pgq_off[np.newaxis, :]  # (N, PGQ)

    # Clamp to valid range
    abs_quanta = np.clip(abs_quanta, 0, T_total - 1)

    # Build flat index: n * (n_groups * T_total) + g * T_total + q
    stride = n_groups * T_total
    n_idx = np.repeat(np.arange(N, dtype=np.int64), len(pgq_g))
    g_flat = np.tile(pgq_g, N).astype(np.int64)
    q_flat = abs_quanta.ravel()

    flat_idx = n_idx * stride + g_flat * T_total + q_flat

    occ_flat = np.bincount(flat_idx.astype(np.int64), minlength=N * stride)
    is_prac_occ = occ_flat.reshape(N, n_groups, T_total) > 0  # (N, G, T) bool

    # -- XOR penalty between paired subgroups --
    left_idx = pairs[:, 0]  # (S,) group indices
    right_idx = pairs[:, 1]  # (S,) group indices

    left_occ = is_prac_occ[:, left_idx, :]  # (N, S, T)
    right_occ = is_prac_occ[:, right_idx, :]  # (N, S, T)

    xor_mismatch = left_occ ^ right_occ  # (N, S, T) bool
    raw_xor_sum = xor_mismatch.sum(axis=2).astype(np.float64)  # (N, S)

    # ── Subtract unavoidable subcohort load difference (floor) ──
    left_load = left_occ.sum(axis=2).astype(np.float64)  # (N, S)
    right_load = right_occ.sum(axis=2).astype(np.float64)  # (N, S)
    unavoidable_diff = np.abs(left_load - right_load)  # (N, S)
    penalty_per_pair = np.maximum(raw_xor_sum - unavoidable_diff, 0.0)  # (N, S)

    return penalty_per_pair.sum(axis=1)  # (N,)
