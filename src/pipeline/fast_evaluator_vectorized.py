"""True population-level vectorized hard-constraint evaluator.

Eliminates per-individual Python loops by expanding events into quanta,
then using ``np.add.at`` / ``np.bincount`` over the *entire population*
in a single numpy call.

Public API
----------
    prepare_vectorized_data(pkl_data) -> VectorizedEvalData
    fast_evaluate_hard_vectorized(X, vdata) -> G   # shape (N, 8)

The result is numerically identical to ``fast_evaluate_hard_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bitset_time import T

# Constraint column order (Academic Nomenclature)
HARD_CONSTRAINT_NAMES = [
    "CTE",  # Cohort Temporal Exclusivity
    "FTE",  # Faculty Temporal Exclusivity
    "SRE",  # Spatial Resource Exclusivity
    "FPC",  # Faculty Pedagogical Congruence
    "FFC",  # Facility Feature Congruence
    "FCA",  # Faculty Chronological Availability
    "CQF",  # Curriculum Quanta Fulfillment
    "ICTD",  # Intra-Course Temporal Dispersion
]


# ------------------------------------------------------------------
# Precomputed data
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VectorizedEvalData:
    """All precomputed arrays needed by the vectorized kernel.

    Expansion arrays
    ~~~~~~~~~~~~~~~~
    Each event *e* with duration *d* is expanded into *d* quanta entries.
    ``exp_event[k]`` gives the event index, ``exp_offset[k]`` gives the
    quantum offset (0 … d-1) for the *k*-th expanded entry.

    Total entries per individual: ``Q = sum(durations)``.

    For groups each event is further expanded by its group count:
    ``grp_exp_event``, ``grp_exp_offset``, ``grp_exp_group`` with
    ``GQ = sum(dur_e * n_groups_e)`` entries per individual.

    Membership & availability
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ``inst_allowed[e, i]``  – True if instructor *i* is allowed for event *e*
    ``room_allowed[e, r]``  – True if room *r* is allowed for event *e*
    ``inst_avail_bool[i, q]`` – True if instructor *i* available at quantum *q*
    ``room_avail_bool[r, q]`` – True if room *r* available at quantum *q*
    """

    n_events: int
    n_rooms: int
    n_instructors: int
    n_groups: int

    # Per-event durations, shape (E,)
    durations: np.ndarray

    # Room / instructor expansion: Q entries each
    Q: int  # sum(durations)
    exp_event: np.ndarray  # int32  (Q,)
    exp_offset: np.ndarray  # int32  (Q,)

    # Group expansion: GQ entries
    GQ: int  # sum(dur_e * n_groups_e)
    grp_exp_event: np.ndarray  # int32  (GQ,)
    grp_exp_offset: np.ndarray  # int32  (GQ,)
    grp_exp_group: np.ndarray  # int32  (GQ,)

    # Membership boolean arrays
    inst_allowed: np.ndarray  # bool (E, n_instructors)
    room_allowed: np.ndarray  # bool (E, n_rooms)

    # Availability boolean arrays
    inst_avail_bool: np.ndarray  # bool (n_instructors, T)
    room_avail_bool: np.ndarray  # bool (n_rooms, T)


def prepare_vectorized_data(pkl_data: dict) -> VectorizedEvalData:
    """Build VectorizedEvalData from the events_with_domains.pkl dict."""
    events = pkl_data["events"]
    E = len(events)

    # Group index mapping
    all_gids: set[str] = set()
    for ev in events:
        all_gids.update(ev["group_ids"])
    group_to_idx = {gid: i for i, gid in enumerate(sorted(all_gids))}
    n_groups = len(group_to_idx)

    durations = np.array([ev["num_quanta"] for ev in events], dtype=np.int32)
    event_group_indices = [
        [group_to_idx[gid] for gid in ev["group_ids"]] for ev in events
    ]

    # Dimensions
    n_instructors = (
        max((max(ai) for ai in pkl_data["allowed_instructors"] if ai), default=0) + 1
    )
    n_rooms = max((max(ar) for ar in pkl_data["allowed_rooms"] if ar), default=0) + 1

    # ------------------------------------------------------------------
    # Room / instructor expansion
    # ------------------------------------------------------------------
    Q = int(durations.sum())
    exp_event = np.empty(Q, dtype=np.int32)
    exp_offset = np.empty(Q, dtype=np.int32)
    pos = 0
    for e in range(E):
        d = int(durations[e])
        exp_event[pos : pos + d] = e
        exp_offset[pos : pos + d] = np.arange(d, dtype=np.int32)
        pos += d

    # ------------------------------------------------------------------
    # Group expansion
    # ------------------------------------------------------------------
    GQ = sum(int(durations[e]) * len(event_group_indices[e]) for e in range(E))
    grp_exp_event = np.empty(GQ, dtype=np.int32)
    grp_exp_offset = np.empty(GQ, dtype=np.int32)
    grp_exp_group = np.empty(GQ, dtype=np.int32)
    pos = 0
    for e in range(E):
        d = int(durations[e])
        gidxs = event_group_indices[e]
        for gidx in gidxs:
            grp_exp_event[pos : pos + d] = e
            grp_exp_offset[pos : pos + d] = np.arange(d, dtype=np.int32)
            grp_exp_group[pos : pos + d] = gidx
            pos += d

    # ------------------------------------------------------------------
    # Membership boolean arrays
    # ------------------------------------------------------------------
    inst_allowed = np.zeros((E, n_instructors), dtype=np.bool_)
    for e, ai in enumerate(pkl_data["allowed_instructors"]):
        for idx in ai:
            if idx < n_instructors:
                inst_allowed[e, idx] = True

    room_allowed = np.zeros((E, n_rooms), dtype=np.bool_)
    for e, ar in enumerate(pkl_data["allowed_rooms"]):
        for idx in ar:
            if idx < n_rooms:
                room_allowed[e, idx] = True

    # ------------------------------------------------------------------
    # Availability boolean arrays
    # ------------------------------------------------------------------
    inst_avail_bool = np.ones((n_instructors, T), dtype=np.bool_)
    for idx, slots in pkl_data.get("instructor_available_quanta", {}).items():
        idx = int(idx)
        if idx < n_instructors and slots is not None:
            inst_avail_bool[idx, :] = False
            for q in slots:
                if 0 <= q < T:
                    inst_avail_bool[idx, q] = True

    room_avail_bool = np.ones((n_rooms, T), dtype=np.bool_)
    for idx, slots in pkl_data.get("room_available_quanta", {}).items():
        idx = int(idx)
        if idx < n_rooms and slots is not None:
            room_avail_bool[idx, :] = False
            for q in slots:
                if 0 <= q < T:
                    room_avail_bool[idx, q] = True

    return VectorizedEvalData(
        n_events=E,
        n_rooms=n_rooms,
        n_instructors=n_instructors,
        n_groups=n_groups,
        durations=durations,
        Q=Q,
        exp_event=exp_event,
        exp_offset=exp_offset,
        GQ=GQ,
        grp_exp_event=grp_exp_event,
        grp_exp_offset=grp_exp_offset,
        grp_exp_group=grp_exp_group,
        inst_allowed=inst_allowed,
        room_allowed=room_allowed,
        inst_avail_bool=inst_avail_bool,
        room_avail_bool=room_avail_bool,
    )


# ------------------------------------------------------------------
# Vectorized evaluation kernel
# ------------------------------------------------------------------


def fast_evaluate_hard_vectorized(
    X: np.ndarray,
    vdata: VectorizedEvalData,
) -> np.ndarray:
    """Evaluate hard constraints for a full population — no per-individual loops.

    Parameters
    ----------
    X : ndarray, shape (N, 3*E)
        Population matrix (interleaved chromosomes).
    vdata : VectorizedEvalData
        Precomputed from ``prepare_vectorized_data()``.

    Returns
    -------
    G : ndarray, shape (N, 8), int64
        Per-constraint violation counts.  Column order matches
        ``HARD_CONSTRAINT_NAMES``.
    """
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]
    E = vdata.n_events

    # ---- Extract sliced views: shape (N, E) ----
    inst_assign = X[:, 0::3]
    room_assign = X[:, 1::3]
    time_assign = X[:, 2::3]

    # ---- Pre-fetch expansion arrays ----
    Q = vdata.Q
    exp_event = vdata.exp_event  # (Q,)
    exp_offset = vdata.exp_offset  # (Q,)
    GQ = vdata.GQ
    grp_exp_event = vdata.grp_exp_event  # (GQ,)
    grp_exp_offset = vdata.grp_exp_offset  # (GQ,)
    grp_exp_group = vdata.grp_exp_group  # (GQ,)
    n_rooms = vdata.n_rooms
    n_inst = vdata.n_instructors
    n_groups = vdata.n_groups

    # ---- Expand starts to quanta: shape (N, Q) ----
    starts = time_assign[:, exp_event]  # (N, Q)
    quanta = starts + exp_offset[np.newaxis, :]  # (N, Q)

    # ---- Resource assignments at expanded positions ----
    rooms = room_assign[:, exp_event]  # (N, Q)
    insts = inst_assign[:, exp_event]  # (N, Q)

    # ---- Flat index arrays for scatter-add ----
    n_idx_ri = np.repeat(np.arange(N, dtype=np.int64), Q)  # (N*Q,)
    q_flat = quanta.ravel()  # (N*Q,)
    r_flat = rooms.ravel()  # (N*Q,)
    i_flat = insts.ravel()  # (N*Q,)

    # ================================================================
    # Room exclusivity: violation = Q - #unique(room, quantum) cells
    # Identity: sum(max(cnt-1,0)) = total_entries - count(cnt>0)
    # Avoids large int64 temporaries from maximum/subtract.
    # ================================================================
    stride_r = n_rooms * T
    flat_idx_r = n_idx_ri * stride_r + r_flat * T + q_flat  # (N*Q,)
    room_cnt = np.bincount(flat_idx_r, minlength=N * stride_r)  # (N*stride_r,)
    room_viol = Q - (room_cnt.reshape(N, stride_r) > 0).sum(axis=1)  # (N,)

    # ================================================================
    # Instructor exclusivity: same identity as room
    # ================================================================
    stride_i = n_inst * T
    flat_idx_i = n_idx_ri * stride_i + i_flat * T + q_flat
    inst_cnt = np.bincount(flat_idx_i, minlength=N * stride_i)  # (N*stride_i,)
    inst_viol = Q - (inst_cnt.reshape(N, stride_i) > 0).sum(axis=1)  # (N,)

    # ================================================================
    # Group exclusivity (separate expansion for multi-group events)
    # ================================================================
    grp_starts = time_assign[:, grp_exp_event]  # (N, GQ)
    grp_quanta = grp_starts + grp_exp_offset[np.newaxis, :]  # (N, GQ)

    n_idx_g = np.repeat(np.arange(N, dtype=np.int64), GQ)  # (N*GQ,)
    gq_flat = grp_quanta.ravel()  # (N*GQ,)
    gg_flat = np.tile(grp_exp_group, N)  # (N*GQ,)

    stride_g = n_groups * T
    flat_idx_g = n_idx_g * stride_g + gg_flat * T + gq_flat
    group_cnt = np.bincount(flat_idx_g, minlength=N * stride_g)  # (N*stride_g,)
    group_viol = GQ - (group_cnt.reshape(N, stride_g) > 0).sum(axis=1)  # (N,)

    # ================================================================
    # Instructor qualifications: bool lookup
    # ================================================================
    e_idx = np.arange(E, dtype=np.int64)
    # inst_allowed[event, instructor] -> bool;  broadcast over N
    qual_ok = vdata.inst_allowed[e_idx[np.newaxis, :], inst_assign]  # (N, E)
    qual_viol = np.sum(~qual_ok, axis=1)  # (N,)

    # ================================================================
    # Room suitability: bool lookup
    # ================================================================
    suit_ok = vdata.room_allowed[e_idx[np.newaxis, :], room_assign]  # (N, E)
    suit_viol = np.sum(~suit_ok, axis=1)  # (N,)

    # ================================================================
    # Instructor time-availability
    # ================================================================
    # inst_avail_bool[inst, quantum] -> bool
    unavail_inst = ~vdata.inst_avail_bool[i_flat, q_flat]  # (N*Q,) bool
    inst_avail_viol = np.bincount(
        n_idx_ri,
        weights=unavail_inst.view(np.uint8).astype(np.float64),
        minlength=N,
    ).astype(np.int64)

    # ================================================================
    # Assemble G
    # ================================================================
    G = np.empty((N, 8), dtype=np.int64)
    G[:, 0] = group_viol
    G[:, 1] = inst_viol
    G[:, 2] = room_viol
    G[:, 3] = qual_viol
    G[:, 4] = suit_viol
    G[:, 5] = inst_avail_viol
    G[:, 6] = 0  # course_completeness — always 0

    # ================================================================
    # Sibling same-day: penalize sub-sessions of the same course on
    # the same day.  Uses sparse sibling_pairs from VectorizedLookups.
    # ================================================================
    _QPD = 7  # quanta per day (T=42 / 6 days)
    sibling_pairs = getattr(vdata, "sibling_pairs", None)
    if sibling_pairs is not None and len(sibling_pairs) > 0:
        days = time_assign // _QPD  # (N, E)
        sp_i = sibling_pairs[:, 0]  # (P,)
        sp_j = sibling_pairs[:, 1]  # (P,)
        # days[:, sp_i] and days[:, sp_j] are (N, P)
        same_day = days[:, sp_i] == days[:, sp_j]  # (N, P) bool
        G[:, 7] = same_day.sum(axis=1)  # (N,)
    else:
        G[:, 7] = 0

    return G
