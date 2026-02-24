"""Batch fast evaluator using bitset occupancy (uint64, T=42).

.. deprecated::
    This module is superseded by ``fast_evaluator_vectorized.py`` which is
    4-5× faster.  Retained for equivalence testing only.  The hot-path in
    ``SchedulingProblem._evaluate()`` no longer calls this module.

Exposes:
    fast_evaluate_hard_batch(X, data) -> G
        X: shape (N, 3E) interleaved chromosome matrix
        data: precomputed BatchEvalData (from prepare_batch_data)
        G: shape (N, 8) per-constraint violation counts

Constraint order matches fast_evaluator_vectorized.py exactly:
    0: student_group_exclusivity
    1: instructor_exclusivity
    2: room_exclusivity
    3: instructor_qualifications
    4: room_suitability
    5: instructor_time_availability
    6: course_completeness (always 0)
    7: sibling_same_day (not computed in batch fallback — always 0)

Violation counting convention (must match original):
    Exclusivity: For each (resource, quantum), if k events overlap, count k-1.
    Equivalently: for each event placed, count how many already-placed quanta
    it overlaps with. Summing this over all events gives the same total as
    sum(len(bucket)-1 for bucket if len>1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bitset_time import FULL_MASK, T, mask_from_interval, mask_from_quanta

# Pre-compute mask LUT: _MASK_LUT[dur][start] = uint64 mask
# dur in [0..max_dur], start in [0..T-1]
_MAX_DUR = 12  # largest duration we support (actual max is 10)
_MASK_LUT: np.ndarray = np.zeros((_MAX_DUR + 1, T), dtype=np.uint64)
for _d in range(_MAX_DUR + 1):
    for _s in range(T):
        _MASK_LUT[_d, _s] = mask_from_interval(_s, _d)

# Byte-level popcount LUT (inlined to avoid function call overhead)
_PC_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _fast_popcount(v: int | np.uint64) -> int:
    """Inline popcount using byte LUT — avoids bin() string overhead."""
    vi = int(v)
    return int(
        _PC_LUT[vi & 0xFF]
        + _PC_LUT[(vi >> 8) & 0xFF]
        + _PC_LUT[(vi >> 16) & 0xFF]
        + _PC_LUT[(vi >> 24) & 0xFF]
        + _PC_LUT[(vi >> 32) & 0xFF]
        + _PC_LUT[(vi >> 40) & 0xFF]
    )


# Hard constraint names in canonical order (Academic Nomenclature)
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


@dataclass(frozen=True, slots=True)
class BatchEvalData:
    """Precomputed data structures for fast batch evaluation.

    Built once from the pkl data; reused for every batch evaluation call.
    """

    n_events: int
    n_rooms: int
    n_instructors: int
    n_groups: int

    # Per-event arrays (length E)
    durations: np.ndarray  # int32, num_quanta per event
    event_group_indices: list[list[int]]  # group indices per event

    # Allowed sets as Python sets for O(1) membership
    allowed_instructors_sets: list[set[int]]
    allowed_rooms_sets: list[set[int]]

    # Boolean membership arrays: shape (E, max_idx)
    # inst_allowed[e, i] = True if instructor i is allowed for event e
    inst_allowed: np.ndarray  # bool, shape (E, n_instructors)
    room_allowed: np.ndarray  # bool, shape (E, n_rooms)

    # Availability bitsets: indexed by entity id → uint64 mask
    # FULL_MASK means always available; otherwise only listed quanta
    inst_avail_masks: np.ndarray  # uint64 shape (n_instructors,)
    room_avail_masks: np.ndarray  # uint64 shape (n_rooms,)

    # Group string to index mapping
    group_to_idx: dict[str, int] = field(default_factory=dict)


def prepare_batch_data(pkl_data: dict) -> BatchEvalData:
    """Precompute BatchEvalData from events_with_domains.pkl dict."""
    events = pkl_data["events"]
    E = len(events)

    # Build group index mapping
    all_group_ids: set[str] = set()
    for ev in events:
        all_group_ids.update(ev["group_ids"])
    group_to_idx = {gid: i for i, gid in enumerate(sorted(all_group_ids))}
    n_groups = len(group_to_idx)

    # Per-event data
    durations = np.array([ev["num_quanta"] for ev in events], dtype=np.int32)
    event_group_indices = [
        [group_to_idx[gid] for gid in ev["group_ids"]] for ev in events
    ]

    # Allowed sets
    allowed_instructors_sets = [set(ai) for ai in pkl_data["allowed_instructors"]]
    allowed_rooms_sets = [set(ar) for ar in pkl_data["allowed_rooms"]]

    # Instructor count: max index + 1
    n_instructors = (
        max((max(ai) for ai in pkl_data["allowed_instructors"] if ai), default=0) + 1
    )
    n_rooms = max((max(ar) for ar in pkl_data["allowed_rooms"] if ar), default=0) + 1

    # Boolean membership arrays
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

    # Availability masks
    inst_avail_raw = pkl_data.get("instructor_available_quanta", {})
    inst_avail_masks = np.full(n_instructors, FULL_MASK, dtype=np.uint64)
    for idx, slots in inst_avail_raw.items():
        idx = int(idx)
        if idx < n_instructors:
            if slots is None:
                inst_avail_masks[idx] = FULL_MASK
            else:
                inst_avail_masks[idx] = mask_from_quanta(slots)

    room_avail_raw = pkl_data.get("room_available_quanta", {})
    room_avail_masks = np.full(n_rooms, FULL_MASK, dtype=np.uint64)
    for idx, slots in room_avail_raw.items():
        idx = int(idx)
        if idx < n_rooms:
            if slots is None:
                room_avail_masks[idx] = FULL_MASK
            else:
                room_avail_masks[idx] = mask_from_quanta(slots)

    return BatchEvalData(
        n_events=E,
        n_rooms=n_rooms,
        n_instructors=n_instructors,
        n_groups=n_groups,
        durations=durations,
        event_group_indices=event_group_indices,
        allowed_instructors_sets=allowed_instructors_sets,
        allowed_rooms_sets=allowed_rooms_sets,
        inst_allowed=inst_allowed,
        room_allowed=room_allowed,
        inst_avail_masks=inst_avail_masks,
        room_avail_masks=room_avail_masks,
        group_to_idx=group_to_idx,
    )


def fast_evaluate_hard_batch(
    X: np.ndarray,
    data: BatchEvalData,
) -> np.ndarray:
    """Evaluate hard constraints for a batch of individuals.

    Parameters
    ----------
    X : ndarray, shape (N, 3*E)
        Population matrix. Each row is an interleaved chromosome.
    data : BatchEvalData
        Precomputed from prepare_batch_data().

    Returns
    -------
    G : ndarray, shape (N, 8)
        Per-constraint violation counts. Column order matches
        HARD_CONSTRAINT_NAMES.
    """
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]
    G = np.zeros((N, 8), dtype=np.int64)

    E = data.n_events
    durations = data.durations
    event_groups = data.event_group_indices
    inst_allowed = data.inst_allowed
    room_allowed = data.room_allowed
    inst_avail = data.inst_avail_masks
    room_avail = data.room_avail_masks
    n_rooms = data.n_rooms
    n_inst = data.n_instructors
    n_groups = data.n_groups
    mask_lut = _MASK_LUT
    fpc = _fast_popcount

    for n in range(N):
        row = X[n]
        inst_assign = row[0::3]  # shape (E,)
        room_assign = row[1::3]
        time_assign = row[2::3]

        # Occupancy masks per resource
        room_occ = np.zeros(n_rooms, dtype=np.uint64)
        inst_occ = np.zeros(n_inst, dtype=np.uint64)
        group_occ = np.zeros(n_groups, dtype=np.uint64)

        group_viol = 0
        inst_viol = 0
        room_viol = 0
        qual_viol = 0
        suit_viol = 0
        inst_avail_viol = 0

        for e in range(E):
            i_idx = int(inst_assign[e])
            r_idx = int(room_assign[e])
            t = int(time_assign[e])
            dur = int(durations[e])

            event_mask = mask_lut[dur, t]

            # --- Exclusivity: count overlapping quanta with already-placed ---
            # Room
            overlap = int(room_occ[r_idx] & event_mask)
            if overlap:
                room_viol += fpc(overlap)
            room_occ[r_idx] |= event_mask

            # Instructor
            overlap = int(inst_occ[i_idx] & event_mask)
            if overlap:
                inst_viol += fpc(overlap)
            inst_occ[i_idx] |= event_mask

            # Groups
            for gidx in event_groups[e]:
                overlap = int(group_occ[gidx] & event_mask)
                if overlap:
                    group_viol += fpc(overlap)
                group_occ[gidx] |= event_mask

            # --- Qualification / Suitability (boolean array lookup) ---
            if not inst_allowed[e, i_idx]:
                qual_viol += 1
            if not room_allowed[e, r_idx]:
                suit_viol += 1

            # --- Time availability ---
            unavail = int(event_mask & ~inst_avail[i_idx])
            if unavail:
                inst_avail_viol += fpc(unavail)

        G[n, 0] = group_viol
        G[n, 1] = inst_viol
        G[n, 2] = room_viol
        G[n, 3] = qual_viol
        G[n, 4] = suit_viol
        G[n, 5] = inst_avail_viol
        G[n, 6] = 0  # course_completeness always 0
        G[n, 7] = 0  # sibling_same_day — not computed in batch fallback

    return G


def fast_evaluate_hard_single(
    x: np.ndarray,
    data: BatchEvalData,
) -> dict[str, int]:
    """Evaluate a single individual, returning a dict matching original API."""
    G = fast_evaluate_hard_batch(x.reshape(1, -1), data)
    return {name: int(G[0, i]) for i, name in enumerate(HARD_CONSTRAINT_NAMES)}
