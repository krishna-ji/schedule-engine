"""Consolidated vectorized lookup tables for the scheduling engine.

Converts jagged ``events_with_domains.pkl`` data into dense NumPy arrays
suitable for the vectorized evaluator, repair, and operators.

All Python loops run **once at startup** (over E events).  The resulting
arrays enable zero-loop evaluation over the population N.

Public API
----------
    build_vectorized_lookups(pkl_data) -> VectorizedLookups

The returned object is duck-type compatible with ``VectorizedEvalData``
and can be passed directly to ``fast_evaluate_hard_vectorized()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bitset_time import T


@dataclass(frozen=True, slots=True)
class VectorizedLookups:
    """Dense NumPy lookup tables built from events_with_domains.pkl.

    Sections
    --------
    Dimensions & metadata:
        n_events, n_rooms, n_instructors, n_groups, event_metadata, durations

    Expansion arrays (event → quanta):
        Q, exp_event, exp_offset  — for room/instructor constraints
        GQ, grp_exp_event, grp_exp_offset, grp_exp_group — for group constraints

    Membership & availability:
        inst_allowed (E, n_inst)  — qualification boolean
        room_allowed (E, n_rooms) — suitability boolean
        inst_avail_bool (n_inst, T) — time availability
        room_avail_bool (n_rooms, T) — time availability

    Padded domain matrices (for repair & sampling):
        inst_domains (E, max_dom), inst_dom_len (E,)
        room_domains (E, max_dom), room_dom_len (E,)
        time_domains (E, max_dom), time_dom_len (E,)

    Group structures:
        group_membership (E, n_groups)  — event-group membership
        group_conflict_matrix (E, E)    — shared-group boolean
        event_group_lists — per-event group index lists (for repair)
        group_to_idx — string ID → integer index
    """

    # ---- Dimensions ----
    n_events: int
    n_rooms: int
    n_instructors: int
    n_groups: int

    # ---- Per-event metadata ----
    event_metadata: np.ndarray  # (E, 4) int32: [dur, type_enum, n_groups, is_prac]
    durations: np.ndarray  # (E,) int32

    # ---- Expansion: event → quanta ----
    Q: int  # sum(durations)
    exp_event: np.ndarray  # (Q,) int32 — event index
    exp_offset: np.ndarray  # (Q,) int32 — offset 0..d-1

    # ---- Expansion: event → quanta × groups ----
    GQ: int  # sum(dur_e * n_groups_e)
    grp_exp_event: np.ndarray  # (GQ,) int32
    grp_exp_offset: np.ndarray  # (GQ,) int32
    grp_exp_group: np.ndarray  # (GQ,) int32

    # ---- Membership boolean matrices ----
    inst_allowed: np.ndarray  # (E, n_inst) bool
    room_allowed: np.ndarray  # (E, n_rooms) bool

    # ---- Availability boolean matrices ----
    inst_avail_bool: np.ndarray  # (n_inst, T) bool
    room_avail_bool: np.ndarray  # (n_rooms, T) bool

    # ---- Padded domain matrices (repair & sampling) ----
    inst_domains: np.ndarray  # (E, max_inst_dom) int64
    inst_dom_len: np.ndarray  # (E,) int64
    room_domains: np.ndarray  # (E, max_room_dom) int64
    room_dom_len: np.ndarray  # (E,) int64
    time_domains: np.ndarray  # (E, max_time_dom) int64
    time_dom_len: np.ndarray  # (E,) int64

    # ---- Group structures ----
    group_membership: np.ndarray  # (E, n_groups) bool
    group_conflict_matrix: np.ndarray  # (E, E) bool
    event_group_lists: list  # list[list[int]] per event
    group_to_idx: dict  # str -> int

    # ---- Sibling sub-session structures ----
    sibling_matrix: np.ndarray  # (E, E) bool — True if same course sub-sessions
    sibling_pairs: np.ndarray  # (P, 2) int32 — upper-triangle (i, j) pairs
    sibling_event_to_course: np.ndarray  # (E,) int32 — course group id per event
    quanta_per_day: int  # QPD constant (default 7)

    # ---- Paired cohort practical alignment ----
    cohort_event_pairs: (
        np.ndarray
    )  # (P, 2) int32 — (event_A, event_B) practical pairs (legacy)
    cohort_subgroup_pairs: np.ndarray  # (S, 2) int32 — (group_idx_A, group_idx_B) pairs
    is_practical: np.ndarray  # (E,) bool — True if event is a practical


def build_vectorized_lookups(pkl_data: dict) -> VectorizedLookups:
    """Convert jagged pkl domain data into dense NumPy arrays.

    Parameters
    ----------
    pkl_data : dict
        Loaded ``events_with_domains.pkl`` dictionary.

    Returns
    -------
    VectorizedLookups
        All arrays needed by evaluator, repair, and operators.
        Duck-type compatible with ``VectorizedEvalData``.
    """
    events = pkl_data["events"]
    E = len(events)
    ai = pkl_data["allowed_instructors"]  # list[list[int]]
    ar = pkl_data["allowed_rooms"]  # list[list[int]]
    at = pkl_data["allowed_starts"]  # list[list[int]]

    # ---- Dimensions ----
    n_inst = max((max(d) for d in ai if d), default=0) + 1
    n_rooms = max((max(d) for d in ar if d), default=0) + 1

    # ---- Group index mapping ----
    all_gids: set[str] = set()
    for ev in events:
        all_gids.update(ev["group_ids"])
    group_to_idx = {gid: i for i, gid in enumerate(sorted(all_gids))}
    n_groups = len(group_to_idx)

    # ---- Per-event metadata  (E, 4): [dur, type_enum, n_groups, is_prac] ----
    durations = np.array([ev["num_quanta"] for ev in events], dtype=np.int32)
    type_enum = np.array(
        [0 if ev.get("course_type", "theory") == "theory" else 1 for ev in events],
        dtype=np.int32,
    )
    group_counts = np.array([len(ev["group_ids"]) for ev in events], dtype=np.int32)
    is_prac = (type_enum == 1).astype(np.int32)
    event_metadata = np.column_stack([durations, type_enum, group_counts, is_prac])

    # ---- Per-event group indices ----
    event_group_lists: list[list[int]] = [
        [group_to_idx[gid] for gid in ev["group_ids"]] for ev in events
    ]

    # ---- Group membership matrix  (E, n_groups) bool ----
    group_membership = np.zeros((E, n_groups), dtype=np.bool_)
    # Vectorized scatter via row/col pairs
    _gm_rows = np.repeat(
        np.arange(E, dtype=np.int32), [len(g) for g in event_group_lists]
    )
    _gm_cols = (
        np.concatenate([np.array(g, dtype=np.int32) for g in event_group_lists])
        if E > 0
        else np.empty(0, dtype=np.int32)
    )
    group_membership[_gm_rows, _gm_cols] = True

    # ---- Group conflict matrix  (E, E) bool: matmul on uint8 ----
    _gm_u8 = group_membership.astype(np.uint8)  # (E, n_groups)
    group_conflict_matrix = (_gm_u8 @ _gm_u8.T) > 0  # (E, E) bool
    np.fill_diagonal(group_conflict_matrix, False)

    # ---- Sibling matrix  (E, E) bool ----
    # Two events are siblings if they share the same (course_id, course_type,
    # sorted(group_ids)) — i.e. sub-sessions of the same course offering.
    # Build via integer course-group key → cluster id.
    _course_keys: list[tuple] = [
        (ev["course_id"], ev["course_type"], tuple(sorted(ev["group_ids"])))
        for ev in events
    ]
    _unique_keys: dict[tuple, int] = {}
    sibling_event_to_course = np.empty(E, dtype=np.int32)
    for e, ck in enumerate(_course_keys):
        if ck not in _unique_keys:
            _unique_keys[ck] = len(_unique_keys)
        sibling_event_to_course[e] = _unique_keys[ck]

    # Vectorized: sibling_matrix[i,j] = (course_id[i] == course_id[j]) & (i != j)
    sibling_matrix = (
        sibling_event_to_course[:, None] == sibling_event_to_course[None, :]
    )  # (E, E) bool
    np.fill_diagonal(sibling_matrix, False)

    # Sparse upper-triangle pairs for efficient constraint evaluation
    _sib_i, _sib_j = np.nonzero(np.triu(sibling_matrix, k=1))
    sibling_pairs = (
        np.column_stack([_sib_i, _sib_j]).astype(np.int32)
        if len(_sib_i) > 0
        else np.empty((0, 2), dtype=np.int32)
    )

    # QPD constant
    quanta_per_day = T // (T // 7) if T == 42 else 7  # 42 / 6 days = 7

    # ---- Paired cohort practical event pairs ----
    raw_pairs = pkl_data.get("paired_practical_events", [])
    if raw_pairs:
        cohort_event_pairs = np.array(raw_pairs, dtype=np.int32)  # (P, 2)
    else:
        cohort_event_pairs = np.empty((0, 2), dtype=np.int32)

    # ---- Cohort subgroup pairs (group-index level) ----
    # Prefer explicit cohort_pairs (string IDs) from pkl; fall back to
    # deriving from legacy paired_practical_events.
    raw_cohort_pairs = pkl_data.get("cohort_pairs", [])
    if raw_cohort_pairs:
        _sg_pairs_list = []
        _sg_seen: set[tuple[int, int]] = set()
        for left_id, right_id in raw_cohort_pairs:
            li = group_to_idx.get(left_id)
            ri = group_to_idx.get(right_id)
            if li is not None and ri is not None:
                key = (min(li, ri), max(li, ri))
                if key not in _sg_seen:
                    _sg_seen.add(key)
                    _sg_pairs_list.append((li, ri))
        cohort_subgroup_pairs = (
            np.array(_sg_pairs_list, dtype=np.int32)
            if _sg_pairs_list
            else np.empty((0, 2), dtype=np.int32)
        )
    elif raw_pairs:
        # Derive from legacy event pairs: extract unique group pairs
        _sg_pairs_list = []
        _sg_seen_set: set[tuple[int, int]] = set()
        for ea, eb in raw_pairs:
            for ga in event_group_lists[ea]:
                for gb in event_group_lists[eb]:
                    if ga != gb:
                        key = (min(ga, gb), max(ga, gb))
                        if key not in _sg_seen_set:
                            _sg_seen_set.add(key)
                            _sg_pairs_list.append((ga, gb))
        cohort_subgroup_pairs = (
            np.array(_sg_pairs_list, dtype=np.int32)
            if _sg_pairs_list
            else np.empty((0, 2), dtype=np.int32)
        )
    else:
        cohort_subgroup_pairs = np.empty((0, 2), dtype=np.int32)

    # ---- Per-event is_practical boolean ----
    _is_practical = is_prac.astype(np.bool_)  # (E,) bool

    # ================================================================
    # Expansion arrays — fully vectorized via np.repeat
    # ================================================================

    Q = int(durations.sum())

    # exp_event: each event e repeated durations[e] times  (Q,)
    exp_event = np.repeat(np.arange(E, dtype=np.int32), durations)

    # exp_offset: 0..d-1 within each event  (Q,)
    _cum = np.empty(E + 1, dtype=np.int64)
    _cum[0] = 0
    np.cumsum(durations, out=_cum[1:])
    _global_starts = np.repeat(_cum[:E], durations)  # (Q,) event start pos
    exp_offset = np.arange(Q, dtype=np.int32) - _global_starts.astype(np.int32)

    # ---- Group expansion (GQ entries) — double np.repeat ----
    n_groups_per_event = np.array([len(g) for g in event_group_lists], dtype=np.int32)
    GQ_events = int(n_groups_per_event.sum())  # event-group pairs
    # Expand events by their group count
    _eg_event = np.repeat(
        np.arange(E, dtype=np.int32), n_groups_per_event
    )  # (GQ_events,)
    _eg_group = (
        np.concatenate([np.array(g, dtype=np.int32) for g in event_group_lists])
        if GQ_events > 0
        else np.empty(0, dtype=np.int32)
    )  # (GQ_events,)
    _eg_dur = durations[_eg_event]  # (GQ_events,)

    GQ = int(_eg_dur.sum())
    grp_exp_event = np.repeat(_eg_event, _eg_dur)  # (GQ,)
    grp_exp_group = np.repeat(_eg_group, _eg_dur)  # (GQ,)

    # grp_exp_offset: 0..d-1 within each (event, group) block
    _gcum = np.empty(GQ_events + 1, dtype=np.int64)
    _gcum[0] = 0
    np.cumsum(_eg_dur, out=_gcum[1:])
    _gstarts = np.repeat(_gcum[:GQ_events], _eg_dur)
    grp_exp_offset = np.arange(GQ, dtype=np.int32) - _gstarts.astype(np.int32)

    # ================================================================
    # Membership boolean arrays — vectorized scatter
    # ================================================================

    inst_allowed = np.zeros((E, n_inst), dtype=np.bool_)
    _ia_lens = [len(a) for a in ai]
    if sum(_ia_lens) > 0:
        _ia_rows = np.repeat(np.arange(E, dtype=np.int32), _ia_lens)
        _ia_cols = np.concatenate([np.array(a, dtype=np.int64) for a in ai if a])
        _ia_valid = _ia_cols < n_inst
        inst_allowed[_ia_rows[_ia_valid], _ia_cols[_ia_valid]] = True

    room_allowed = np.zeros((E, n_rooms), dtype=np.bool_)
    _ra_lens = [len(a) for a in ar]
    if sum(_ra_lens) > 0:
        _ra_rows = np.repeat(np.arange(E, dtype=np.int32), _ra_lens)
        _ra_cols = np.concatenate([np.array(a, dtype=np.int64) for a in ar if a])
        _ra_valid = _ra_cols < n_rooms
        room_allowed[_ra_rows[_ra_valid], _ra_cols[_ra_valid]] = True

    # ================================================================
    # Availability boolean arrays
    # ================================================================

    inst_avail_bool = np.ones((n_inst, T), dtype=np.bool_)
    for idx_s, slots in pkl_data.get("instructor_available_quanta", {}).items():
        idx = int(idx_s)
        if idx < n_inst and slots is not None:
            inst_avail_bool[idx, :] = False
            _sq = np.array([q for q in slots if 0 <= q < T], dtype=np.int32)
            if len(_sq):
                inst_avail_bool[idx, _sq] = True

    room_avail_bool = np.ones((n_rooms, T), dtype=np.bool_)
    for idx_s, slots in pkl_data.get("room_available_quanta", {}).items():
        idx = int(idx_s)
        if idx < n_rooms and slots is not None:
            room_avail_bool[idx, :] = False
            _sq = np.array([q for q in slots if 0 <= q < T], dtype=np.int32)
            if len(_sq):
                room_avail_bool[idx, _sq] = True

    # ================================================================
    # Padded domain matrices (for repair & sampling)
    # ================================================================

    max_inst_dom = max((len(d) for d in ai), default=1) or 1
    max_room_dom = max((len(d) for d in ar), default=1) or 1
    max_time_dom = max((len(d) for d in at), default=1) or 1

    inst_domains = np.zeros((E, max_inst_dom), dtype=np.int64)
    inst_dom_len = np.zeros(E, dtype=np.int64)
    room_domains = np.zeros((E, max_room_dom), dtype=np.int64)
    room_dom_len = np.zeros(E, dtype=np.int64)
    time_domains = np.zeros((E, max_time_dom), dtype=np.int64)
    time_dom_len = np.zeros(E, dtype=np.int64)

    for e in range(E):
        di = ai[e]
        if di:
            inst_dom_len[e] = len(di)
            inst_domains[e, : len(di)] = di
        dr = ar[e]
        if dr:
            room_dom_len[e] = len(dr)
            room_domains[e, : len(dr)] = dr
        dt = at[e]
        if dt:
            time_dom_len[e] = len(dt)
            time_domains[e, : len(dt)] = dt

    return VectorizedLookups(
        n_events=E,
        n_rooms=n_rooms,
        n_instructors=n_inst,
        n_groups=n_groups,
        event_metadata=event_metadata,
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
        inst_domains=inst_domains,
        inst_dom_len=inst_dom_len,
        room_domains=room_domains,
        room_dom_len=room_dom_len,
        time_domains=time_domains,
        time_dom_len=time_dom_len,
        group_membership=group_membership,
        group_conflict_matrix=group_conflict_matrix,
        event_group_lists=event_group_lists,
        group_to_idx=group_to_idx,
        sibling_matrix=sibling_matrix,
        sibling_pairs=sibling_pairs,
        sibling_event_to_course=sibling_event_to_course,
        quanta_per_day=quanta_per_day,
        cohort_event_pairs=cohort_event_pairs,
        cohort_subgroup_pairs=cohort_subgroup_pairs,
        is_practical=_is_practical,
    )
