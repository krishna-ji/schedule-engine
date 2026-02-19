"""Vectorized repair analysis — population-level conflict detection and domain fixing.

Replaces per-individual Python loops in repair_operator_bitset.py for the
ANALYSIS substeps (domain clamping, count construction, conflict detection).
The actual placement search remains sequential per-individual.

API
---
    fix_domains_batch(X, allowed_I, allowed_R, allowed_T) -> X_fixed
    build_counts_batch(X, durations, event_groups, n_rooms, n_inst, n_groups, T) -> (rc, ic, gc)
    count_conflicts_batch(X, durations, event_groups, n_rooms, n_inst, n_groups, T) -> C
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------
# Stage 1 — Vectorized domain fixing
# ------------------------------------------------------------------


def fix_domains_batch(
    X: np.ndarray,
    allowed_instructors: list[list[int]],
    allowed_rooms: list[list[int]],
    allowed_starts: list[list[int]],
    *,
    inst_avail: dict | None = None,
    durations: np.ndarray | None = None,
) -> np.ndarray:
    """Clamp all assignments to valid domains across entire population.

    Parameters
    ----------
    X : ndarray, shape (N, 3*E)
        Population matrix (modified in-place).
    allowed_instructors, allowed_rooms, allowed_starts :
        Per-event allowed values lists.
    inst_avail : optional dict[int, set|None]
        Instructor available quanta (from pkl_data).
    durations : optional ndarray (E,) int32
        Event durations (needed for instructor-time availability check).

    Returns
    -------
    X : same array, modified in-place.
    """
    E = len(allowed_instructors)

    inst = X[:, 0::3]  # (N, E)
    room = X[:, 1::3]  # (N, E)
    time = X[:, 2::3]  # (N, E)

    # Build per-event allowed sets for fast membership testing
    for e in range(E):
        ai = allowed_instructors[e]
        ar = allowed_rooms[e]
        at = allowed_starts[e]

        if ai:
            ai_set = set(ai)
            # Vectorized: check which individuals have invalid instructors
            invalid_mask = np.array([int(v) not in ai_set for v in inst[:, e]])
            if invalid_mask.any():
                inst[invalid_mask, e] = ai[0]

        if ar:
            ar_set = set(ar)
            invalid_mask = np.array([int(v) not in ar_set for v in room[:, e]])
            if invalid_mask.any():
                room[invalid_mask, e] = ar[0]

        if at:
            at_set = set(at)
            invalid_mask = np.array([int(v) not in at_set for v in time[:, e]])
            if invalid_mask.any():
                time[invalid_mask, e] = at[0]

    # Instructor-availability time clamping (matches _valid_starts_for logic)
    if inst_avail is not None and durations is not None:
        for e in range(E):
            at = allowed_starts[e]
            if not at:
                continue
            dur = int(durations[e])
            # Group individuals by their assigned instructor
            unique_insts = np.unique(inst[:, e])
            for i_idx in unique_insts:
                i_idx = int(i_idx)
                slots = inst_avail.get(i_idx)
                if slots is None:
                    continue
                # Compute valid starts for this (event, instructor)
                valid = [s for s in at if all(q in slots for q in range(s, s + dur))]
                if not valid:
                    continue
                valid_set = set(valid)
                # Find individuals with this instructor and invalid time
                mask = inst[:, e] == i_idx
                for n in np.where(mask)[0]:
                    if int(time[n, e]) not in valid_set:
                        time[n, e] = valid[0]

    return X


# ------------------------------------------------------------------
# Count array construction
# ------------------------------------------------------------------


def build_counts_batch(
    X: np.ndarray,
    durations: np.ndarray,
    event_groups: list[list[int]],
    n_rooms: int,
    n_inst: int,
    n_groups: int,
    T: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3D occupancy count arrays for entire population.

    Parameters
    ----------
    X : ndarray, shape (N, 3*E)
    durations : ndarray, shape (E,) int32
    event_groups : list of lists of group indices per event
    n_rooms, n_inst, n_groups : dimensions
    T : total quanta (default 42)

    Returns
    -------
    rc : ndarray (N, n_rooms, T) int16 — room counts
    ic : ndarray (N, n_inst, T) int16 — instructor counts
    gc : ndarray (N, n_groups, T) int16 — group counts
    """
    N = X.shape[0]
    E = len(durations)

    inst = X[:, 0::3]  # (N, E)
    room = X[:, 1::3]  # (N, E)
    time = X[:, 2::3]  # (N, E)

    rc = np.zeros((N, n_rooms, T), dtype=np.int16)
    ic = np.zeros((N, n_inst, T), dtype=np.int16)
    gc = np.zeros((N, n_groups, T), dtype=np.int16)

    # Build expansion arrays (event → quanta) — same pattern as hard evaluator
    Q = int(durations.sum())
    exp_event = np.empty(Q, dtype=np.int32)
    exp_offset = np.empty(Q, dtype=np.int32)
    pos = 0
    for e in range(E):
        d = int(durations[e])
        exp_event[pos : pos + d] = e
        exp_offset[pos : pos + d] = np.arange(d, dtype=np.int32)
        pos += d

    # Expand to absolute quanta: (N, Q)
    starts = time[:, exp_event]  # (N, Q)
    quanta = starts + exp_offset[np.newaxis, :]  # (N, Q)
    rooms = room[:, exp_event]  # (N, Q)
    insts = inst[:, exp_event]  # (N, Q)

    # Flat indices for scatter-add: rc[n, room, quantum]
    n_idx = np.repeat(np.arange(N, dtype=np.int64), Q)
    q_flat = quanta.ravel().astype(np.int64)
    r_flat = rooms.ravel().astype(np.int64)
    i_flat = insts.ravel().astype(np.int64)

    # Clamp to valid range
    q_flat = np.clip(q_flat, 0, T - 1)

    # Room counts: rc[n, room, quantum] += 1
    rc_flat_idx = n_idx * (n_rooms * T) + r_flat * T + q_flat
    np.add.at(rc.ravel(), rc_flat_idx, 1)

    # Instructor counts: ic[n, inst, quantum] += 1
    ic_flat_idx = n_idx * (n_inst * T) + i_flat * T + q_flat
    np.add.at(ic.ravel(), ic_flat_idx, 1)

    # Group counts: expand each event's quanta for each group
    # Build group expansion
    GQ = sum(int(durations[e]) * len(event_groups[e]) for e in range(E))
    grp_exp_event = np.empty(GQ, dtype=np.int32)
    grp_exp_offset = np.empty(GQ, dtype=np.int32)
    grp_exp_group = np.empty(GQ, dtype=np.int32)
    pos = 0
    for e in range(E):
        d = int(durations[e])
        for gidx in event_groups[e]:
            grp_exp_event[pos : pos + d] = e
            grp_exp_offset[pos : pos + d] = np.arange(d, dtype=np.int32)
            grp_exp_group[pos : pos + d] = gidx
            pos += d

    grp_starts = time[:, grp_exp_event]  # (N, GQ)
    grp_quanta = grp_starts + grp_exp_offset[np.newaxis, :]  # (N, GQ)
    grp_quanta_flat = np.clip(grp_quanta.ravel().astype(np.int64), 0, T - 1)

    n_idx_g = np.repeat(np.arange(N, dtype=np.int64), GQ)
    g_flat = np.tile(grp_exp_group, N).astype(np.int64)

    gc_flat_idx = n_idx_g * (n_groups * T) + g_flat * T + grp_quanta_flat
    np.add.at(gc.ravel(), gc_flat_idx, 1)

    return rc, ic, gc


# ------------------------------------------------------------------
# Conflict detection
# ------------------------------------------------------------------


def count_conflicts_batch(
    X: np.ndarray,
    durations: np.ndarray,
    event_groups: list[list[int]],
    n_rooms: int,
    n_inst: int,
    n_groups: int,
    T: int = 42,
) -> np.ndarray:
    """Count conflicts per event per individual across entire population.

    A conflict for event e is any quantum where the assigned resource
    has count > 1 (i.e., double-booked).

    Parameters
    ----------
    X : ndarray, shape (N, 3*E)
    durations, event_groups : per-event metadata
    n_rooms, n_inst, n_groups, T : dimensions

    Returns
    -------
    C : ndarray (N, E) int32 — conflict count per event per individual
    """
    N = X.shape[0]
    E = len(durations)

    # Build counts
    rc, ic, gc = build_counts_batch(
        X, durations, event_groups, n_rooms, n_inst, n_groups, T
    )

    inst = X[:, 0::3]  # (N, E)
    room = X[:, 1::3]  # (N, E)
    time = X[:, 2::3]  # (N, E)

    C = np.zeros((N, E), dtype=np.int32)

    # For each event, count quanta where its assigned resources have count > 1
    for e in range(E):
        d = int(durations[e])
        gidxs = event_groups[e]
        starts = time[:, e].astype(np.int64)  # (N,)
        rooms_e = room[:, e].astype(np.int64)  # (N,)
        insts_e = inst[:, e].astype(np.int64)  # (N,)

        for q_off in range(d):
            q = np.clip(starts + q_off, 0, T - 1)  # (N,)
            n_idx = np.arange(N, dtype=np.int64)

            # Room conflict: rc[n, rooms_e[n], q[n]] > 1
            rc_vals = rc[n_idx, rooms_e, q]
            C[:, e] += (rc_vals > 1).astype(np.int32)

            # Instructor conflict
            ic_vals = ic[n_idx, insts_e, q]
            C[:, e] += (ic_vals > 1).astype(np.int32)

            # Group conflicts
            for gidx in gidxs:
                gc_vals = gc[n_idx, gidx, q]
                C[:, e] += (gc_vals > 1).astype(np.int32)

    return C


# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------


def repair_summary_batch(
    X: np.ndarray,
    durations: np.ndarray,
    event_groups: list[list[int]],
    n_rooms: int,
    n_inst: int,
    n_groups: int,
    T: int = 42,
) -> dict:
    """Compute population-level repair analysis summary.

    Returns
    -------
    dict with:
        conflicts_per_individual : ndarray (N,) — total conflicts per individual
        events_with_conflicts : ndarray (N,) — number of events with conflicts
        max_event_conflicts : ndarray (N,) — worst event per individual
        population_feasible_frac : float — fraction with 0 conflicts
    """
    C = count_conflicts_batch(X, durations, event_groups, n_rooms, n_inst, n_groups, T)
    return {
        "conflicts_per_individual": C.sum(axis=1),
        "events_with_conflicts": (C > 0).sum(axis=1),
        "max_event_conflicts": C.max(axis=1),
        "population_feasible_frac": float((C.sum(axis=1) == 0).mean()),
        "total_population_conflicts": int(C.sum()),
    }
