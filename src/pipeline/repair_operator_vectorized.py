"""Vectorized repair operator with numpy-accelerated conflict resolution.

Replaces the O(E^2) per-individual Python dict-based repair with:
  1. Population-level domain fix (boolean membership arrays)
  2. Population-level stochastic conflict resolution (bincount occupancy)

Stage 1 fixes out-of-domain assignments via random domain sampling.
Stage 2 detects per-event conflict scores for the entire population in
one vectorized pass (room / instructor / group double-booking +
availability violations), then stochastically resamples time and room
for conflicting events.  Zero Python loops over N or E.

For pymoo integration, ``PymooVectorizedRepair`` is a drop-in replacement
for ``PymooSchedulingRepair``.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np

from .bitset_time import T

logger = logging.getLogger(__name__)


class VectorizedRepair:
    """Repair engine using numpy occupancy arrays for fast conflict resolution."""

    def __init__(self, events_data_path: str = ".cache/events_with_domains.pkl"):
        with open(events_data_path, "rb") as f:
            data = pickle.load(f)

        self.events: list[dict] = data["events"]
        self.n_events: int = len(self.events)
        E = self.n_events

        # ---- Raw domain lists ----
        ai = data["allowed_instructors"]
        ar = data["allowed_rooms"]
        at = data["allowed_starts"]

        # ---- Padded domain matrices for vectorized domain fix ----
        self._inst_max_dom = max((len(d) for d in ai), default=1) or 1
        self._room_max_dom = max((len(d) for d in ar), default=1) or 1
        self._time_max_dom = max((len(d) for d in at), default=1) or 1

        self.inst_domains = np.zeros((E, self._inst_max_dom), dtype=np.int64)
        self.inst_dom_len = np.zeros(E, dtype=np.int64)
        self.room_domains = np.zeros((E, self._room_max_dom), dtype=np.int64)
        self.room_dom_len = np.zeros(E, dtype=np.int64)
        self.time_domains = np.zeros((E, self._time_max_dom), dtype=np.int64)
        self.time_dom_len = np.zeros(E, dtype=np.int64)

        for e in range(E):
            di = ai[e]
            if di:
                self.inst_dom_len[e] = len(di)
                self.inst_domains[e, : len(di)] = di
            dr = ar[e]
            if dr:
                self.room_dom_len[e] = len(dr)
                self.room_domains[e, : len(dr)] = dr
            dt = at[e]
            if dt:
                self.time_dom_len[e] = len(dt)
                self.time_domains[e, : len(dt)] = dt

        # ---- Domain-integrity warnings ----
        _n_empty_inst = int((self.inst_dom_len == 0).sum())
        _n_empty_room = int((self.room_dom_len == 0).sum())
        if _n_empty_inst:
            logger.warning(
                "VectorizedRepair: %d events have empty instructor domains",
                _n_empty_inst,
            )
        if _n_empty_room:
            logger.warning(
                "VectorizedRepair: %d events have empty room domains",
                _n_empty_room,
            )

        # ---- Per-event metadata ----
        self.durations = np.array(
            [ev["num_quanta"] for ev in self.events], dtype=np.int32
        )

        # Resource counts
        self.n_instructors = max((max(d) for d in ai if d), default=0) + 1
        self.n_rooms = max((max(d) for d in ar if d), default=0) + 1

        # ---- Group mapping ----
        all_gids: set[str] = set()
        for ev in self.events:
            all_gids.update(ev["group_ids"])
        group_to_idx = {gid: i for i, gid in enumerate(sorted(all_gids))}
        self.n_groups = len(group_to_idx)

        # Per-event group indices
        self._event_groups: list[list[int]] = [
            [group_to_idx[gid] for gid in ev["group_ids"]] for ev in self.events
        ]

        # Group -> events and utilization (for group deconfliction ordering)
        # (Kept for potential future use; not needed by vectorized repair.)

        # ---- Expansion arrays (vectorized occupancy building) ----
        Q = int(self.durations.sum())
        self.exp_event = np.empty(Q, dtype=np.int32)
        self.exp_offset = np.empty(Q, dtype=np.int32)
        pos = 0
        for e in range(E):
            d = int(self.durations[e])
            self.exp_event[pos : pos + d] = e
            self.exp_offset[pos : pos + d] = np.arange(d, dtype=np.int32)
            pos += d

        GQ = sum(int(self.durations[e]) * len(self._event_groups[e]) for e in range(E))
        self.grp_exp_event = np.empty(GQ, dtype=np.int32)
        self.grp_exp_offset = np.empty(GQ, dtype=np.int32)
        self.grp_exp_group = np.empty(GQ, dtype=np.int32)
        pos = 0
        for e in range(E):
            d = int(self.durations[e])
            for gidx in self._event_groups[e]:
                self.grp_exp_event[pos : pos + d] = e
                self.grp_exp_offset[pos : pos + d] = np.arange(d, dtype=np.int32)
                self.grp_exp_group[pos : pos + d] = gidx
                pos += d

        # ---- Availability boolean arrays (resource x T) ----
        self.inst_avail = np.ones((self.n_instructors, T), dtype=np.bool_)
        for idx, slots in data.get("instructor_available_quanta", {}).items():
            idx = int(idx)
            if idx < self.n_instructors and slots is not None:
                self.inst_avail[idx, :] = False
                for q in slots:
                    if 0 <= q < T:
                        self.inst_avail[idx, q] = True

        self.room_avail = np.ones((self.n_rooms, T), dtype=np.bool_)
        for idx, slots in data.get("room_available_quanta", {}).items():
            idx = int(idx)
            if idx < self.n_rooms and slots is not None:
                self.room_avail[idx, :] = False
                for q in slots:
                    if 0 <= q < T:
                        self.room_avail[idx, q] = True

        # ---- Membership boolean arrays for domain fix ----
        self.inst_allowed = np.zeros((E, self.n_instructors), dtype=np.bool_)
        for e, a in enumerate(ai):
            for idx in a:
                if idx < self.n_instructors:
                    self.inst_allowed[e, idx] = True

        self.room_allowed = np.zeros((E, self.n_rooms), dtype=np.bool_)
        for e, a in enumerate(ar):
            for idx in a:
                if idx < self.n_rooms:
                    self.room_allowed[e, idx] = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def repair_batch(self, X: np.ndarray, passes: int = 3) -> np.ndarray:
        """Repair population X of shape (N, 3*E).

        Stage 1: Fix domain violations (vectorized across population).
        Stage 2: Stochastic conflict resolution (vectorized across population).
        """
        X = X.copy().astype(np.int64)
        self._fix_domains_vec(X)
        self._repair_conflicts_vec(X, passes)
        return X

    # ------------------------------------------------------------------
    # Stage 1: domain fix (vectorized across population)
    # ------------------------------------------------------------------

    def _fix_domains_vec(self, X: np.ndarray) -> None:
        """Fix domain violations in-place.  Vectorized over population.

        Invalid assignments are replaced with a **random** valid value
        from the event's domain (not always the first), improving
        population diversity.
        """
        E = self.n_events
        inst = X[:, 0::3]  # (N, E)
        room = X[:, 1::3]  # (N, E)
        time = X[:, 2::3]  # (N, E)
        e_idx = np.arange(E, dtype=np.int64)

        # ---- Instructor domain: random replacement ----
        inst_clamped = np.clip(inst, 0, self.n_instructors - 1)
        inst_ok = self.inst_allowed[e_idx[np.newaxis, :], inst_clamped]  # (N, E)
        inst_bad = ~inst_ok
        if inst_bad.any():
            bi, be = np.nonzero(inst_bad)  # bad (individual, event) pairs
            dom_len = self.inst_dom_len[be]  # (K,) valid domain sizes
            rand_idx = (np.random.random(len(bi)) * dom_len).astype(np.int64)
            rand_idx = np.minimum(rand_idx, np.maximum(dom_len - 1, 0))
            X[bi, 3 * be] = self.inst_domains[be, rand_idx]

        # ---- Room domain: random replacement ----
        room_clamped = np.clip(room, 0, self.n_rooms - 1)
        room_ok = self.room_allowed[e_idx[np.newaxis, :], room_clamped]  # (N, E)
        room_bad = ~room_ok
        if room_bad.any():
            bi, be = np.nonzero(room_bad)
            dom_len = self.room_dom_len[be]
            rand_idx = (np.random.random(len(bi)) * dom_len).astype(np.int64)
            rand_idx = np.minimum(rand_idx, np.maximum(dom_len - 1, 0))
            X[bi, 3 * be + 1] = self.room_domains[be, rand_idx]

        # ---- Time domain: random replacement ----
        time_vals = time[:, :, np.newaxis]                 # (N, E, 1)
        time_doms = self.time_domains[np.newaxis, :, :]    # (1, E, max_dom)
        dom_mask = (
            np.arange(self._time_max_dom)[np.newaxis, :]
            < self.time_dom_len[:, np.newaxis]
        )                                                  # (E, max_dom)
        matches = (time_vals == time_doms) & dom_mask[np.newaxis, :, :]
        time_ok = matches.any(axis=2)                      # (N, E)
        time_bad = ~time_ok & (self.time_dom_len[np.newaxis, :] > 0)
        if time_bad.any():
            bi, be = np.nonzero(time_bad)
            dom_len = self.time_dom_len[be]
            rand_idx = (np.random.random(len(bi)) * dom_len).astype(np.int64)
            rand_idx = np.minimum(rand_idx, np.maximum(dom_len - 1, 0))
            X[bi, 3 * be + 2] = self.time_domains[be, rand_idx]

    # ------------------------------------------------------------------
    # Stage 2: population-level stochastic conflict resolution
    # ------------------------------------------------------------------

    def _score_all_batch(self, X: np.ndarray) -> np.ndarray:
        """Per-event conflict scores for all individuals.

        Uses expansion arrays and ``np.bincount`` for O(Q + GQ)
        work per individual, fully vectorized across the population.

        Returns
        -------
        scores : ndarray, shape (N, E), int32
            Sum of conflicting-quantum indicators per event per individual.
            Room/instructor/group double-bookings each contribute 1 per
            quantum; availability violations contribute 10 per quantum.
        """
        N = X.shape[0]
        E = self.n_events
        Q = len(self.exp_event)
        GQ = len(self.grp_exp_event)

        inst = np.clip(X[:, 0::3], 0, self.n_instructors - 1).astype(np.int64)
        room = np.clip(X[:, 1::3], 0, self.n_rooms - 1).astype(np.int64)
        time = X[:, 2::3].astype(np.int64)

        n_idx = np.arange(N, dtype=np.int64)[:, None]  # (N, 1)

        # Expand to quantum level
        starts_exp = time[:, self.exp_event]                                  # (N, Q)
        quanta_exp = np.clip(starts_exp + self.exp_offset[None, :], 0, T - 1) # (N, Q)
        rooms_exp = room[:, self.exp_event]                                   # (N, Q)
        insts_exp = inst[:, self.exp_event]                                   # (N, Q)

        # Linearized per-individual event index (for aggregation)
        event_lin = (n_idx * E + self.exp_event[None, :]).ravel()  # (N*Q,)
        NE = N * E

        # --- Room double-booking ---
        nRT = np.int64(self.n_rooms) * np.int64(T)
        room_keys = (n_idx * nRT + rooms_exp * T + quanta_exp).ravel()
        room_cnt = np.bincount(room_keys, minlength=int(N * nRT))
        room_conflict = (room_cnt[room_keys] > 1).astype(np.float64)

        # --- Instructor double-booking ---
        nIT = np.int64(self.n_instructors) * np.int64(T)
        inst_keys = (n_idx * nIT + insts_exp * T + quanta_exp).ravel()
        inst_cnt = np.bincount(inst_keys, minlength=int(N * nIT))
        inst_conflict = (inst_cnt[inst_keys] > 1).astype(np.float64)

        # --- Availability violations (heavier weight) ---
        inst_unavail = (
            ~self.inst_avail[insts_exp.ravel(), quanta_exp.ravel()]
        ).astype(np.float64) * 10.0
        room_unavail = (
            ~self.room_avail[rooms_exp.ravel(), quanta_exp.ravel()]
        ).astype(np.float64) * 10.0

        # Aggregate per-quantum scores to per-event via bincount
        q_score = room_conflict + inst_conflict + inst_unavail + room_unavail
        scores = np.bincount(event_lin, weights=q_score, minlength=NE)

        # --- Group double-booking ---
        grp_starts = time[:, self.grp_exp_event]                                # (N, GQ)
        grp_quanta = np.clip(
            grp_starts + self.grp_exp_offset[None, :], 0, T - 1
        )                                                                       # (N, GQ)
        nGT = np.int64(self.n_groups) * np.int64(T)
        grp_keys = (
            n_idx * nGT
            + self.grp_exp_group[None, :].astype(np.int64) * T
            + grp_quanta
        ).ravel()
        grp_cnt = np.bincount(grp_keys, minlength=int(N * nGT))
        grp_conflict = (grp_cnt[grp_keys] > 1).astype(np.float64)

        grp_event_lin = (n_idx * E + self.grp_exp_event[None, :]).ravel()
        scores += np.bincount(grp_event_lin, weights=grp_conflict, minlength=NE)

        return scores[:NE].reshape(N, E).astype(np.int32)

    def _repair_conflicts_vec(self, X: np.ndarray, passes: int = 3) -> None:
        """Population-level stochastic conflict resolution (in-place).

        For each pass:
          1. Compute per-event conflict scores for all (N, E)
          2. Identify events with score > 0
          3. Resample time (always) and room (50%) from domain matrices
          4. Repeat

        This replaces the serial per-individual greedy repair with a
        fully vectorized stochastic approach.  The GA's selection
        pressure drives convergence; the repair only needs to *reduce*
        conflicts, not eliminate them in a single shot.
        """
        rng = np.random.default_rng()

        for _ in range(passes):
            scores = self._score_all_batch(X)  # (N, E)
            conflict_mask = scores > 0
            if not conflict_mask.any():
                break

            bi, be = np.nonzero(conflict_mask)

            # --- Always resample time slot ---
            t_dl = self.time_dom_len[be]
            t_valid = t_dl > 0
            t_bi, t_be, t_dl_v = bi[t_valid], be[t_valid], t_dl[t_valid]
            t_idx = (rng.random(len(t_bi)) * t_dl_v).astype(np.int64)
            t_idx = np.minimum(t_idx, t_dl_v - 1)
            X[t_bi, 3 * t_be + 2] = self.time_domains[t_be, t_idx]

            # --- Resample room for ~50 % of conflicts ---
            do_room = rng.random(len(bi)) < 0.5
            r_bi, r_be = bi[do_room], be[do_room]
            r_dl = self.room_dom_len[r_be]
            r_valid = r_dl > 0
            r_bi, r_be, r_dl_v = r_bi[r_valid], r_be[r_valid], r_dl[r_valid]
            r_idx = (rng.random(len(r_bi)) * r_dl_v).astype(np.int64)
            r_idx = np.minimum(r_idx, r_dl_v - 1)
            X[r_bi, 3 * r_be + 1] = self.room_domains[r_be, r_idx]


# ======================================================================
# Pymoo Repair wrapper
# ======================================================================

try:
    from pymoo.core.repair import Repair

    class PymooVectorizedRepair(Repair):
        """Pymoo-compatible vectorized repair -- drop-in for PymooSchedulingRepair."""

        def __init__(
            self,
            events_data_path: str = ".cache/events_with_domains.pkl",
            passes: int = 3,
        ):
            super().__init__()
            self.engine = VectorizedRepair(events_data_path)
            self.passes = passes

        def _do(self, problem, x, **kwargs):
            import logging as _logging

            if x.ndim == 1:
                x = x.reshape(1, -1)
            result = self.engine.repair_batch(x, passes=self.passes)
            _logging.getLogger(__name__).debug(
                "Repair: %d individuals, %d passes",
                x.shape[0], self.passes,
            )
            return result

except ImportError:
    pass  # pymoo not installed
