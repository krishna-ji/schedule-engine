"""Vectorized repair operator with numpy-accelerated conflict resolution.

Replaces the O(E^2) per-individual Python dict-based repair with:
  1. Population-level domain fix (boolean membership arrays)
  2. Per-individual conflict resolution using 2D numpy occupancy arrays
  3. Group-aware deconfliction with vectorized placement search

Key speedup: numpy ``int32[resource, timeslot]`` arrays replace
``dict[tuple, set]`` occupancy maps.  Candidate evaluation is
vectorized across all valid start-times x rooms in one shot.

For pymoo integration, ``PymooVectorizedRepair`` is a drop-in replacement
for ``PymooSchedulingRepair``.
"""

from __future__ import annotations

import pickle

import numpy as np

from .bitset_time import T


class VectorizedRepair:
    """Repair engine using numpy occupancy arrays for fast conflict resolution."""

    def __init__(self, events_data_path: str = "events_with_domains.pkl"):
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
        self._group_events: list[list[int]] = [[] for _ in range(self.n_groups)]
        self._group_util = np.zeros(self.n_groups, dtype=np.int32)
        for e in range(E):
            for gidx in self._event_groups[e]:
                self._group_events[gidx].append(e)
                self._group_util[gidx] += int(self.durations[e])
        self._sorted_groups = list(np.argsort(-self._group_util))

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

        # ---- Pre-computed quanta matrices per event (for _find_best) ----
        # Also pre-split columns to avoid .sum(axis=1) overhead in tight loop.
        self._event_times: list[np.ndarray] = []
        self._event_quanta: list[np.ndarray] = []
        self._event_qcols: list[list[np.ndarray]] = []  # quanta[:,c] columns
        for e in range(E):
            n_t = int(self.time_dom_len[e])
            if n_t == 0:
                self._event_times.append(np.empty(0, dtype=np.int64))
                self._event_quanta.append(np.empty((0, 0), dtype=np.int64))
                self._event_qcols.append([])
                continue
            times = self.time_domains[e, :n_t]
            dur = int(self.durations[e])
            q_mat = times[:, None] + np.arange(dur, dtype=np.int64)[None, :]
            valid = q_mat[:, -1] < T
            t_valid = times[valid].copy()
            q_valid = q_mat[valid].copy()
            self._event_times.append(t_valid)
            self._event_quanta.append(q_valid)
            self._event_qcols.append([q_valid[:, c].copy() for c in range(dur)])

        # Pre-computed room domain arrays per event
        self._event_rooms: list[np.ndarray] = []
        for e in range(E):
            n_r = int(self.room_dom_len[e])
            self._event_rooms.append(self.room_domains[e, :n_r].copy().astype(np.int64))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def repair_batch(self, X: np.ndarray, passes: int = 3) -> np.ndarray:
        """Repair population X of shape (N, 3*E).

        Stage 1: Fix domain violations (vectorized across population).
        Stage 2: Incremental conflict fixing (per individual, numpy arrays).
        Stage 3: Group-aware deconfliction (per individual).
        """
        X = X.copy().astype(np.int64)
        self._fix_domains_vec(X)
        for n in range(X.shape[0]):
            self._repair_individual(X[n], passes)
        return X

    # ------------------------------------------------------------------
    # Stage 1: domain fix (vectorized across population)
    # ------------------------------------------------------------------

    def _fix_domains_vec(self, X: np.ndarray) -> None:
        """Fix domain violations in-place.  Vectorized over population."""
        E = self.n_events
        inst = X[:, 0::3]
        room = X[:, 1::3]
        time = X[:, 2::3]
        e_idx = np.arange(E, dtype=np.int64)

        # Instructor domain
        inst_clamped = np.clip(inst, 0, self.n_instructors - 1)
        inst_ok = self.inst_allowed[e_idx[np.newaxis, :], inst_clamped]
        inst_bad = ~inst_ok
        if inst_bad.any():
            bi, be = np.nonzero(inst_bad)
            X[bi, 3 * be] = self.inst_domains[be, 0]

        # Room domain
        room_clamped = np.clip(room, 0, self.n_rooms - 1)
        room_ok = self.room_allowed[e_idx[np.newaxis, :], room_clamped]
        room_bad = ~room_ok
        if room_bad.any():
            bi, be = np.nonzero(room_bad)
            X[bi, 3 * be + 1] = self.room_domains[be, 0]

        # Time domain (padded broadcasting check)
        time_vals = time[:, :, np.newaxis]  # (N, E, 1)
        time_doms = self.time_domains[np.newaxis, :, :]  # (1, E, max_dom)
        dom_mask = (
            np.arange(self._time_max_dom)[np.newaxis, :]
            < self.time_dom_len[:, np.newaxis]
        )
        matches = (time_vals == time_doms) & dom_mask[np.newaxis, :, :]
        time_ok = matches.any(axis=2)
        time_bad = ~time_ok & (self.time_dom_len[np.newaxis, :] > 0)
        if time_bad.any():
            bi, be = np.nonzero(time_bad)
            X[bi, 3 * be + 2] = self.time_domains[be, 0]

    # ------------------------------------------------------------------
    # Stage 2 & 3: per-individual conflict resolution
    # ------------------------------------------------------------------

    def _repair_individual(self, x: np.ndarray, passes: int) -> None:
        """Smart conflict resolution for one chromosome (in-place)."""
        inst = x[0::3]  # views into x
        room = x[1::3]
        time = x[2::3]

        room_occ, inst_occ, group_occ = self._build_occupancy(inst, room, time)

        # Stage 2: incremental conflict fixing
        for pass_idx in range(passes):
            scores = self._score_all(inst, room, time, room_occ, inst_occ, group_occ)
            conflict_events = np.nonzero(scores > 0)[0]
            if len(conflict_events) == 0:
                break
            # Alternate ordering: most-conflicted-first / least-first
            if pass_idx % 2 == 0:
                order = conflict_events[np.argsort(-scores[conflict_events])]
            else:
                order = conflict_events[np.argsort(scores[conflict_events])]
            for e_int in order:
                e = int(e_int)
                if (
                    self._event_score(
                        e, inst, room, time, room_occ, inst_occ, group_occ
                    )
                    == 0
                ):
                    continue
                self._remove_occ(e, inst, room, time, room_occ, inst_occ, group_occ)
                self._find_best(e, inst, room, time, room_occ, inst_occ, group_occ)
                self._add_occ(e, inst, room, time, room_occ, inst_occ, group_occ)

        # Stage 3: group-aware deconfliction
        self._fix_groups(inst, room, time, room_occ, inst_occ, group_occ)

    # ------------------------------------------------------------------
    # Occupancy array management
    # ------------------------------------------------------------------

    def _build_occupancy(
        self,
        inst: np.ndarray,
        room: np.ndarray,
        time: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build 2D count arrays from current assignment (vectorized)."""
        room_occ = np.zeros((self.n_rooms, T), dtype=np.int32)
        inst_occ = np.zeros((self.n_instructors, T), dtype=np.int32)
        group_occ = np.zeros((self.n_groups, T), dtype=np.int32)

        starts = time[self.exp_event].astype(np.int64)
        quanta = np.clip(starts + self.exp_offset, 0, T - 1)
        rooms_q = np.clip(room[self.exp_event].astype(np.int64), 0, self.n_rooms - 1)
        insts_q = np.clip(
            inst[self.exp_event].astype(np.int64), 0, self.n_instructors - 1
        )

        np.add.at(room_occ, (rooms_q, quanta), 1)
        np.add.at(inst_occ, (insts_q, quanta), 1)

        grp_starts = time[self.grp_exp_event].astype(np.int64)
        grp_quanta = np.clip(grp_starts + self.grp_exp_offset, 0, T - 1)
        np.add.at(group_occ, (self.grp_exp_group, grp_quanta), 1)

        return room_occ, inst_occ, group_occ

    def _remove_occ(
        self,
        e: int,
        inst: np.ndarray,
        room: np.ndarray,
        time: np.ndarray,
        room_occ: np.ndarray,
        inst_occ: np.ndarray,
        group_occ: np.ndarray,
    ) -> None:
        """Remove event *e* from occupancy arrays."""
        s = int(time[e])
        d = int(self.durations[e])
        r = int(room[e])
        i = int(inst[e])
        for q in range(s, min(s + d, T)):
            room_occ[r, q] -= 1
            inst_occ[i, q] -= 1
            for gidx in self._event_groups[e]:
                group_occ[gidx, q] -= 1

    def _add_occ(
        self,
        e: int,
        inst: np.ndarray,
        room: np.ndarray,
        time: np.ndarray,
        room_occ: np.ndarray,
        inst_occ: np.ndarray,
        group_occ: np.ndarray,
    ) -> None:
        """Add event *e* to occupancy arrays."""
        s = int(time[e])
        d = int(self.durations[e])
        r = int(room[e])
        i = int(inst[e])
        for q in range(s, min(s + d, T)):
            room_occ[r, q] += 1
            inst_occ[i, q] += 1
            for gidx in self._event_groups[e]:
                group_occ[gidx, q] += 1

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_all(
        self,
        inst: np.ndarray,
        room: np.ndarray,
        time: np.ndarray,
        room_occ: np.ndarray,
        inst_occ: np.ndarray,
        group_occ: np.ndarray,
    ) -> np.ndarray:
        """Conflict score for every event (vectorized).  Returns (E,) int32."""
        E = self.n_events
        starts = time[self.exp_event].astype(np.int64)
        quanta = np.clip(starts + self.exp_offset, 0, T - 1)
        rooms_q = np.clip(room[self.exp_event].astype(np.int64), 0, self.n_rooms - 1)
        insts_q = np.clip(
            inst[self.exp_event].astype(np.int64), 0, self.n_instructors - 1
        )

        scores = np.zeros(E, dtype=np.int32)

        # Room double-booking
        rc = (room_occ[rooms_q, quanta] > 1).astype(np.int32)
        np.add.at(scores, self.exp_event, rc)

        # Instructor double-booking
        ic = (inst_occ[insts_q, quanta] > 1).astype(np.int32)
        np.add.at(scores, self.exp_event, ic)

        # Group double-booking
        grp_starts = time[self.grp_exp_event].astype(np.int64)
        grp_quanta = np.clip(grp_starts + self.grp_exp_offset, 0, T - 1)
        gc = (group_occ[self.grp_exp_group, grp_quanta] > 1).astype(np.int32)
        np.add.at(scores, self.grp_exp_event, gc)

        # Instructor availability (heavy penalty)
        iu = (~self.inst_avail[insts_q, quanta]).astype(np.int32) * 100
        np.add.at(scores, self.exp_event, iu)

        return scores

    def _event_score(
        self,
        e: int,
        inst: np.ndarray,
        room: np.ndarray,
        time: np.ndarray,
        room_occ: np.ndarray,
        inst_occ: np.ndarray,
        group_occ: np.ndarray,
    ) -> int:
        """Conflict score for event *e* (event IS in occupancy -- check >1)."""
        s = int(time[e])
        d = int(self.durations[e])
        r = int(room[e])
        i = int(inst[e])
        score = 0
        for q in range(s, min(s + d, T)):
            if room_occ[r, q] > 1:
                score += 1
            if inst_occ[i, q] > 1:
                score += 1
            for gidx in self._event_groups[e]:
                if group_occ[gidx, q] > 1:
                    score += 1
            if not self.inst_avail[i, q]:
                score += 100
        return score

    def _check_placement(
        self,
        e: int,
        i_val: int,
        r_val: int,
        t_val: int,
        room_occ: np.ndarray,
        inst_occ: np.ndarray,
        group_occ: np.ndarray,
    ) -> int:
        """Score hypothetical placement (event NOT in occupancy -- check >0)."""
        d = int(self.durations[e])
        score = 0
        for q in range(t_val, min(t_val + d, T)):
            if room_occ[r_val, q] > 0:
                score += 1
            if inst_occ[i_val, q] > 0:
                score += 1
            for gidx in self._event_groups[e]:
                if group_occ[gidx, q] > 0:
                    score += 1
            if not self.inst_avail[i_val, q]:
                score += 100
            if not self.room_avail[r_val, q]:
                score += 100
        return score

    # ------------------------------------------------------------------
    # Placement search (vectorized across candidates)
    # ------------------------------------------------------------------

    def _find_best(
        self,
        e: int,
        inst: np.ndarray,
        room: np.ndarray,
        time: np.ndarray,
        room_occ: np.ndarray,
        inst_occ: np.ndarray,
        group_occ: np.ndarray,
    ) -> None:
        """Find and apply best (inst, room, time) for event *e*.

        Event MUST already be removed from occupancy arrays.
        Updates ``inst[e]``, ``room[e]``, ``time[e]`` in-place.

        Strategy (vectorized candidate evaluation):
          Phase 1 -- all times with current instructor + room
          Phase 2 -- all rooms at group-free times (current instructor)
          Phase 3 -- up to 8 alternate instructors x group-free times x rooms
        """
        cur_i = int(inst[e])
        cur_r = int(room[e])
        cur_t = int(time[e])

        best_score = self._check_placement(
            e, cur_i, cur_r, cur_t, room_occ, inst_occ, group_occ
        )
        if best_score == 0:
            return
        best_i, best_r, best_t = cur_i, cur_r, cur_t

        # Pre-computed candidate times and quanta
        times = self._event_times[e]
        quanta = self._event_quanta[e]  # (K, d)
        K = len(times)
        if K == 0:
            return

        # ---- Phase 1: all times, current instructor + room ----
        inst_avail_ok = self.inst_avail[cur_i, quanta].all(axis=1)  # (K,)
        iavl_penalty = (~inst_avail_ok).astype(np.int32) * 100 * int(self.durations[e])

        group_score = np.zeros(K, dtype=np.int32)
        for gidx in self._event_groups[e]:
            group_score += (group_occ[gidx][quanta] > 0).sum(axis=1).astype(np.int32)

        room_score_cur = (room_occ[cur_r][quanta] > 0).sum(axis=1).astype(np.int32)
        inst_score_cur = (inst_occ[cur_i][quanta] > 0).sum(axis=1).astype(np.int32)
        ravl_penalty = (~self.room_avail[cur_r][quanta]).sum(axis=1).astype(
            np.int32
        ) * 100

        total = (
            iavl_penalty + group_score + room_score_cur + inst_score_cur + ravl_penalty
        )

        best_k = int(np.argmin(total))
        if total[best_k] == 0:
            time[e] = times[best_k]
            return
        if total[best_k] < best_score:
            best_score = int(total[best_k])
            best_t = int(times[best_k])

        # ---- Phase 2: all rooms at group-free + instructor-available times ----
        group_free = (group_score == 0) & inst_avail_ok
        rooms_e = self._event_rooms[e]
        n_rooms_e = len(rooms_e)

        if group_free.any() and n_rooms_e > 0:
            gf_idx = np.nonzero(group_free)[0]
            gf_quanta = quanta[gf_idx]  # (F, d)

            r_occ = room_occ[rooms_e[:, None, None], gf_quanta[None, :, :]]
            r_scores = (r_occ > 0).sum(axis=2).astype(np.int32)
            r_avl = self.room_avail[rooms_e[:, None, None], gf_quanta[None, :, :]]
            r_avl_penalty = (~r_avl).sum(axis=2).astype(np.int32) * 100
            i_scores_gf = inst_score_cur[gf_idx]
            total_rf = r_scores + r_avl_penalty + i_scores_gf[None, :]

            best_flat = int(np.argmin(total_rf))
            best_ri, best_fi = divmod(best_flat, len(gf_idx))
            val_rf = int(total_rf[best_ri, best_fi])
            if val_rf < best_score:
                best_score = val_rf
                best_r = int(rooms_e[best_ri])
                best_t = int(times[gf_idx[best_fi]])
                if best_score == 0:
                    room[e] = best_r
                    time[e] = best_t
                    return

        # ---- Phase 3: alternate instructors ----
        n_inst_e = int(self.inst_dom_len[e])
        if n_inst_e > 1 and best_score > 0:
            insts_e = self.inst_domains[e, : min(n_inst_e, 8)]
            for alt_i_np in insts_e:
                alt_i = int(alt_i_np)
                if alt_i == cur_i:
                    continue

                avail = self.inst_avail[alt_i, quanta].all(axis=1)
                if not avail.any():
                    continue

                av_idx = np.nonzero(avail)[0]
                av_quanta = quanta[av_idx]

                g_score = np.zeros(len(av_idx), dtype=np.int32)
                for gidx in self._event_groups[e]:
                    g_score += (
                        (group_occ[gidx][av_quanta] > 0).sum(axis=1).astype(np.int32)
                    )

                gf_alt = g_score == 0
                if not gf_alt.any():
                    continue

                gf_av_idx = np.nonzero(gf_alt)[0]
                gf_av_quanta = av_quanta[gf_av_idx]

                i_score_alt = (
                    (inst_occ[alt_i][gf_av_quanta] > 0).sum(axis=1).astype(np.int32)
                )

                if n_rooms_e > 0:
                    r_occ_a = room_occ[rooms_e[:, None, None], gf_av_quanta[None, :, :]]
                    r_scores_a = (r_occ_a > 0).sum(axis=2).astype(np.int32)
                    r_avl_a = self.room_avail[
                        rooms_e[:, None, None], gf_av_quanta[None, :, :]
                    ]
                    r_avl_pen_a = (~r_avl_a).sum(axis=2).astype(np.int32) * 100
                    total_a = r_scores_a + r_avl_pen_a + i_score_alt[None, :]
                    best_a_flat = int(np.argmin(total_a))
                    best_a_ri, best_a_fi = divmod(best_a_flat, len(gf_av_idx))
                    val_a = int(total_a[best_a_ri, best_a_fi])
                    if val_a < best_score:
                        best_score = val_a
                        best_i = alt_i
                        best_r = int(rooms_e[best_a_ri])
                        best_t = int(times[av_idx[gf_av_idx[best_a_fi]]])
                        if best_score == 0:
                            inst[e] = best_i
                            room[e] = best_r
                            time[e] = best_t
                            return

        # Apply best found (even if score > 0)
        inst[e] = best_i
        room[e] = best_r
        time[e] = best_t

    # ------------------------------------------------------------------
    # Stage 3: group-aware deconfliction
    # ------------------------------------------------------------------

    def _fix_groups(
        self,
        inst: np.ndarray,
        room: np.ndarray,
        time: np.ndarray,
        room_occ: np.ndarray,
        inst_occ: np.ndarray,
        group_occ: np.ndarray,
    ) -> None:
        """Process groups atomically: remove all events, re-insert greedily.

        Tightest groups (highest utilization) are processed first.
        """
        for _round in range(2):
            any_fix = False
            for gidx in self._sorted_groups:
                events = self._group_events[gidx]
                if not events:
                    continue

                # Check if group has any double-booking
                has_conflict = False
                for ev in events:
                    s = int(time[ev])
                    d = int(self.durations[ev])
                    for q in range(s, min(s + d, T)):
                        if group_occ[gidx, q] > 1:
                            has_conflict = True
                            break
                    if has_conflict:
                        break
                if not has_conflict:
                    continue

                any_fix = True

                # Remove all events in this group
                for ev in events:
                    self._remove_occ(
                        ev, inst, room, time, room_occ, inst_occ, group_occ
                    )

                # Re-insert sorted by duration (longest first)
                events_sorted = sorted(events, key=lambda ev: -int(self.durations[ev]))
                for ev in events_sorted:
                    self._find_best(ev, inst, room, time, room_occ, inst_occ, group_occ)
                    self._add_occ(ev, inst, room, time, room_occ, inst_occ, group_occ)

            if not any_fix:
                break

        # Final residual cleanup
        for _pass in range(2):
            scores = self._score_all(inst, room, time, room_occ, inst_occ, group_occ)
            conflict_events = np.nonzero(scores > 0)[0]
            if len(conflict_events) == 0:
                break
            conflict_events = conflict_events[np.argsort(-scores[conflict_events])]
            any_fix = False
            for e_int in conflict_events:
                ev = int(e_int)
                if (
                    self._event_score(
                        ev, inst, room, time, room_occ, inst_occ, group_occ
                    )
                    == 0
                ):
                    continue
                any_fix = True
                self._remove_occ(ev, inst, room, time, room_occ, inst_occ, group_occ)
                self._find_best(ev, inst, room, time, room_occ, inst_occ, group_occ)
                self._add_occ(ev, inst, room, time, room_occ, inst_occ, group_occ)
            if not any_fix:
                break


# ======================================================================
# Pymoo Repair wrapper
# ======================================================================

try:
    from pymoo.core.repair import Repair

    class PymooVectorizedRepair(Repair):
        """Pymoo-compatible vectorized repair -- drop-in for PymooSchedulingRepair."""

        def __init__(
            self,
            events_data_path: str = "events_with_domains.pkl",
            passes: int = 3,
        ):
            super().__init__()
            self.engine = VectorizedRepair(events_data_path)
            self.passes = passes

        def _do(self, problem, x, **kwargs):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return self.engine.repair_batch(x, passes=self.passes)

except ImportError:
    pass  # pymoo not installed
