#!/usr/bin/env python3
"""Accelerated repair operator using 2-D numpy count arrays.

Drop-in replacement for SchedulingRepair that replaces dict-of-sets
occupancy maps with 2-D int16 count arrays:
    room_cnt[r, q]  — number of events on room r at quantum q
    inst_cnt[i, q]  — number of events on instructor i at quantum q
    grp_cnt [g, q]  — number of events on group g at quantum q

This avoids dict hash overhead and gives O(1) per (resource, quantum)
add / remove / conflict-check with better cache locality.

Public API (same as SchedulingRepair):
    BitsetSchedulingRepair(events_data_path)
        .construct_feasible(rng) -> ndarray
        .repair(chromosome) -> ndarray

    repair_batch(X, repairer) -> X_repaired
    PymooBitsetRepair — pymoo Repair wrapper
"""

from __future__ import annotations

import pickle
from collections import defaultdict

import numpy as np

from .bitset_time import FULL_MASK, T, mask_from_interval, mask_from_quanta

# Pre-compute mask LUT for bitset-based group-free-time checks
_MAX_DUR = 12
_MASK_LUT: np.ndarray = np.zeros((_MAX_DUR + 1, T), dtype=np.uint64)
for _d in range(_MAX_DUR + 1):
    for _s in range(T):
        _MASK_LUT[_d, _s] = mask_from_interval(_s, _d)


class BitsetSchedulingRepair:
    """Accelerated repair using numpy count arrays + bitset scoring."""

    def __init__(self, events_data_path: str = ".cache/events_with_domains.pkl"):
        with open(events_data_path, "rb") as f:
            data = pickle.load(f)

        self.events: list[dict] = data["events"]
        self.allowed_instructors: list[list[int]] = data["allowed_instructors"]
        self.allowed_rooms: list[list[int]] = data["allowed_rooms"]
        self.allowed_starts: list[list[int]] = data["allowed_starts"]
        self.n_events: int = len(self.events)

        self.inst_avail: dict[int, set | None] = data.get(
            "instructor_available_quanta", {}
        )
        self.room_avail: dict[int, set | None] = data.get("room_available_quanta", {})

        # Dimensions
        self.n_rooms = max((max(ar) for ar in self.allowed_rooms if ar), default=0) + 1
        self.n_instructors = (
            max((max(ai) for ai in self.allowed_instructors if ai), default=0) + 1
        )

        # Group index mapping (string -> int)
        all_gids: set[str] = set()
        for ev in self.events:
            all_gids.update(ev["group_ids"])
        self.group_to_idx = {gid: i for i, gid in enumerate(sorted(all_gids))}
        self.n_groups = len(self.group_to_idx)

        # Cached per-event data
        self.event_group_indices: list[list[int]] = [
            [self.group_to_idx[gid] for gid in ev["group_ids"]] for ev in self.events
        ]
        self.durations = np.array(
            [ev["num_quanta"] for ev in self.events], dtype=np.int32
        )

        # Boolean availability arrays: shape (resource, T)
        self.inst_avail_arr = np.ones((self.n_instructors, T), dtype=np.bool_)
        for idx, slots in self.inst_avail.items():
            idx = int(idx)
            if idx < self.n_instructors and slots is not None:
                self.inst_avail_arr[idx, :] = False
                for q in slots:
                    if 0 <= q < T:
                        self.inst_avail_arr[idx, q] = True

        self.room_avail_arr = np.ones((self.n_rooms, T), dtype=np.bool_)
        for idx, slots in self.room_avail.items():
            idx = int(idx)
            if idx < self.n_rooms and slots is not None:
                self.room_avail_arr[idx, :] = False
                for q in slots:
                    if 0 <= q < T:
                        self.room_avail_arr[idx, q] = True

        # Bitset availability masks (for fast group-free-time checks)
        self.inst_avail_masks = np.full(self.n_instructors, FULL_MASK, dtype=np.uint64)
        for idx, slots in self.inst_avail.items():
            idx = int(idx)
            if idx < self.n_instructors and slots is not None:
                self.inst_avail_masks[idx] = mask_from_quanta(slots)

        self.room_avail_masks = np.full(self.n_rooms, FULL_MASK, dtype=np.uint64)
        for idx, slots in self.room_avail.items():
            idx = int(idx)
            if idx < self.n_rooms and slots is not None:
                self.room_avail_masks[idx] = mask_from_quanta(slots)

        self._inst_time_cache: dict[tuple[int, int], list[int] | None] = {}

        # Per-event practical flag
        self.is_practical = np.array(
            [
                ev.get("course_type", "theory").lower() == "practical"
                for ev in self.events
            ],
            dtype=np.bool_,
        )

        # Paired cohort group mapping: group_idx -> paired_group_idx
        self.paired_with_group: dict[int, int] = {}
        for left_id, right_id in data.get("cohort_pairs", []):
            li = self.group_to_idx.get(left_id)
            ri = self.group_to_idx.get(right_id)
            if li is not None and ri is not None:
                self.paired_with_group[li] = ri
                self.paired_with_group[ri] = li

        # Paired event mapping: event_idx -> paired_event_idx (bidirectional)
        self.paired_event_map: dict[int, int] = {}
        for a, b in data.get("paired_practical_events", []):
            self.paired_event_map[a] = b
            self.paired_event_map[b] = a

        # Practical-event occupancy (rebuilt in _make_counts)
        self._prac_occ = np.zeros((self.n_groups, T), dtype=np.int16)

    # ------------------------------------------------------------------
    # Count array helpers
    # ------------------------------------------------------------------

    def _make_counts(self):
        """Allocate fresh count arrays."""
        self._prac_occ = np.zeros((self.n_groups, T), dtype=np.int16)
        return (
            np.zeros((self.n_rooms, T), dtype=np.int16),
            np.zeros((self.n_instructors, T), dtype=np.int16),
            np.zeros((self.n_groups, T), dtype=np.int16),
        )

    def _add(self, e, inst, room, time, rc, ic, gc):
        """Add event e to count arrays."""
        s = int(time[e])
        dur = int(self.durations[e])
        r = int(room[e])
        i = int(inst[e])
        end = s + dur
        rc[r, s:end] += 1
        ic[i, s:end] += 1
        for gidx in self.event_group_indices[e]:
            gc[gidx, s:end] += 1
        if self.is_practical[e]:
            for gidx in self.event_group_indices[e]:
                self._prac_occ[gidx, s:end] += 1

    def _remove(self, e, inst, room, time, rc, ic, gc):
        """Remove event e from count arrays."""
        s = int(time[e])
        dur = int(self.durations[e])
        r = int(room[e])
        i = int(inst[e])
        end = s + dur
        rc[r, s:end] -= 1
        ic[i, s:end] -= 1
        for gidx in self.event_group_indices[e]:
            gc[gidx, s:end] -= 1
        if self.is_practical[e]:
            for gidx in self.event_group_indices[e]:
                self._prac_occ[gidx, s:end] -= 1

    def _count_conflicts(self, e, inst, room, time, rc, ic, gc) -> int:
        """Count conflicts for event e (which IS in the maps, so >1 means conflict)."""
        s = int(time[e])
        dur = int(self.durations[e])
        r = int(room[e])
        i = int(inst[e])
        end = s + dur
        conflicts = 0
        # Use scalar loop for small dur — avoids numpy slice overhead
        for q in range(s, end):
            if rc[r, q] > 1:
                conflicts += 1
            if ic[i, q] > 1:
                conflicts += 1
            for gidx in self.event_group_indices[e]:
                if gc[gidx, q] > 1:
                    conflicts += 1
        return conflicts

    def _check_placement(self, e, i_idx, r_idx, t, rc, ic, gc) -> int:
        """Check conflicts for hypothetical placement. Event must be REMOVED from maps."""
        dur = int(self.durations[e])
        end = t + dur
        conflicts = 0

        # Availability — scalar loop for small dur
        ia = self.inst_avail_arr
        ra = self.room_avail_arr
        for q in range(t, end):
            if not ia[i_idx, q]:
                conflicts += 100
            if not ra[r_idx, q]:
                conflicts += 100

        # Exclusivity — any event already there means conflict
        for q in range(t, end):
            if rc[r_idx, q] > 0:
                conflicts += 1
            if ic[i_idx, q] > 0:
                conflicts += 1
            for gidx in self.event_group_indices[e]:
                if gc[gidx, q] > 0:
                    conflicts += 1

        return conflicts

    # ------------------------------------------------------------------
    # Build occupancy
    # ------------------------------------------------------------------

    def _build_counts(self, inst, room, time):
        """Build count arrays from current assignments."""
        rc, ic, gc = self._make_counts()
        for e in range(self.n_events):
            self._add(e, inst, room, time, rc, ic, gc)
        return rc, ic, gc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def construct_feasible(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Construct a near-feasible chromosome using constructive heuristic."""
        if rng is None:
            rng = np.random.default_rng()

        E = self.n_events
        out = np.zeros(3 * E, dtype=int)
        inst = out[0::3]
        room = out[1::3]
        time = out[2::3]

        for e in range(E):
            ai = self.allowed_instructors[e]
            ar = self.allowed_rooms[e]
            at = self.allowed_starts[e]
            inst[e] = rng.choice(ai) if ai else 0
            room[e] = rng.choice(ar) if ar else 0
            time[e] = rng.choice(at) if at else 0

        rc, ic, gc = self._make_counts()

        grp_events: dict[str, list[int]] = defaultdict(list)
        grp_util: dict[str, int] = defaultdict(int)
        for e in range(E):
            for gid in self.events[e]["group_ids"]:
                grp_events[gid].append(e)
                grp_util[gid] += self.events[e]["num_quanta"]

        event_priority = {}
        for e in range(E):
            gids = self.events[e]["group_ids"]
            event_priority[e] = max((grp_util.get(gid, 0) for gid in gids), default=0)

        event_order = sorted(
            range(E),
            key=lambda e: (-event_priority[e], -self.events[e]["num_quanta"]),
        )

        for e in event_order:
            self._find_placement(e, inst, room, time, rc, ic, gc)
            self._add(e, inst, room, time, rc, ic, gc)

        return out

    def repair(
        self,
        chromosome: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Repair a 3*E interleaved chromosome (returns copy).

        Parameters
        ----------
        chromosome : ndarray, shape (3*E,)
        rng : numpy Generator, optional
            When provided, introduces stochastic tie-breaking in placement
            and randomised conflict-processing order so that successive
            repair passes explore *different* local-minimum basins instead
            of converging to the same deterministic fixed point.
        """
        expected = 3 * self.n_events
        if len(chromosome) != expected:
            raise ValueError(
                f"Expected chromosome length {expected}, got {len(chromosome)}"
            )

        out = chromosome.copy()
        inst = out[0::3]
        room = out[1::3]
        time = out[2::3]

        self._fix_domains(inst, room, time)
        self._fix_conflicts(inst, room, time, rng=rng)
        self._fix_group_conflicts(inst, room, time, rng=rng)

        return out

    # ------------------------------------------------------------------
    # Stage 1 — domain violations
    # ------------------------------------------------------------------

    def _fix_domains(self, inst, room, time):
        for e in range(self.n_events):
            ai = self.allowed_instructors[e]
            ar = self.allowed_rooms[e]
            at = self.allowed_starts[e]

            if ai and int(inst[e]) not in ai:
                inst[e] = ai[0]
            if ar and int(room[e]) not in ar:
                room[e] = ar[0]
            if at and int(time[e]) not in at:
                time[e] = at[0]

            valid_starts = self._valid_starts_for(e, int(inst[e]))
            if (
                valid_starts is not None
                and int(time[e]) not in valid_starts
                and valid_starts
            ):
                time[e] = valid_starts[0]

    # ------------------------------------------------------------------
    # Stage 2 — conflict fixing
    # ------------------------------------------------------------------

    def _fix_conflicts(self, inst, room, time, *, rng=None):
        rc, ic, gc = self._build_counts(inst, room, time)

        max_passes = 8
        for _pass in range(max_passes):
            conflict_events = []
            for e in range(self.n_events):
                cc = self._count_conflicts(e, inst, room, time, rc, ic, gc)
                if cc > 0:
                    conflict_events.append((cc, e))

            if not conflict_events:
                break

            if rng is not None:
                # Stochastic: shuffle so different orderings explore
                # different greedy paths (still prioritise worst first
                # on average via partial sort + shuffle within tiers).
                rng.shuffle(conflict_events)
            elif _pass % 2 == 0:
                conflict_events.sort(reverse=True)
            else:
                conflict_events.sort()

            repaired_this_pass: set[int] = set()
            for _, e in conflict_events:
                if e in repaired_this_pass:
                    continue
                cc = self._count_conflicts(e, inst, room, time, rc, ic, gc)
                if cc == 0:
                    continue

                if e in self.paired_event_map:
                    e_pair = self.paired_event_map[e]
                    self._remove(e, inst, room, time, rc, ic, gc)
                    self._remove(e_pair, inst, room, time, rc, ic, gc)
                    self._find_paired_placement(
                        e, e_pair, inst, room, time, rc, ic, gc, rng=rng
                    )
                    self._add(e, inst, room, time, rc, ic, gc)
                    self._add(e_pair, inst, room, time, rc, ic, gc)
                    repaired_this_pass.add(e)
                    repaired_this_pass.add(e_pair)
                else:
                    self._remove(e, inst, room, time, rc, ic, gc)
                    self._find_placement(e, inst, room, time, rc, ic, gc, rng=rng)
                    self._add(e, inst, room, time, rc, ic, gc)

    # ------------------------------------------------------------------
    # Find placement
    # ------------------------------------------------------------------

    def _find_placement(self, e, inst, room, time, rc, ic, gc, *, rng=None) -> bool:
        """Find best (inst, room, time) placement for event e.

        Fully vectorized: computes a 2-D cost matrix of shape
        ``(len(time_candidates), len(allowed_rooms))`` per instructor
        in one shot — no per-timeslot Python loop.

        Event must already be removed from count arrays.

        When *rng* is provided, ties are broken randomly so successive
        repair passes explore different local-minimum basins.
        """
        ai = self.allowed_instructors[e]
        ar_list = self.allowed_rooms[e]
        cur_i = int(inst[e])
        cur_r = int(room[e])
        cur_t = int(time[e])

        # Quick check: current placement already conflict-free?
        best_cost = float(self._check_placement(e, cur_i, cur_r, cur_t, rc, ic, gc))
        if best_cost == 0:
            return True

        best_i, best_r, best_t = cur_i, cur_r, cur_t
        dur = int(self.durations[e])
        gidxs = self.event_group_indices[e]
        n_groups_e = len(gidxs)

        # Allowed rooms as numpy array (reused across instructors)
        ar = np.array(ar_list, dtype=np.intp) if ar_list else np.empty(0, dtype=np.intp)
        n_rooms = len(ar)
        if n_rooms == 0:
            return best_cost == 0

        ia = self.inst_avail_arr
        ra = self.room_avail_arr

        # ── Helper: build 2-D cost matrix for one instructor ─────
        def _score_instructor(i_idx, starts):
            """Compute cost matrix (n_times, n_rooms) for *i_idx*.

            The matrix has **integer** hard-constraint costs plus a small
            **fractional** soft-constraint proxy (< 1.0 total) that acts
            as a tie-breaker without ever overriding hard priorities.

            Soft proxies (all multiplied by 0.01 so they stay sub-integer):
              - Lunch penalty:  +1 per quantum that overlaps the lunch
                window (within-day quantum 2, i.e. 12:00-13:00).
              - Compactness bonus: −0.5 per quantum that is adjacent to
                an already-occupied group slot, encouraging contiguous
                placement.
              - Paired cohort magnet: −0.8 per quantum where the paired
                subgroup already has a practical scheduled, rewarding
                temporal alignment of paired cohort practicals.

            Returns
            -------
            total : ndarray, shape (n_times, n_rooms), float64
            starts_arr : ndarray of the start times used
            """
            at = np.asarray(starts, dtype=np.intp)
            n_times = len(at)
            if n_times == 0:
                return np.empty((0, n_rooms), dtype=np.float64), at

            # Build offset indices: (n_times, dur)
            offsets = at[:, None] + np.arange(dur, dtype=np.intp)  # (nT, dur)

            # ── Fixed cost: instructor availability penalty ──────
            inst_unavail = ~ia[i_idx, offsets]  # (nT, dur)
            inst_avail_pen = np.count_nonzero(inst_unavail, axis=1) * 100  # (nT,)

            # ── Fixed cost: instructor exclusivity ───────────────
            inst_clash = np.count_nonzero(ic[i_idx, offsets] > 0, axis=1)  # (nT,)

            # ── Fixed cost: group exclusivity ────────────────────
            if n_groups_e > 0:
                gidx_arr = np.array(gidxs, dtype=np.intp)
                gc_vals = gc[gidx_arr][:, offsets]  # (nG, nT, dur)
                grp_clash = np.count_nonzero(gc_vals > 0, axis=(0, 2))  # (nT,)
            else:
                grp_clash = np.zeros(n_times, dtype=np.intp)

            fixed_costs = inst_avail_pen + inst_clash + grp_clash  # (nT,)

            # ── Room cost: room exclusivity ──────────────────────
            rc_vals = rc[ar[:, None, None], offsets[None, :, :]]  # (nR, nT, dur)
            room_clash = np.count_nonzero(rc_vals > 0, axis=2)  # (nR, nT)

            # ── Room cost: room availability penalty ─────────────
            ra_vals = ra[ar[:, None, None], offsets[None, :, :]]  # (nR, nT, dur)
            room_avail_pen = np.count_nonzero(~ra_vals, axis=2) * 100  # (nR, nT)

            room_costs = room_clash + room_avail_pen  # (nR, nT)

            # Integer hard total: fixed_costs[nT] broadcast + room_costs[nR, nT]
            total_hard = (fixed_costs[None, :] + room_costs).astype(
                np.float64
            )  # (nR, nT)

            # ── Soft proxy: lunch penalty (floating lunch {2,3,4}) ──
            # A lunch violation ONLY occurs if the proposed block
            # crushes ALL THREE lunch quanta (12:00-15:00).
            _QPD = 7
            within_day = offsets % _QPD  # (nT, dur)
            lunch_hits = (
                (within_day == 2).any(axis=1)
                & (within_day == 3).any(axis=1)
                & (within_day == 4).any(axis=1)
            ).astype(
                np.float64
            )  # (nT,)

            # ── Soft proxy: compactness bonus ─────────────────────
            # For each group, check if quanta immediately before or
            # after the proposed block are already occupied → bonus
            compact_bonus = np.zeros(n_times, dtype=np.float64)
            if n_groups_e > 0:
                # Quanta just before and after the block
                before_q = at - 1  # (nT,)  quantum before block start
                after_q = at + dur  # (nT,)  quantum after block end
                for gidx in gidxs:
                    # Check adjacency (guard bounds)
                    safe_before = np.clip(before_q, 0, gc.shape[1] - 1)
                    safe_after = np.clip(after_q, 0, gc.shape[1] - 1)
                    adj_before = (before_q >= 0) & (gc[gidx, safe_before] > 0)
                    adj_after = (after_q < gc.shape[1]) & (gc[gidx, safe_after] > 0)
                    compact_bonus += adj_before.astype(np.float64) * 0.5
                    compact_bonus += adj_after.astype(np.float64) * 0.5

            # ── Soft proxy: paired cohort alignment magnet ────────
            alignment_bonus = np.zeros(n_times, dtype=np.float64)
            if self.is_practical[e] and n_groups_e > 0:
                for gidx in gidxs:
                    paired_gidx = self.paired_with_group.get(gidx)
                    if paired_gidx is not None:
                        # Reward placing this practical where paired
                        # subgroup already has a practical scheduled
                        paired_active = (
                            self._prac_occ[paired_gidx, offsets] > 0
                        )  # (nT, dur) bool
                        alignment_bonus += (
                            paired_active.sum(axis=1).astype(np.float64) * 0.8
                        )

            # Combine: fractional soft (stays < 1.0 even worst case)
            soft_proxy = (
                lunch_hits.astype(np.float64) - compact_bonus - alignment_bonus
            )  # (nT,)
            # Prevent reward hacking: shift array so minimum is 0.0
            # Without this, negative proxies (e.g. -0.02) combined with
            # hard cost 1 yield 0.98, and int(0.98)==0 tricks the
            # algorithm into thinking the hard constraint is satisfied.
            soft_proxy = soft_proxy - np.min(soft_proxy)
            total = total_hard + 0.01 * soft_proxy[None, :]  # (nR, nT)

            # Return as (nT, nR) for consistency with (time, room) axes
            return total.T, at

        # ── Helper: extract best from cost matrix ────────────────
        def _pick_best(cost_matrix, starts_arr, i_idx):
            """Update best_* from a (nT, nR) float64 cost matrix."""
            nonlocal best_cost, best_i, best_r, best_t
            if cost_matrix.size == 0:
                return False  # nothing to pick

            min_val = float(cost_matrix.min())
            if min_val > best_cost and rng is None:
                return False  # can't improve

            if min_val < best_cost:
                best_cost = min_val
                if rng is not None:
                    ties = np.argwhere(cost_matrix == min_val)
                    pick = ties[rng.integers(len(ties))]
                else:
                    pick = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
                best_t = int(starts_arr[pick[0]])
                best_r = int(ar[pick[1]])
                best_i = i_idx
                if int(min_val) == 0:
                    return True  # hard cost zero — perfect
            elif rng is not None and min_val == best_cost:
                # Accumulate ties for stochastic selection later —
                # but for speed just do a single random pick within
                # this matrix and compare with current best via coin flip.
                ties = np.argwhere(cost_matrix == min_val)
                pick = ties[rng.integers(len(ties))]
                # 50/50 chance to replace current best with this tie
                if rng.random() < len(ties) / (len(ties) + 1):
                    best_t = int(starts_arr[pick[0]])
                    best_r = int(ar[pick[1]])
                    best_i = i_idx
            return False

        # ── Phase 1: Current instructor, group-free times ────────
        time_candidates = self._valid_starts_for(e, cur_i) or self.allowed_starts[e]

        # Vectorized group-free filter
        tc_arr = np.asarray(time_candidates, dtype=np.intp)
        if n_groups_e > 0 and len(tc_arr) > 0:
            gidx_arr = np.array(gidxs, dtype=np.intp)
            tc_offsets = tc_arr[:, None] + np.arange(dur, dtype=np.intp)  # (nT, dur)
            gc_check = gc[gidx_arr][:, tc_offsets]  # (nG, nT, dur)
            has_grp_conflict = np.any(gc_check > 0, axis=(0, 2))  # (nT,)
            group_free_mask = ~has_grp_conflict
            group_free_times = tc_arr[group_free_mask]
        else:
            group_free_times = tc_arr

        # Score all group-free times for current instructor at once
        if len(group_free_times) > 0:
            cost_mat, starts = _score_instructor(cur_i, group_free_times)
            if _pick_best(cost_mat, starts, cur_i):
                inst[e] = best_i
                room[e] = best_r
                time[e] = best_t
                return True

        # ── Phase 2: Other instructors, group-free times ─────────
        instr_order = list(ai[:8])
        if rng is not None:
            rng.shuffle(instr_order)

        for i_idx in instr_order:
            i_times = self._valid_starts_for(e, i_idx) or self.allowed_starts[e]
            i_tc = np.asarray(i_times, dtype=np.intp)
            if len(i_tc) == 0:
                continue

            # Group-free filter for this instructor's times
            if n_groups_e > 0:
                i_offsets = i_tc[:, None] + np.arange(dur, dtype=np.intp)
                gc_check_i = gc[gidx_arr][:, i_offsets]
                i_gf_mask = ~np.any(gc_check_i > 0, axis=(0, 2))
                i_gf_times = i_tc[i_gf_mask]
            else:
                i_gf_times = i_tc

            if len(i_gf_times) == 0:
                continue

            cost_mat, starts = _score_instructor(i_idx, i_gf_times)
            if _pick_best(cost_mat, starts, i_idx):
                inst[e] = best_i
                room[e] = best_r
                time[e] = best_t
                return True

        # ── Phase 3: Fallback — all times (incl. group conflicts) ─
        if len(group_free_times) == 0:
            cost_mat, starts = _score_instructor(cur_i, tc_arr)
            _pick_best(cost_mat, starts, cur_i)

        # ── Assign best found ────────────────────────────────────
        inst[e] = best_i
        room[e] = best_r
        time[e] = best_t
        return int(best_cost) == 0

    # ------------------------------------------------------------------
    # Simultaneous paired placement
    # ------------------------------------------------------------------

    def _find_paired_placement(self, e1, e2, inst, room, time, rc, ic, gc, *, rng=None):
        """Simultaneous placement of paired practicals at the same start time.

        Both events must already be removed from count arrays.
        Finds the best common start time and assigns each event to a
        *different* room, minimising the joint hard-constraint cost.

        Updates *inst*, *room*, *time* arrays for both events.

        Returns
        -------
        bool
            ``True`` if a zero-hard-cost placement was found.
        """
        dur1 = int(self.durations[e1])
        dur2 = int(self.durations[e2])

        ar1_list = self.allowed_rooms[e1]
        ar2_list = self.allowed_rooms[e2]
        ai1 = self.allowed_instructors[e1]
        ai2 = self.allowed_instructors[e2]
        gidxs1 = self.event_group_indices[e1]
        gidxs2 = self.event_group_indices[e2]
        ia = self.inst_avail_arr
        ra = self.room_avail_arr

        best_cost: float = float("inf")
        best_i1 = int(inst[e1])
        best_r1 = int(room[e1])
        best_i2 = int(inst[e2])
        best_r2 = int(room[e2])
        best_t = int(time[e1])

        # Instructor ordering: current first, then alternatives (capped)
        cur_i1, cur_i2 = int(inst[e1]), int(inst[e2])
        i1_cands = [cur_i1] + [i for i in ai1 if i != cur_i1]
        i2_cands = [cur_i2] + [i for i in ai2 if i != cur_i2]
        i1_cands = i1_cands[:6]
        i2_cands = i2_cands[:6]

        for i1 in i1_cands:
            if best_cost == 0:
                break

            vs1 = self._valid_starts_for(e1, i1) or self.allowed_starts[e1]
            if not vs1:
                continue
            vs1_set = set(vs1)

            for i2 in i2_cands:
                if best_cost == 0:
                    break

                # Same instructor at the same time → hard conflict
                same_inst = i1 == i2

                vs2 = self._valid_starts_for(e2, i2) or self.allowed_starts[e2]
                if not vs2:
                    continue

                common_starts = sorted(vs1_set & set(vs2))
                if not common_starts:
                    continue

                # --- Vectorised fixed cost (instructor + group) per timeslot ---
                cs = np.array(common_starts, dtype=np.intp)
                n_cs = len(cs)
                fixed_cost = np.zeros(n_cs, dtype=np.int32)

                if same_inst:
                    fixed_cost[:] += max(dur1, dur2) * 100

                off1 = cs[:, None] + np.arange(dur1, dtype=np.intp)  # (n_cs, dur1)
                off2 = cs[:, None] + np.arange(dur2, dtype=np.intp)  # (n_cs, dur2)

                # Instructor clashes
                fixed_cost += np.sum(ic[i1, off1] > 0, axis=1).astype(np.int32)
                fixed_cost += np.sum(ic[i2, off2] > 0, axis=1).astype(np.int32)

                # Group clashes
                for gidx in gidxs1:
                    fixed_cost += np.sum(gc[gidx, off1] > 0, axis=1).astype(np.int32)
                for gidx in gidxs2:
                    fixed_cost += np.sum(gc[gidx, off2] > 0, axis=1).astype(np.int32)

                # Evaluate timeslots in ascending fixed-cost order
                order = np.argsort(fixed_cost)

                for idx in order:
                    fc = int(fixed_cost[idx])
                    if fc >= best_cost:
                        break  # remainder is worse

                    t_s = int(cs[idx])

                    # --- Room costs for this timeslot ---
                    r1_scores: list[tuple[int, int]] = []
                    for r in ar1_list:
                        c = 0
                        for q in range(t_s, t_s + dur1):
                            if not ra[r, q]:
                                c += 100
                            if rc[r, q] > 0:
                                c += 1
                        r1_scores.append((c, r))
                    r1_scores.sort()

                    r2_scores: list[tuple[int, int]] = []
                    for r in ar2_list:
                        c = 0
                        for q in range(t_s, t_s + dur2):
                            if not ra[r, q]:
                                c += 100
                            if rc[r, q] > 0:
                                c += 1
                        r2_scores.append((c, r))
                    r2_scores.sort()

                    # Best room pair with r1 ≠ r2
                    for cr1, r1 in r1_scores:
                        if fc + cr1 >= best_cost:
                            break
                        for cr2, r2 in r2_scores:
                            if r1 == r2:
                                continue
                            total = fc + cr1 + cr2
                            if total < best_cost or (
                                total == best_cost
                                and rng is not None
                                and rng.random() < 0.3
                            ):
                                best_cost = total
                                best_t = t_s
                                best_i1, best_r1 = i1, r1
                                best_i2, best_r2 = i2, r2
                            break  # r2_scores sorted → first valid r2 is best

        if best_cost == float("inf"):
            # No simultaneous placement found — fall back to sequential
            self._find_placement(e1, inst, room, time, rc, ic, gc, rng=rng)
            self._add(e1, inst, room, time, rc, ic, gc)
            self._find_placement(e2, inst, room, time, rc, ic, gc, rng=rng)
            # Undo e1 add so caller can add both
            self._remove(e1, inst, room, time, rc, ic, gc)
            return False

        # Apply best found placement
        inst[e1] = best_i1
        room[e1] = best_r1
        time[e1] = best_t
        inst[e2] = best_i2
        room[e2] = best_r2
        time[e2] = best_t

        return best_cost == 0

    # ------------------------------------------------------------------
    # Stage 3 — group deconfliction
    # ------------------------------------------------------------------

    def _fix_group_conflicts(self, inst, room, time, *, rng=None):
        rc, ic, gc = self._build_counts(inst, room, time)

        grp_events: dict[str, list[int]] = defaultdict(list)
        grp_util: dict[str, int] = defaultdict(int)
        for e in range(self.n_events):
            for gid in self.events[e]["group_ids"]:
                grp_events[gid].append(e)
                grp_util[gid] += self.events[e]["num_quanta"]

        sorted_groups = sorted(grp_util, key=lambda g: -grp_util[g])

        for _round in range(4):
            any_fix = False
            # Stochastic: shuffle group order so different passes
            # break symmetry in different ways.
            round_groups = list(sorted_groups)
            if rng is not None:
                rng.shuffle(round_groups)

            for gid in round_groups:
                evts = grp_events[gid]
                gidx = self.group_to_idx[gid]

                # Check if this group has conflicts
                has_conflict = False
                for e in evts:
                    s = int(time[e])
                    dur = int(self.durations[e])
                    for q in range(s, s + dur):
                        if gc[gidx, q] > 1:
                            has_conflict = True
                            break
                    if has_conflict:
                        break
                if not has_conflict:
                    continue

                any_fix = True

                # Remove all events in group
                for e in evts:
                    self._remove(e, inst, room, time, rc, ic, gc)

                # Re-insert longest first (shuffle ties when stochastic)
                evts_sorted = sorted(evts, key=lambda e: -self.events[e]["num_quanta"])
                if rng is not None:
                    rng.shuffle(evts_sorted)
                for e in evts_sorted:
                    self._find_placement(e, inst, room, time, rc, ic, gc, rng=rng)
                    self._add(e, inst, room, time, rc, ic, gc)

            if not any_fix:
                break

        # Final pass
        for _pass in range(3):
            conflict_events = []
            for e in range(self.n_events):
                cc = self._count_conflicts(e, inst, room, time, rc, ic, gc)
                if cc > 0:
                    conflict_events.append((cc, e))
            if not conflict_events:
                break
            if rng is not None:
                rng.shuffle(conflict_events)
            else:
                conflict_events.sort(reverse=True)
            repaired_this_pass: set[int] = set()
            for _, e in conflict_events:
                if e in repaired_this_pass:
                    continue
                cc = self._count_conflicts(e, inst, room, time, rc, ic, gc)
                if cc == 0:
                    continue
                if e in self.paired_event_map:
                    e_pair = self.paired_event_map[e]
                    self._remove(e, inst, room, time, rc, ic, gc)
                    self._remove(e_pair, inst, room, time, rc, ic, gc)
                    self._find_paired_placement(
                        e, e_pair, inst, room, time, rc, ic, gc, rng=rng
                    )
                    self._add(e, inst, room, time, rc, ic, gc)
                    self._add(e_pair, inst, room, time, rc, ic, gc)
                    repaired_this_pass.add(e)
                    repaired_this_pass.add(e_pair)
                else:
                    self._remove(e, inst, room, time, rc, ic, gc)
                    self._find_placement(e, inst, room, time, rc, ic, gc, rng=rng)
                    self._add(e, inst, room, time, rc, ic, gc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _valid_starts_for(self, event_idx: int, inst_idx: int) -> list[int] | None:
        key = (event_idx, inst_idx)
        if key in self._inst_time_cache:
            return self._inst_time_cache[key]

        slots = self.inst_avail.get(inst_idx)
        if slots is None:
            self._inst_time_cache[key] = None
            return None

        dur = self.events[event_idx]["num_quanta"]
        valid = [
            s
            for s in self.allowed_starts[event_idx]
            if all(q in slots for q in range(s, s + dur))
        ]
        self._inst_time_cache[key] = valid
        return valid


# ======================================================================
# Batch repair
# ======================================================================


def repair_batch(
    X: np.ndarray,
    repairer: BitsetSchedulingRepair | None = None,
    events_data_path: str = ".cache/events_with_domains.pkl",
) -> np.ndarray:
    """Repair a batch of chromosomes.

    Parameters
    ----------
    X : ndarray, shape (N, 3E)
    repairer : BitsetSchedulingRepair, optional
    events_data_path : str

    Returns
    -------
    X_repaired : ndarray, shape (N, 3E)
    """
    if repairer is None:
        repairer = BitsetSchedulingRepair(events_data_path)

    X = np.asarray(X, dtype=int)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = repairer.repair(X[i])
    return out


# ======================================================================
# Pymoo Repair wrapper
# ======================================================================

try:
    from pymoo.core.repair import Repair

    class PymooBitsetRepair(Repair):
        """Pymoo-compatible repair operator with multi-pass stochastic repair."""

        def __init__(
            self,
            events_data_path: str = ".cache/events_with_domains.pkl",
            passes: int = 3,
        ):
            super().__init__()
            self.engine = BitsetSchedulingRepair(events_data_path)
            self.passes = passes

        def _do(self, problem, x, **kwargs):
            import logging as _logging

            if x.ndim == 1:
                x = x.reshape(1, -1)
            out = np.empty_like(x)
            rng = np.random.default_rng()
            for i in range(x.shape[0]):
                best = x[i]
                for p in range(self.passes):
                    rng_p = np.random.default_rng(rng.integers(2**31) + p)
                    candidate = self.engine.repair(best, rng=rng_p)
                    best = candidate
                out[i] = best
            _logging.getLogger(__name__).debug(
                "Repair: %d individuals, %d passes",
                x.shape[0],
                self.passes,
            )
            return out

except ImportError:
    pass
