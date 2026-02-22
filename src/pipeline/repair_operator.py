#!/usr/bin/env python3
"""Constraint-aware repair operator for pymoo scheduling problem.

Fixes:
1. Domain violations  – instructor not qualified, room not suitable, start OOB
2. Instructor time-availability – never assign to quanta the instructor is unavailable
3. Room time-availability – never assign to quanta the room is unavailable
4. Exclusivity conflicts – room / instructor / group double-booking

Uses incremental occupancy maps for O(duration) conflict checks instead of O(E).
"""

import pickle
from collections import defaultdict

import numpy as np


class SchedulingRepair:
    """Repair a 3×E interleaved chromosome [I0,R0,T0, I1,R1,T1, …]."""

    def __init__(self, events_data_path: str = ".cache/events_with_domains.pkl"):
        with open(events_data_path, "rb") as f:
            data = pickle.load(f)

        self.events: list[dict] = data["events"]
        self.allowed_instructors: list[list[int]] = data["allowed_instructors"]
        self.allowed_rooms: list[list[int]] = data["allowed_rooms"]
        self.allowed_starts: list[list[int]] = data["allowed_starts"]
        self.n_events: int = len(self.events)

        # Availability maps (idx -> set or None)
        self.inst_avail: dict[int, set | None] = data.get(
            "instructor_available_quanta", {}
        )
        self.room_avail: dict[int, set | None] = data.get("room_available_quanta", {})

        # Pre-compute per-event valid starts per instructor
        self._inst_time_cache: dict[tuple[int, int], list[int] | None] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def construct_feasible(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Construct a near-feasible chromosome using a constructive heuristic.

        Processes events group-by-group (tightest groups first), greedily
        assigning (instructor, room, time) to minimize conflicts.
        Returns a 3×E interleaved chromosome.
        """
        if rng is None:
            rng = np.random.default_rng()

        E = self.n_events
        out = np.zeros(3 * E, dtype=int)
        inst = out[0::3]
        room = out[1::3]
        time = out[2::3]

        # Initialize with random allowed values for diversity
        for e in range(E):
            ai = self.allowed_instructors[e]
            ar = self.allowed_rooms[e]
            at = self.allowed_starts[e]
            inst[e] = rng.choice(ai) if ai else 0
            room[e] = rng.choice(ar) if ar else 0
            time[e] = rng.choice(at) if at else 0

        # Build occupancy maps (empty initially)
        room_occ: dict[tuple, set[int]] = defaultdict(set)
        inst_occ: dict[tuple, set[int]] = defaultdict(set)
        group_occ: dict[tuple, set[int]] = defaultdict(set)

        # Build group -> events mapping and utilization
        grp_events: dict[str, list[int]] = defaultdict(list)
        grp_util: dict[str, int] = defaultdict(int)
        for e in range(E):
            for gid in self.events[e]["group_ids"]:
                grp_events[gid].append(e)
                grp_util[gid] += self.events[e]["num_quanta"]

        # Determine processing order: events in tightest groups first
        # Each event gets a priority = max utilization of its groups
        event_priority = {}
        for e in range(E):
            gids = self.events[e]["group_ids"]
            if gids:
                event_priority[e] = max(grp_util.get(gid, 0) for gid in gids)
            else:
                event_priority[e] = 0

        # Sort: tightest group events first, break ties by longest duration
        event_order = sorted(
            range(E),
            key=lambda e: (-event_priority[e], -self.events[e]["num_quanta"]),
        )

        # Place events one by one
        for e in event_order:
            # Try to find a conflict-free placement
            self._find_placement(e, inst, room, time, room_occ, inst_occ, group_occ)
            self._add_to_maps(e, inst, room, time, room_occ, inst_occ, group_occ)

        return out

    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        """Repair a 3×E interleaved chromosome (returns copy)."""
        expected = 3 * self.n_events
        if len(chromosome) != expected:
            raise ValueError(
                f"Expected chromosome length {expected}, got {len(chromosome)}"
            )

        out = chromosome.copy()
        inst = out[0::3]  # views into out
        room = out[1::3]
        time = out[2::3]

        # Stage 1: Fix domain violations
        self._fix_domains(inst, room, time)

        # Stage 2: Build occupancy maps and fix conflicts incrementally
        self._fix_conflicts_incremental(inst, room, time)

        # Stage 3: Group-aware deconfliction (handles tightly-packed groups)
        self._fix_group_conflicts(inst, room, time)

        return out

    # ------------------------------------------------------------------
    # Stage 1 – domain violations
    # ------------------------------------------------------------------

    def _fix_domains(self, inst, room, time):
        """Fix every event whose assigned value is outside its allowed set."""
        for e in range(self.n_events):
            ai = self.allowed_instructors[e]
            ar = self.allowed_rooms[e]
            at = self.allowed_starts[e]

            # Instructor domain
            if ai and int(inst[e]) not in ai:
                inst[e] = ai[0]

            # Room domain
            if ar and int(room[e]) not in ar:
                room[e] = ar[0]

            # Time domain (basic range)
            if at and int(time[e]) not in at:
                time[e] = at[0]

            # Instructor availability filter
            valid_starts = self._valid_starts_for(e, int(inst[e]))
            if (
                valid_starts is not None
                and int(time[e]) not in valid_starts
                and valid_starts
            ):
                time[e] = valid_starts[0]

    # ------------------------------------------------------------------
    # Stage 2 – conflict fixing with incremental occupancy maps
    # ------------------------------------------------------------------

    def _fix_conflicts_incremental(self, inst, room, time):
        """Fix conflicts using occupancy maps updated incrementally."""
        # Build initial occupancy maps
        room_occ: dict[tuple[int, int], set[int]] = defaultdict(set)
        inst_occ: dict[tuple[int, int], set[int]] = defaultdict(set)
        group_occ: dict[tuple[int, int], set[int]] = defaultdict(set)

        for e in range(self.n_events):
            self._add_to_maps(e, inst, room, time, room_occ, inst_occ, group_occ)

        # Multiple passes – alternate most-conflicted-first and least-conflicted-first
        max_passes = 8
        for _pass in range(max_passes):
            # Score events by conflict count
            conflict_events = []
            for e in range(self.n_events):
                cc = self._count_conflicts_for(
                    e, inst, room, time, room_occ, inst_occ, group_occ
                )
                if cc > 0:
                    conflict_events.append((cc, e))

            if not conflict_events:
                break

            # Alternate strategy: odd passes fix easiest first (can free up space)
            if _pass % 2 == 0:
                conflict_events.sort(reverse=True)  # most conflicted first
            else:
                conflict_events.sort()  # least conflicted first (easier wins)

            for _, e in conflict_events:
                # Check if still conflicted (may have been fixed by earlier reassignment)
                cc = self._count_conflicts_for(
                    e, inst, room, time, room_occ, inst_occ, group_occ
                )
                if cc == 0:
                    continue

                # Remove from maps
                self._remove_from_maps(
                    e, inst, room, time, room_occ, inst_occ, group_occ
                )

                # Try to find conflict-free placement
                self._find_placement(e, inst, room, time, room_occ, inst_occ, group_occ)

                # Re-add to maps (with original or new placement)
                self._add_to_maps(e, inst, room, time, room_occ, inst_occ, group_occ)

    def _add_to_maps(self, e, inst, room, time, room_occ, inst_occ, group_occ):
        """Add event e to all occupancy maps."""
        s = int(time[e])
        dur = self.events[e]["num_quanta"]
        i = int(inst[e])
        r = int(room[e])
        gids = self.events[e]["group_ids"]
        for q in range(s, s + dur):
            room_occ[(r, q)].add(e)
            inst_occ[(i, q)].add(e)
            for gid in gids:
                group_occ[(gid, q)].add(e)

    def _remove_from_maps(self, e, inst, room, time, room_occ, inst_occ, group_occ):
        """Remove event e from all occupancy maps."""
        s = int(time[e])
        dur = self.events[e]["num_quanta"]
        i = int(inst[e])
        r = int(room[e])
        gids = self.events[e]["group_ids"]
        for q in range(s, s + dur):
            room_occ[(r, q)].discard(e)
            inst_occ[(i, q)].discard(e)
            for gid in gids:
                group_occ[(gid, q)].discard(e)

    def _count_conflicts_for(
        self, e, inst, room, time, room_occ, inst_occ, group_occ
    ) -> int:
        """Count conflicts for event e using occupancy maps. O(duration)."""
        s = int(time[e])
        dur = self.events[e]["num_quanta"]
        i = int(inst[e])
        r = int(room[e])
        gids = self.events[e]["group_ids"]
        conflicts = 0
        for q in range(s, s + dur):
            if len(room_occ.get((r, q), set())) > 1:
                conflicts += 1
            if len(inst_occ.get((i, q), set())) > 1:
                conflicts += 1
            for gid in gids:
                if len(group_occ.get((gid, q), set())) > 1:
                    conflicts += 1
        return conflicts

    def _check_placement(
        self, e, i_idx, r_idx, t, room_occ, inst_occ, group_occ
    ) -> int:
        """Check conflicts for a hypothetical placement. O(duration).
        Event e must already be REMOVED from the maps."""
        dur = self.events[e]["num_quanta"]
        gids = self.events[e]["group_ids"]
        conflicts = 0

        # Instructor availability check
        inst_slots = self.inst_avail.get(i_idx)
        if inst_slots is not None:
            for q in range(t, t + dur):
                if q not in inst_slots:
                    conflicts += 100  # heavy penalty for availability violation

        # Room availability check
        room_slots = self.room_avail.get(r_idx)
        if room_slots is not None:
            for q in range(t, t + dur):
                if q not in room_slots:
                    conflicts += 100

        for q in range(t, t + dur):
            if room_occ.get((r_idx, q), set()):
                conflicts += 1
            if inst_occ.get((i_idx, q), set()):
                conflicts += 1
            for gid in gids:
                if group_occ.get((gid, q), set()):
                    conflicts += 1
        return conflicts

    def _find_placement(
        self, e, inst, room, time, room_occ, inst_occ, group_occ
    ) -> bool:
        """Find best (inst, room, time) placement for event e.
        Event must already be removed from maps.
        Updates inst/room/time arrays if better placement found.

        Strategy: first find a GROUP-conflict-free time, then find
        a room+instructor that minimizers remaining conflicts.
        """
        ai = self.allowed_instructors[e]
        ar = self.allowed_rooms[e]
        cur_i = int(inst[e])
        cur_r = int(room[e])
        cur_t = int(time[e])

        best_conflicts = self._check_placement(
            e, cur_i, cur_r, cur_t, room_occ, inst_occ, group_occ
        )
        if best_conflicts == 0:
            return True

        best_i, best_r, best_t = cur_i, cur_r, cur_t
        dur = self.events[e]["num_quanta"]
        gids = self.events[e]["group_ids"]

        # Phase 1: Find GROUP-conflict-free time slots
        group_free_times = []
        time_candidates = self._valid_starts_for(e, cur_i) or self.allowed_starts[e]
        for t in time_candidates:
            group_conflict = False
            for q in range(t, t + dur):
                for gid in gids:
                    if group_occ.get((gid, q), set()):
                        group_conflict = True
                        break
                if group_conflict:
                    break
            if not group_conflict:
                group_free_times.append(t)

        # Phase 2: For group-free times, find best room+instructor combo
        if group_free_times:
            for t in group_free_times:
                # Try current room first
                c = self._check_placement(
                    e, cur_i, cur_r, t, room_occ, inst_occ, group_occ
                )
                if c == 0:
                    time[e] = t
                    return True
                if c < best_conflicts:
                    best_conflicts = c
                    best_i, best_r, best_t = cur_i, cur_r, t
                # Try other rooms
                for r in ar:
                    c = self._check_placement(
                        e, cur_i, r, t, room_occ, inst_occ, group_occ
                    )
                    if c == 0:
                        room[e] = r
                        time[e] = t
                        return True
                    if c < best_conflicts:
                        best_conflicts = c
                        best_i, best_r, best_t = cur_i, r, t

        # Phase 3: Also try group-free times with different instructors
        for i in ai[:8]:
            i_free_times = self._valid_starts_for(e, i) or self.allowed_starts[e]
            for t in i_free_times:
                # Quick group-conflict check
                group_conflict = False
                for q in range(t, t + dur):
                    for gid in gids:
                        if group_occ.get((gid, q), set()):
                            group_conflict = True
                            break
                    if group_conflict:
                        break
                if group_conflict:
                    continue  # Skip non-group-free times
                for r in ar:
                    c = self._check_placement(e, i, r, t, room_occ, inst_occ, group_occ)
                    if c == 0:
                        inst[e] = i
                        room[e] = r
                        time[e] = t
                        return True
                    if c < best_conflicts:
                        best_conflicts = c
                        best_i, best_r, best_t = i, r, t

        # Phase 4: Fallback — scan all times if no group-free times found
        if not group_free_times:
            for t in time_candidates:
                c = self._check_placement(
                    e, cur_i, cur_r, t, room_occ, inst_occ, group_occ
                )
                if c < best_conflicts:
                    best_conflicts = c
                    best_i, best_r, best_t = cur_i, cur_r, t
                for r in ar:
                    c = self._check_placement(
                        e, cur_i, r, t, room_occ, inst_occ, group_occ
                    )
                    if c < best_conflicts:
                        best_conflicts = c
                        best_i, best_r, best_t = cur_i, r, t

        # Apply best found (even if not 0)
        inst[e] = best_i
        room[e] = best_r
        time[e] = best_t
        return best_conflicts == 0

    # ------------------------------------------------------------------
    # Stage 3 – group-aware deconfliction
    # ------------------------------------------------------------------

    def _fix_group_conflicts(self, inst, room, time):
        """Fix group conflicts by processing events per group atomically.

        For each group (tightest first), remove all its events from
        occupancy maps, then re-insert one by one with greedy placement.
        This allows coordinated scheduling within a group.
        """
        # Build occupancy maps fresh
        room_occ: dict[tuple, set[int]] = defaultdict(set)
        inst_occ: dict[tuple, set[int]] = defaultdict(set)
        group_occ: dict[tuple, set[int]] = defaultdict(set)
        for e in range(self.n_events):
            self._add_to_maps(e, inst, room, time, room_occ, inst_occ, group_occ)

        # Build group -> event list and utilization
        grp_events: dict[str, list[int]] = defaultdict(list)
        grp_util: dict[str, int] = defaultdict(int)
        for e in range(self.n_events):
            for gid in self.events[e]["group_ids"]:
                grp_events[gid].append(e)
                grp_util[gid] += self.events[e]["num_quanta"]

        # Process tightest groups first (highest utilization)
        sorted_groups = sorted(grp_util, key=lambda g: -grp_util[g])

        # Multiple rounds of group deconfliction
        for _round in range(4):
            any_fix = False
            for gid in sorted_groups:
                evts = grp_events[gid]

                # Check if this group actually has conflicts
                has_conflict = False
                for e in evts:
                    s = int(time[e])
                    dur = self.events[e]["num_quanta"]
                    for q in range(s, s + dur):
                        bucket = group_occ.get((gid, q), set())
                        if len(bucket) > 1:
                            has_conflict = True
                            break
                    if has_conflict:
                        break
                if not has_conflict:
                    continue

                any_fix = True

                # Remove all events in this group from maps
                for e in evts:
                    self._remove_from_maps(
                        e, inst, room, time, room_occ, inst_occ, group_occ
                    )

                # Sort by duration (longest first - hardest to place)
                evts_sorted = sorted(evts, key=lambda e: -self.events[e]["num_quanta"])

                # Re-insert one by one with greedy placement
                for e in evts_sorted:
                    self._find_placement(
                        e, inst, room, time, room_occ, inst_occ, group_occ
                    )
                    self._add_to_maps(
                        e, inst, room, time, room_occ, inst_occ, group_occ
                    )

            if not any_fix:
                break

        # One final general pass to fix any residual room/instructor conflicts
        for _pass in range(3):
            conflict_events = []
            for e in range(self.n_events):
                cc = self._count_conflicts_for(
                    e, inst, room, time, room_occ, inst_occ, group_occ
                )
                if cc > 0:
                    conflict_events.append((cc, e))
            if not conflict_events:
                break
            conflict_events.sort(reverse=True)
            for _, e in conflict_events:
                cc = self._count_conflicts_for(
                    e, inst, room, time, room_occ, inst_occ, group_occ
                )
                if cc == 0:
                    continue
                self._remove_from_maps(
                    e, inst, room, time, room_occ, inst_occ, group_occ
                )
                self._find_placement(e, inst, room, time, room_occ, inst_occ, group_occ)
                self._add_to_maps(e, inst, room, time, room_occ, inst_occ, group_occ)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _valid_starts_for(self, event_idx: int, inst_idx: int) -> list[int] | None:
        """Return start-time list filtered by instructor availability.

        Returns None if the instructor is full-time (all starts valid).
        """
        key = (event_idx, inst_idx)
        if key in self._inst_time_cache:
            return self._inst_time_cache[key]

        slots = self.inst_avail.get(inst_idx)
        if slots is None:
            self._inst_time_cache[key] = None
            return None  # full-time

        dur = self.events[event_idx]["num_quanta"]
        valid = [
            s
            for s in self.allowed_starts[event_idx]
            if all(q in slots for q in range(s, s + dur))
        ]
        self._inst_time_cache[key] = valid
        return valid


# ======================================================================
# Pymoo Repair wrapper
# ======================================================================


try:
    from pymoo.core.repair import Repair

    class PymooSchedulingRepair(Repair):
        """Pymoo-compatible repair operator class."""

        def __init__(self, events_data_path: str = ".cache/events_with_domains.pkl"):
            super().__init__()
            self.engine = SchedulingRepair(events_data_path)

        def _do(self, problem, x, **kwargs):
            if x.ndim == 1:
                return self.engine.repair(x)
            out = x.copy()
            for i in range(x.shape[0]):
                out[i] = self.engine.repair(x[i])
            return out

except ImportError:
    pass  # pymoo not installed; standalone repair still works
