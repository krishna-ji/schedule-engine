"""
Usage Tracker — Even distribution of instructors, rooms, and time slots.

The root cause of clustering: `random.choice()` everywhere with no memory.
This module tracks how many times each resource is used and biases selection
toward LEAST-USED options.

Usage:
    tracker = UsageTracker(context)
    tracker.build_from_individual(genes)   # snapshot current state
    tracker.remove_gene(gene)              # before re-assigning
    new_inst = tracker.pick_least_used_instructor(candidates)
    new_start = tracker.pick_least_used_start(valid_starts, num_quanta)
    new_room = tracker.pick_least_used_room(candidates, start, num_quanta)
    tracker.add_gene(new_gene)             # commit assignment
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.gene import SessionGene

__all__ = ["UsageTracker"]


@dataclass
class UsageTracker:
    """Tracks per-resource usage counts to enable spreading / load-balancing.

    Counters:
        time_load[q]             — how many genes occupy quantum q
        instructor_load[inst][q] — how many genes inst teaches at q
        room_load[room][q]       — how many genes use room at q
        group_load[group][q]     — how many genes group attends at q
        instructor_total[inst]   — total quanta assigned to instructor
    """

    # ---- Internal counters ----
    time_load: Counter = field(default_factory=Counter)
    instructor_load: dict[str, Counter] = field(default_factory=dict)
    room_load: dict[str, Counter] = field(default_factory=dict)
    group_load: dict[str, Counter] = field(default_factory=dict)
    instructor_total: Counter = field(default_factory=Counter)

    # ================================================================
    # Build / update
    # ================================================================

    def build_from_individual(self, genes: list[SessionGene]) -> None:
        """Build usage counters from scratch for an individual."""
        self.time_load.clear()
        self.instructor_load.clear()
        self.room_load.clear()
        self.group_load.clear()
        self.instructor_total.clear()
        for gene in genes:
            self.add_gene(gene)

    def add_gene(self, gene: SessionGene) -> None:
        """Increment all counters for a gene's resource footprint."""
        quanta = range(gene.start_quanta, gene.start_quanta + gene.num_quanta)

        for q in quanta:
            self.time_load[q] += 1

        # Instructor
        inst = gene.instructor_id
        if inst not in self.instructor_load:
            self.instructor_load[inst] = Counter()
        for q in quanta:
            self.instructor_load[inst][q] += 1
        self.instructor_total[inst] += gene.num_quanta

        # Room
        room = gene.room_id
        if room not in self.room_load:
            self.room_load[room] = Counter()
        for q in quanta:
            self.room_load[room][q] += 1

        # Groups
        for gid in gene.group_ids:
            if gid not in self.group_load:
                self.group_load[gid] = Counter()
            for q in quanta:
                self.group_load[gid][q] += 1

    def remove_gene(self, gene: SessionGene) -> None:
        """Decrement all counters for a gene (call before re-assigning)."""
        quanta = range(gene.start_quanta, gene.start_quanta + gene.num_quanta)

        for q in quanta:
            self.time_load[q] -= 1
            if self.time_load[q] <= 0:
                del self.time_load[q]

        inst = gene.instructor_id
        if inst in self.instructor_load:
            for q in quanta:
                self.instructor_load[inst][q] -= 1
                if self.instructor_load[inst][q] <= 0:
                    del self.instructor_load[inst][q]
        self.instructor_total[inst] -= gene.num_quanta
        if self.instructor_total[inst] <= 0:
            del self.instructor_total[inst]

        room = gene.room_id
        if room in self.room_load:
            for q in quanta:
                self.room_load[room][q] -= 1
                if self.room_load[room][q] <= 0:
                    del self.room_load[room][q]

        for gid in gene.group_ids:
            if gid in self.group_load:
                for q in quanta:
                    self.group_load[gid][q] -= 1
                    if self.group_load[gid][q] <= 0:
                        del self.group_load[gid][q]

    # ================================================================
    # Spreading queries
    # ================================================================

    def pick_least_used_instructor(
        self,
        candidates: list[str],
        *,
        top_k: int = 3,
    ) -> str:
        """Pick instructor with fewest total quanta (with randomised tie-break).

        Returns one of the ``top_k`` least-loaded candidates at random so that
        the same instructor isn't always chosen deterministically.
        """
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        scored = [(cid, self.instructor_total.get(cid, 0)) for cid in candidates]
        scored.sort(key=lambda x: x[1])
        top = scored[: min(top_k, len(scored))]
        return random.choice(top)[0]

    def pick_least_used_start(
        self,
        valid_starts: list[int],
        num_quanta: int,
        *,
        top_k: int = 5,
    ) -> int | None:
        """Pick a start position whose quanta have the lowest cumulative load.

        Scores each candidate start by summing ``time_load[q]`` for
        q in [start, start+num_quanta).  Returns one of the ``top_k``
        lowest-scoring starts at random, or ``None`` if no valid starts.
        """
        if not valid_starts:
            return None

        def _score(start: int) -> int:
            return sum(
                self.time_load.get(q, 0) for q in range(start, start + num_quanta)
            )

        scored = [(s, _score(s)) for s in valid_starts]
        scored.sort(key=lambda x: x[1])
        top = scored[: min(top_k, len(scored))]
        return random.choice(top)[0]

    def pick_least_used_room(
        self,
        candidates: list[str],
        start_quanta: int,
        num_quanta: int,
        *,
        top_k: int = 3,
    ) -> str:
        """Pick room with lowest load during [start, start+num_quanta).

        Prefers rooms that are **free** (load=0) at the chosen time, then
        falls back to least loaded.
        """
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        quanta_range = range(start_quanta, start_quanta + num_quanta)

        def _room_score(rid: str) -> int:
            ctr = self.room_load.get(rid)
            if ctr is None:
                return 0
            return sum(ctr.get(q, 0) for q in quanta_range)

        scored = [(rid, _room_score(rid)) for rid in candidates]
        scored.sort(key=lambda x: x[1])
        top = scored[: min(top_k, len(scored))]
        return random.choice(top)[0]

    # ================================================================
    # Conflict-aware queries (hard constraint helpers)
    # ================================================================

    def instructor_free_starts(
        self,
        instructor_id: str,
        valid_starts: list[int],
        num_quanta: int,
    ) -> list[int]:
        """Return starts where instructor has ZERO load for the full block."""
        ctr = self.instructor_load.get(instructor_id)
        if ctr is None:
            return list(valid_starts)  # never used → all free
        return [
            s
            for s in valid_starts
            if all(ctr.get(q, 0) == 0 for q in range(s, s + num_quanta))
        ]

    def room_free_starts(
        self,
        room_id: str,
        valid_starts: list[int],
        num_quanta: int,
    ) -> list[int]:
        """Return starts where room has ZERO load for the full block."""
        ctr = self.room_load.get(room_id)
        if ctr is None:
            return list(valid_starts)
        return [
            s
            for s in valid_starts
            if all(ctr.get(q, 0) == 0 for q in range(s, s + num_quanta))
        ]

    def group_free_starts(
        self,
        group_ids: list[str],
        valid_starts: list[int],
        num_quanta: int,
        family_map: dict[str, set[str]] | None = None,
    ) -> list[int]:
        """Return starts where NONE of the groups (or their families) are busy.

        If ``family_map`` is provided, also blocks quanta used by related
        groups (parent / sibling).
        """
        # Expand groups to include family members
        all_groups: set[str] = set(group_ids)
        if family_map:
            for gid in group_ids:
                all_groups.update(family_map.get(gid, set()))

        # Collect relevant counters
        relevant_counters: list[Counter] = []
        for gid in all_groups:
            ctr = self.group_load.get(gid)
            if ctr is not None:
                relevant_counters.append(ctr)

        if not relevant_counters:
            return list(valid_starts)

        result: list[int] = []
        for s in valid_starts:
            free = True
            for q in range(s, s + num_quanta):
                for ctr in relevant_counters:
                    if ctr.get(q, 0) > 0:
                        free = False
                        break
                if not free:
                    break
            if free:
                result.append(s)
        return result

    # ================================================================
    # Diagnostics
    # ================================================================

    def time_load_std(self) -> float:
        """Standard deviation of time-slot usage — lower = more spread."""
        if not self.time_load:
            return 0.0
        vals = list(self.time_load.values())
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return float(variance**0.5)

    def instructor_load_std(self) -> float:
        """Std-dev of per-instructor total load."""
        if not self.instructor_total:
            return 0.0
        vals = list(self.instructor_total.values())
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return float(variance**0.5)

    def summary(self) -> dict[str, float]:
        """Quick diagnostics dict."""
        return {
            "time_slots_used": len(self.time_load),
            "time_load_std": round(self.time_load_std(), 2),
            "instructor_load_std": round(self.instructor_load_std(), 2),
            "instructors_active": len(self.instructor_total),
            "max_time_load": max(self.time_load.values()) if self.time_load else 0,
            "min_time_load": min(self.time_load.values()) if self.time_load else 0,
        }
