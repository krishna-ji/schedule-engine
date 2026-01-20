"""Heuristic selection strategies for experiment notebooks.

Provides round-robin and adaptive heuristic selection for Modes C-E.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Callable

from src.ga.sessiongene import SessionGene

if TYPE_CHECKING:
    from src.notebooks.data_loader import ScheduleData


# ============================================================================
# Simple Repair Heuristics (for notebooks)
# ============================================================================


def fix_instructor_conflicts(
    individual: list[SessionGene],
    data: ScheduleData,
) -> int:
    """Fix instructor double-booking conflicts.

    Returns number of fixes made.
    """
    fixes = 0
    instructor_slots: dict[str, dict[int, SessionGene]] = defaultdict(dict)

    for gene in individual:
        for t in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
            if t in instructor_slots[gene.instructor_id]:
                # Conflict! Move this gene
                new_start = _find_free_slot(
                    gene, instructor_slots[gene.instructor_id], data.qts.total_quanta
                )
                if new_start is not None:
                    gene.start_quanta = new_start
                    fixes += 1
            instructor_slots[gene.instructor_id][t] = gene

    return fixes


def fix_room_conflicts(
    individual: list[SessionGene],
    data: ScheduleData,
) -> int:
    """Fix room double-booking conflicts.

    Returns number of fixes made.
    """
    fixes = 0
    room_slots: dict[str, dict[int, SessionGene]] = defaultdict(dict)

    for gene in individual:
        for t in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
            if t in room_slots[gene.room_id]:
                # Conflict! Move to different time
                new_start = _find_free_slot(
                    gene, room_slots[gene.room_id], data.qts.total_quanta
                )
                if new_start is not None:
                    gene.start_quanta = new_start
                    fixes += 1
            room_slots[gene.room_id][t] = gene

    return fixes


def fix_group_conflicts(
    individual: list[SessionGene],
    data: ScheduleData,
) -> int:
    """Fix student group double-booking conflicts.

    Returns number of fixes made.
    """
    fixes = 0
    group_slots: dict[str, dict[int, SessionGene]] = defaultdict(dict)

    for gene in individual:
        for group_id in gene.group_ids:
            for t in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
                if t in group_slots[group_id]:
                    new_start = _find_free_slot(
                        gene, group_slots[group_id], data.qts.total_quanta
                    )
                    if new_start is not None:
                        gene.start_quanta = new_start
                        fixes += 1
                        break
                group_slots[group_id][t] = gene

    return fixes


def _find_free_slot(
    gene: SessionGene,
    occupied: dict[int, SessionGene],
    total_quanta: int,
    max_attempts: int = 20,
) -> int | None:
    """Find a free time slot for a gene."""
    duration = gene.num_quanta

    for _ in range(max_attempts):
        start = random.randint(0, max(0, total_quanta - duration))
        slots = set(range(start, start + duration))
        if not any(t in occupied for t in slots):
            return start

    return None


# ============================================================================
# Heuristic Selection Strategies
# ============================================================================


# Available heuristics
HEURISTICS = {
    "fix_instructor": fix_instructor_conflicts,
    "fix_room": fix_room_conflicts,
    "fix_group": fix_group_conflicts,
}


class RoundRobinSelector:
    """Round-robin heuristic selection (Mode C)."""

    def __init__(self, heuristic_names: list[str] | None = None) -> None:
        self.heuristics = heuristic_names or list(HEURISTICS.keys())
        self.index = 0

    def select(self) -> str:
        """Select next heuristic in round-robin order."""
        name = self.heuristics[self.index]
        self.index = (self.index + 1) % len(self.heuristics)
        return name

    def apply(
        self,
        individual: list[SessionGene],
        data: ScheduleData,
    ) -> tuple[str, int]:
        """Select and apply next heuristic."""
        name = self.select()
        fn = HEURISTICS[name]
        fixes = fn(individual, data)
        return name, fixes


class AdaptiveSelector:
    """Adaptive heuristic selection based on success rate (Mode D)."""

    def __init__(
        self,
        heuristic_names: list[str] | None = None,
        learning_rate: float = 0.1,
        min_prob: float = 0.05,
    ) -> None:
        self.heuristics = heuristic_names or list(HEURISTICS.keys())
        self.learning_rate = learning_rate
        self.min_prob = min_prob

        # Initialize equal probabilities
        n = len(self.heuristics)
        self.probs = {name: 1.0 / n for name in self.heuristics}

        # Track performance
        self.successes: dict[str, int] = defaultdict(int)
        self.applications: dict[str, int] = defaultdict(int)

    def select(self) -> str:
        """Select heuristic based on adaptive probabilities."""
        names = list(self.probs.keys())
        weights = [self.probs[n] for n in names]
        return random.choices(names, weights=weights, k=1)[0]

    def update(self, name: str, improvement: int) -> None:
        """Update probabilities based on result."""
        self.applications[name] += 1

        if improvement > 0:
            self.successes[name] += 1

            # Reward successful heuristic
            reward = self.learning_rate * improvement
            self.probs[name] = min(1.0, self.probs[name] + reward)

            # Normalize
            total = sum(self.probs.values())
            self.probs = {
                k: max(self.min_prob, v / total) for k, v in self.probs.items()
            }

    def apply(
        self,
        individual: list[SessionGene],
        data: ScheduleData,
    ) -> tuple[str, int]:
        """Select, apply, and update based on result."""
        name = self.select()
        fn = HEURISTICS[name]
        fixes = fn(individual, data)
        self.update(name, fixes)
        return name, fixes

    def get_stats(self) -> dict[str, float]:
        """Get current heuristic probabilities and success rates."""
        stats = {}
        for name in self.heuristics:
            apps = self.applications[name]
            rate = self.successes[name] / apps if apps > 0 else 0.0
            stats[name] = {"prob": self.probs[name], "success_rate": rate, "apps": apps}
        return stats
