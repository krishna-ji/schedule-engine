"""Timetable: The missing central abstraction.

A Timetable wraps a decoded individual (list of SessionGene) together with
the scheduling context and pre-built indexes that allow O(1) lookups by
group, instructor, room, and quantum.

This class eliminates:
- 9 scattered `decode_individual()` calls across the codebase
- 14 redundant map-rebuilds per evaluation (group-day map rebuilt 3x alone)
- The need for `CourseSession` as a separate decode target

Design principles:
- Indexes are computed ONCE on construction, reused by all consumers
- Immutable after construction (indexes are read-only views)
- No global state, no singletons, no `get_config()` calls
- Every constraint, evaluator, and repair operator receives a Timetable
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.gene import SessionGene
    from src.domain.group import Group
    from src.domain.instructor import Instructor
    from src.domain.room import Room
    from src.domain.types import SchedulingContext
    from src.io.time_system import QuantumTimeSystem


__all__ = ["ConflictPair", "Timetable"]


# Conflict pair type


@dataclass(frozen=True, slots=True)
class ConflictPair:
    """Two genes that conflict on a shared resource at a specific quantum."""

    gene_a_idx: int
    gene_b_idx: int
    resource_type: str  # "group", "instructor", "room"
    resource_id: str
    quantum: int


class Timetable:
    """Pre-indexed view of a complete schedule.

    Constructed from a list of ``SessionGene`` objects and a
    ``SchedulingContext``.  On construction it builds every index that the
    constraint / evaluator / repair layer needs, so downstream consumers
    never have to rebuild maps from scratch.

    Indexes built (computed once, read many):

    1. **Occupancy maps** ``(entity_id, quantum) → list[gene_idx]``
       - ``_group_occ``:      for student-group exclusivity
       - ``_instructor_occ``: for instructor exclusivity
       - ``_room_occ``:       for room exclusivity

    2. **Daily schedule maps** ``entity_id → day_name → set[within_day_q]``
       - ``_group_daily``:      for student compactness / lunch / break
       - ``_instructor_daily``: for instructor compactness

    3. **Completeness map** ``(course_code, course_type, group_id) → int``
       - ``_course_group_quanta``: quanta delivered per course-group pair

    4. **Course daily map** ``(course_id, course_type) → day → list[within_day_q]``
       - ``_course_daily``: for session continuity analysis

    5. **Practical quanta** ``(course_id, course_type, group_id) → set[quantum]``
       - ``_practical_quanta``: for paired-cohort practical alignment

    Args:
        genes: The chromosome (individual) to index.
        context: Immutable scheduling universe (courses, groups,
            instructors, rooms).
        qts: Time system for day-based decomposition. If ``None``,
            day-based indexes will not be built.
    """

    # Construction

    __slots__ = (
        "_context",
        # Course daily
        "_course_daily",
        # Completeness
        "_course_group_quanta",
        "_genes",
        "_genes_at_quantum",
        # Quick per-entity gene lists
        "_genes_by_group",
        "_genes_by_instructor",
        "_genes_by_room",
        # Daily schedule indexes
        "_group_daily",
        # Occupancy indexes  (entity_id, quantum) → list[gene_idx]
        "_group_occ",
        "_instructor_daily",
        "_instructor_occ",
        # Practical quanta
        "_practical_quanta",
        "_qts",
        "_room_occ",
        # Cached decoded sessions (lazy)
        "_sessions",
    )

    def __init__(
        self,
        genes: list[SessionGene],
        context: SchedulingContext,
        qts: QuantumTimeSystem | None = None,
    ) -> None:
        self._genes = genes
        self._context = context
        self._qts = qts
        self._sessions: list | None = None  # lazy
        self._build_indexes()

    # Public: Gene access

    @property
    def genes(self) -> list[SessionGene]:
        """All genes in this timetable (read-only reference)."""
        return self._genes

    @property
    def sessions(self) -> list:
        """Decoded CourseSession list (lazy, cached).

        Bridge to the current constraint API which expects
        ``list[CourseSession]``.  Computed once on first access.
        """
        if self._sessions is None:
            from src.io.decoder import decode_individual

            self._sessions = decode_individual(
                self._genes,
                self._context.courses,
                self._context.instructors,
                self._context.groups,
                self._context.rooms,
            )
        return self._sessions

    @property
    def context(self) -> SchedulingContext:
        return self._context

    @property
    def qts(self) -> QuantumTimeSystem | None:
        return self._qts

    def __len__(self) -> int:
        return len(self._genes)

    def __getitem__(self, idx: int) -> SessionGene:
        return self._genes[idx]

    def __iter__(self) -> Any:
        return iter(self._genes)

    # Public: Per-entity gene lists

    def genes_for_group(self, group_id: str) -> list[int]:
        """Return gene indices for a given group."""
        return self._genes_by_group.get(group_id, [])

    def genes_for_instructor(self, instructor_id: str) -> list[int]:
        """Return gene indices for a given instructor."""
        return self._genes_by_instructor.get(instructor_id, [])

    def genes_for_room(self, room_id: str) -> list[int]:
        """Return gene indices for a given room."""
        return self._genes_by_room.get(room_id, [])

    def genes_at_quantum(self, quantum: int) -> list[int]:
        """Return gene indices active at a specific quantum."""
        return self._genes_at_quantum.get(quantum, [])

    # Public: Occupancy maps (used directly by hard constraints)

    @property
    def group_occupancy(self) -> dict[tuple[str, int], list[int]]:
        """``(group_id, quantum) → [gene_idx, …]``."""
        return self._group_occ

    @property
    def instructor_occupancy(self) -> dict[tuple[str, int], list[int]]:
        """``(instructor_id, quantum) → [gene_idx, …]``."""
        return self._instructor_occ

    @property
    def room_occupancy(self) -> dict[tuple[str, int], list[int]]:
        """``(room_id, quantum) → [gene_idx, …]``."""
        return self._room_occ

    # Public: Daily schedule maps (used by soft constraints)

    @property
    def group_daily(self) -> dict[str, dict[str, set[int]]]:
        """``group_id → day_name → {within_day_quantum, …}``."""
        return self._group_daily

    @property
    def instructor_daily(self) -> dict[str, dict[str, set[int]]]:
        """``instructor_id → day_name → {within_day_quantum, …}``."""
        return self._instructor_daily

    # Public: Completeness & continuity maps

    @property
    def course_group_quanta(self) -> dict[tuple[str, str, str], int]:
        """``(course_code, course_type, group_id) → total_quanta_scheduled``."""
        return self._course_group_quanta

    @property
    def course_daily(self) -> dict[tuple[str, str], dict[str, list[int]]]:
        """``(course_id, course_type) → day_name → [within_day_quantum, …]``."""
        return self._course_daily

    @property
    def practical_quanta(self) -> dict[tuple[str, str, str], set[int]]:
        """``(course_id, course_type, group_id) → {quantum, …}`` for practicals."""
        return self._practical_quanta

    # Public: Conflict detection (high-level)

    def group_conflicts(self) -> list[ConflictPair]:
        """Find all (gene_a, gene_b) pairs with group-time overlap."""
        return self._find_conflicts(self._group_occ, "group")

    def instructor_conflicts(self) -> list[ConflictPair]:
        """Find all (gene_a, gene_b) pairs with instructor-time overlap."""
        return self._find_conflicts(self._instructor_occ, "instructor")

    def room_conflicts(self) -> list[ConflictPair]:
        """Find all (gene_a, gene_b) pairs with room-time overlap."""
        return self._find_conflicts(self._room_occ, "room")

    def all_conflicts(self) -> list[ConflictPair]:
        """All hard exclusivity conflicts (group + instructor + room)."""
        return (
            self.group_conflicts() + self.instructor_conflicts() + self.room_conflicts()
        )

    def count_group_violations(self) -> int:
        """Number of (group, quantum) slots with >1 session."""
        return sum(len(idxs) - 1 for idxs in self._group_occ.values() if len(idxs) > 1)

    def count_instructor_violations(self) -> int:
        """Number of (instructor, quantum) slots with >1 session."""
        return sum(
            len(idxs) - 1 for idxs in self._instructor_occ.values() if len(idxs) > 1
        )

    def count_room_violations(self) -> int:
        """Number of (room, quantum) slots with >1 session."""
        return sum(len(idxs) - 1 for idxs in self._room_occ.values() if len(idxs) > 1)

    # Public: Lookup helpers for per-gene checks

    def course_for_gene(self, gene: SessionGene) -> Course:
        """Resolve the Course object for a gene."""
        return self._context.courses[(gene.course_id, gene.course_type)]

    def instructor_for_gene(self, gene: SessionGene) -> Instructor:
        """Resolve the Instructor object for a gene."""
        return self._context.instructors[gene.instructor_id]

    def room_for_gene(self, gene: SessionGene) -> Room:
        """Resolve the Room object for a gene."""
        return self._context.rooms[gene.room_id]

    def groups_for_gene(self, gene: SessionGene) -> list[Group]:
        """Resolve Group objects for a gene."""
        return [self._context.groups[gid] for gid in gene.group_ids]

    # Public: Factory

    @classmethod
    def from_individual(
        cls,
        individual: list[SessionGene],
        context: SchedulingContext,
        qts: QuantumTimeSystem | None = None,
    ) -> Timetable:
        """Convenience alias — same as ``Timetable(individual, context, qts)``."""
        return cls(individual, context, qts)

    # Internal: Index building

    def _build_indexes(self) -> None:
        """Build all indexes in a single pass over the genes."""
        # Occupancy: (entity_id, quantum) → [gene_idx, ...]
        group_occ: dict[tuple[str, int], list[int]] = defaultdict(list)
        instructor_occ: dict[tuple[str, int], list[int]] = defaultdict(list)
        room_occ: dict[tuple[str, int], list[int]] = defaultdict(list)

        # Per-entity gene lists
        genes_by_group: dict[str, list[int]] = defaultdict(list)
        genes_by_instructor: dict[str, list[int]] = defaultdict(list)
        genes_by_room: dict[str, list[int]] = defaultdict(list)
        genes_at_quantum: dict[int, list[int]] = defaultdict(list)

        # Daily schedules: entity_id → day → set[within_day_q]
        group_daily: dict[str, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        instructor_daily: dict[str, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

        # Completeness: (course_code, course_type, group_id) → total quanta
        course_group_quanta: dict[tuple[str, str, str], int] = defaultdict(int)

        # Course daily: (course_id, course_type) → day → [within_day_q, ...]
        course_daily: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Practical quanta: (course_id, course_type, group_id) → set[quantum]
        practical_quanta: dict[tuple[str, str, str], set[int]] = defaultdict(set)

        qts = self._qts

        for idx, gene in enumerate(self._genes):
            quanta = range(gene.start_quanta, gene.end_quanta)

            # Per-entity gene lists
            for gid in gene.group_ids:
                genes_by_group[gid].append(idx)
            genes_by_instructor[gene.instructor_id].append(idx)
            for co_id in getattr(gene, "co_instructor_ids", []):
                genes_by_instructor[co_id].append(idx)
            genes_by_room[gene.room_id].append(idx)

            # Completeness: accumulate quanta per (course, type, group)
            for gid in gene.group_ids:
                course_group_quanta[
                    (gene.course_id, gene.course_type, gid)
                ] += gene.num_quanta

            # Practical quanta tracking
            if gene.course_type == "practical":
                for gid in gene.group_ids:
                    for q in quanta:
                        practical_quanta[(gene.course_id, gene.course_type, gid)].add(q)

            for q in quanta:
                # Occupancy maps
                for gid in gene.group_ids:
                    group_occ[(gid, q)].append(idx)
                instructor_occ[(gene.instructor_id, q)].append(idx)
                # Co-instructors also occupy the same time slots
                for co_id in getattr(gene, "co_instructor_ids", []):
                    instructor_occ[(co_id, q)].append(idx)
                room_occ[(gene.room_id, q)].append(idx)

                # Per-quantum gene list
                genes_at_quantum[q].append(idx)

                # Day-based indexes (only if QTS available)
                if qts is not None:
                    try:
                        day, within_day = self._quantum_to_day(q, qts)
                    except ValueError:
                        continue

                    for gid in gene.group_ids:
                        group_daily[gid][day].add(within_day)
                    instructor_daily[gene.instructor_id][day].add(within_day)
                    for co_id in getattr(gene, "co_instructor_ids", []):
                        instructor_daily[co_id][day].add(within_day)
                    course_daily[(gene.course_id, gene.course_type)][day].append(
                        within_day
                    )

        # Freeze into instance
        self._group_occ = dict(group_occ)
        self._instructor_occ = dict(instructor_occ)
        self._room_occ = dict(room_occ)
        self._genes_by_group = dict(genes_by_group)
        self._genes_by_instructor = dict(genes_by_instructor)
        self._genes_by_room = dict(genes_by_room)
        self._genes_at_quantum = dict(genes_at_quantum)
        self._group_daily = dict(group_daily)
        self._instructor_daily = dict(instructor_daily)
        self._course_group_quanta = dict(course_group_quanta)
        self._course_daily = dict(course_daily)
        self._practical_quanta = dict(practical_quanta)

    # Internal: Helpers

    @staticmethod
    def _quantum_to_day(quantum: int, qts: QuantumTimeSystem) -> tuple[str, int]:
        """Convert continuous quantum → (day_name, within_day_quantum).

        Inlined to avoid importing from utils.time_helpers and to keep
        the class self-contained.
        """
        for day in qts.DAY_NAMES:
            day_offset = qts.day_quanta_offset.get(day)
            day_count = qts.day_quanta_count.get(day, 0)
            if day_offset is None or day_count <= 0:
                continue
            if day_offset <= quantum < day_offset + day_count:
                return day, quantum - day_offset
        raise ValueError(f"Quantum {quantum} out of valid range")

    @staticmethod
    def _find_conflicts(
        occ_map: dict[tuple[str, int], list[int]],
        resource_type: str,
    ) -> list[ConflictPair]:
        """Extract conflict pairs from an occupancy map."""
        conflicts: list[ConflictPair] = []
        for (res_id, quantum), gene_idxs in occ_map.items():
            if len(gene_idxs) <= 1:
                continue
            # Report all pairwise conflicts
            conflicts.extend(
                ConflictPair(
                    gene_a_idx=gene_idxs[i],
                    gene_b_idx=gene_idxs[j],
                    resource_type=resource_type,
                    resource_id=res_id,
                    quantum=quantum,
                )
                for i in range(len(gene_idxs))
                for j in range(i + 1, len(gene_idxs))
            )
        return conflicts
