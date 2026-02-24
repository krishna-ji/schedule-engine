"""
Gene Domain Store — Pre-computed buckets of valid values per gene.

Instead of recomputing "which instructors / rooms / time-starts are valid?"
on every mutation or initialization call, we compute them ONCE per gene and
cache the result.  Live narrowing (removing blocked times) is cheap O(n).

Usage:
    store = GeneDomainStore(context, qts)
    store.build_domains(genes)               # one-time setup
    domain = store.get_domain(gene_idx)       # O(1) lookup
    free = store.narrow_time_domain(idx, blocked)  # live filter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.utils.room_compatibility import is_room_suitable_for_course

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext
    from src.io.time_system import QuantumTimeSystem

__all__ = ["GeneDomain", "GeneDomainStore"]


@dataclass
class GeneDomain:
    """Pre-computed bucket of valid mutable values for one gene.

    Immutable identity (for reference):
        course_key, group_ids, num_quanta

    Buckets (what this gene CAN be assigned):
        instructors  — qualified instructor IDs
        rooms        — suitable room IDs (type + capacity + features)
        valid_starts — start_quanta values where a contiguous block of
                       num_quanta fits within a single day
    """

    gene_index: int

    # Identity (immutable on gene)
    course_key: tuple[str, str]
    group_ids: list[str]
    num_quanta: int

    # Buckets
    instructors: list[str] = field(default_factory=list)
    rooms: list[str] = field(default_factory=list)
    valid_starts: list[int] = field(default_factory=list)


class GeneDomainStore:
    """Centralised domain storage for all genes in an individual."""

    def __init__(
        self,
        context: SchedulingContext,
        qts: QuantumTimeSystem | None = None,
    ) -> None:
        self.context = context
        self.qts = qts
        self._domains: dict[int, GeneDomain] = {}

        # Pre-compute day boundaries once (used for valid_starts)
        self._day_bounds: list[tuple[int, int]] = []  # [(offset, count), ...]
        if qts is not None:
            for day in qts.DAY_NAMES:
                off = qts.day_quanta_offset.get(day)
                cnt = qts.day_quanta_count.get(day, 0)
                if off is not None and cnt > 0:
                    self._day_bounds.append((off, cnt))

        # Pre-compute available-quanta set for fast membership checks
        self._available_set: set[int] = set(context.available_quanta)

        # Pre-compute per-instructor available quanta (for inst_avail awareness)
        self._instructor_available: dict[str, set[int]] = {}
        for inst_id, inst in context.instructors.items():
            if getattr(inst, "is_full_time", True):
                # Full-time → available at ALL operating quanta
                self._instructor_available[inst_id] = self._available_set
            else:
                # Part-time → only their declared quanta
                self._instructor_available[inst_id] = set(
                    getattr(inst, "available_quanta", set())
                )

    # ================================================================
    # Build
    # ================================================================

    def build_domains(self, genes: list[SessionGene]) -> None:
        """Pre-compute domains for every gene. Call once per generation."""
        self._domains.clear()
        for idx, gene in enumerate(genes):
            self._domains[idx] = self._compute_domain(idx, gene)

    def get_domain(self, gene_idx: int) -> GeneDomain:
        """O(1) lookup for a pre-computed domain."""
        return self._domains[gene_idx]

    def has_domain(self, gene_idx: int) -> bool:
        return gene_idx in self._domains

    def refresh_domain(self, gene_idx: int, gene: SessionGene) -> None:
        """Re-compute a single gene's domain (e.g. after crossover swap)."""
        self._domains[gene_idx] = self._compute_domain(gene_idx, gene)

    # ================================================================
    # Live narrowing
    # ================================================================

    def narrow_time_domain(
        self,
        gene_idx: int,
        blocked_quanta: set[int],
    ) -> list[int]:
        """Return valid_starts minus any that overlap with blocked quanta.

        A start ``s`` is blocked if ANY quantum in [s, s+num_quanta)
        is in ``blocked_quanta``.
        """
        domain = self._domains[gene_idx]
        nq = domain.num_quanta
        if not blocked_quanta:
            return domain.valid_starts
        return [
            s
            for s in domain.valid_starts
            if not any(q in blocked_quanta for q in range(s, s + nq))
        ]

    def instructor_available_starts(
        self,
        gene_idx: int,
        instructor_id: str,
        starts: list[int] | None = None,
    ) -> list[int]:
        """Return starts where the instructor is natively available for the full block.

        This checks the instructor's own ``available_quanta`` (schedule
        preference), NOT the tracker's booking load.  Full-time instructors
        pass every start.
        """
        avail = self._instructor_available.get(instructor_id)
        if avail is None or avail is self._available_set:
            # Unknown or full-time → no restriction
            return (
                list(starts)
                if starts is not None
                else self._domains[gene_idx].valid_starts
            )
        domain = self._domains[gene_idx]
        nq = domain.num_quanta
        pool = starts if starts is not None else domain.valid_starts
        return [s for s in pool if all(q in avail for q in range(s, s + nq))]

    def is_instructor_full_time(self, instructor_id: str) -> bool:
        """True if instructor is full-time (available all quanta)."""
        return self._instructor_available.get(instructor_id) is self._available_set

    # ================================================================
    # Internals
    # ================================================================

    def _compute_domain(self, idx: int, gene: SessionGene) -> GeneDomain:
        """Build the full domain for one gene."""
        course_key = (gene.course_id, gene.course_type)

        return GeneDomain(
            gene_index=idx,
            course_key=course_key,
            group_ids=list(gene.group_ids),
            num_quanta=gene.num_quanta,
            instructors=self._qualified_instructors(course_key),
            rooms=self._suitable_rooms(course_key, gene.group_ids),
            valid_starts=self._valid_starts(gene.num_quanta),
        )

    def _qualified_instructors(self, course_key: tuple[str, str]) -> list[str]:
        """Instructor IDs qualified for a course.

        Returns qualified instructors sorted so that full-time instructors
        come first (they can teach at any time), then part-time instructors
        with actual available quanta.
        """
        full_time: list[str] = []
        part_time_ok: list[str] = []
        part_time_empty: list[str] = []
        for inst_id, inst in self.context.instructors.items():
            if course_key not in getattr(inst, "qualified_courses", []):
                continue
            avail = self._instructor_available.get(inst_id)
            if avail is self._available_set:
                full_time.append(inst_id)
            elif avail:
                part_time_ok.append(inst_id)
            else:
                part_time_empty.append(inst_id)
        result = full_time + part_time_ok
        if not result:
            # Include part-time with zero availability as last resort
            result = part_time_empty
        # STRICT: never allow unqualified instructors — log warning if empty
        if not result:
            logger.warning(
                "No qualified instructors found for %s — gene will keep current instructor",
                course_key,
            )
        return result

    def _suitable_rooms(
        self,
        course_key: tuple[str, str],
        group_ids: list[str],
    ) -> list[str]:
        """Room IDs suitable for this course (type + capacity + features)."""
        course = self.context.courses.get(course_key)
        if not course:
            return list(self.context.rooms.keys())

        required_features = getattr(course, "required_room_features", "lecture")
        req_str = (
            required_features.lower().strip()
            if isinstance(required_features, str)
            else "lecture"
        )

        lab_feats = getattr(course, "specific_lab_features", None) or []

        # max group size
        max_size = 0
        for gid in group_ids:
            grp = self.context.groups.get(gid)
            if grp:
                max_size = max(max_size, getattr(grp, "student_count", 30))
        if max_size == 0:
            max_size = 1

        result: list[str] = []
        for room_id, room in self.context.rooms.items():
            cap = getattr(room, "capacity", 50)
            if cap < max_size:
                continue

            room_features = getattr(room, "room_features", "lecture")
            room_str = (
                room_features.lower().strip()
                if isinstance(room_features, str)
                else "lecture"
            )
            room_spec = getattr(room, "specific_features", None) or []

            if is_room_suitable_for_course(req_str, room_str, lab_feats, room_spec):
                result.append(room_id)

        # STRICT: never allow unsuitable rooms — log warning if empty
        if not result:
            logger.warning(
                "No suitable rooms found for %s — gene will keep current room",
                course_key,
            )
        return result

    def _valid_starts(self, num_quanta: int) -> list[int]:
        """All start positions where num_quanta fit within a single day.

        If num_quanta exceeds a full day, we allow cross-day starts but
        still require the block to be within available quanta.
        """
        if not self._day_bounds:
            # Fallback: treat all available quanta as valid starts
            total = max(self._available_set) + 1 if self._available_set else 42
            return [
                s for s in range(total - num_quanta + 1) if s in self._available_set
            ]

        result: list[int] = []

        total = self.qts.total_quanta if self.qts else 42

        for day_offset, day_count in self._day_bounds:
            if num_quanta <= day_count:
                # Normal case: block fits in one day
                result.extend(
                    range(day_offset, day_offset + day_count - num_quanta + 1)
                )
            # Multi-day session (e.g. 30-quanta practical): start at day boundary
            # but only if the block doesn't overflow the total quantum range
            elif day_offset + num_quanta <= total:
                result.append(day_offset)

        return result


def build_domain_store_for_context(
    context: SchedulingContext,
    qts: QuantumTimeSystem | None = None,
) -> GeneDomainStore:
    """Convenience: create a GeneDomainStore with an optional QTS.

    If ``qts`` is not provided, attempts to fetch the globally-set one.
    """
    if qts is None:
        from src.domain.gene import get_time_system

        qts = get_time_system()
    return GeneDomainStore(context, qts)
