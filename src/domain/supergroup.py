"""Supergroup and Cluster: Decomposition layer for CP-SAT repair.

A **Supergroup** aggregates all groups of the same academic programme across
every semester (e.g. BEI-all = BEI1AB, BEI1A, BEI1B, BEI3AB, BEI3A, BEI3B, …).

A **Cluster** bundles tightly coupled supergroups that share many courses and
instructors and therefore *must* be co-scheduled.  Clean per-programme
decomposition is impossible when two programmes share >10 courses, so clustering
merges programmes like BCT+BEI into one scheduling unit.

Cluster detection is automatic: a programme coupling graph is built from
``Course.enrolled_group_ids``, and programmes connected by ≥2 shared courses are
merged into the same cluster.

The resulting clusters for a typical IOE dataset:

    ARCH     — BAR (100 % independent)
    CIVIL    — BCE (~90 % independent)
    IT       — BCT + BEI (12+ shared courses)
    MECH     — BAM + BME + BIE (15+ shared courses)
    MASTERS  — MEE + MIISE + MMDM (100 % independent)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.group import Group
    from src.domain.types import SchedulingContext

__all__ = [
    "Cluster",
    "Supergroup",
    "build_clusters",
    "build_supergroups",
    "extract_programme_prefix",
]


# ── Prefix extraction ────────────────────────────────────────────────────

# Matches common IOE naming: B{DEPT}{SEM}{SECTION} e.g. BCE3AB, BCT1A, BME8EF
# Also matches master's: MEE1, MIISE2, MMDM3
_PREFIX_RE = re.compile(r"^([A-Z]{2,5})")


def extract_programme_prefix(group_id: str) -> str:
    """Extract the programme prefix from a group_id.

    Examples::

        >>> extract_programme_prefix("BCE3AB")
        'BCE'
        >>> extract_programme_prefix("BME1A")
        'BME'
        >>> extract_programme_prefix("MIISE2")
        'MIISE'

    The heuristic strips trailing digits + optional section letters (A-F).
    """
    m = _PREFIX_RE.match(group_id)
    if not m:
        return group_id  # fallback: use the whole id

    raw = m.group(1)
    # Strip trailing digits that are part of the semester number
    # e.g. "BCE3" should still give "BCE", not "BCE3"
    # But we already matched only alpha chars, so raw is pure letters.
    return raw


# ── Supergroup ────────────────────────────────────────────────────────────


@dataclass
class Supergroup:
    """All groups of a single academic programme across all semesters.

    Attributes:
        programme: Programme prefix (e.g. ``"BCE"``, ``"BCT"``).
        group_ids: Every group_id belonging to this programme (parents +
                   subgroups + standalones).
        course_keys: Set of ``(course_code, course_type)`` keys enrolled by
                     *any* group in this programme.
        instructor_ids: Set of instructor IDs teaching any course in this
                        programme.
    """

    programme: str
    group_ids: set[str] = field(default_factory=set)
    course_keys: set[tuple[str, str]] = field(default_factory=set)
    instructor_ids: set[str] = field(default_factory=set)


def build_supergroups(ctx: SchedulingContext) -> dict[str, Supergroup]:
    """Build one :class:`Supergroup` per programme from a scheduling context.

    Returns a dict keyed by programme prefix.
    """
    sgs: dict[str, Supergroup] = {}

    # 1. Assign every group to its programme
    for gid in ctx.groups:
        prefix = extract_programme_prefix(gid)
        if prefix not in sgs:
            sgs[prefix] = Supergroup(programme=prefix)
        sgs[prefix].group_ids.add(gid)

    # 2. Attach courses and instructors
    for (ccode, ctype), course in ctx.courses.items():
        key = (ccode, ctype)
        for gid in course.enrolled_group_ids:
            prefix = extract_programme_prefix(gid)
            if prefix in sgs:
                sgs[prefix].course_keys.add(key)
                sgs[prefix].instructor_ids.update(course.qualified_instructor_ids)

    return sgs


# ── Cluster ───────────────────────────────────────────────────────────────


@dataclass
class Cluster:
    """A set of tightly coupled supergroups that must be co-scheduled.

    Attributes:
        cluster_id: Human-readable label (e.g. ``"IT"``, ``"MECH"``).
        programmes: Set of programme prefixes in the cluster.
        group_ids: Union of all group IDs across constituent supergroups.
        course_keys: Union of all course keys.
        instructor_ids: Union of all instructor IDs.
    """

    cluster_id: str
    programmes: set[str] = field(default_factory=set)
    group_ids: set[str] = field(default_factory=set)
    course_keys: set[tuple[str, str]] = field(default_factory=set)
    instructor_ids: set[str] = field(default_factory=set)


# ── Cluster detection (Union-Find) ───────────────────────────────────────


class _UnionFind:
    """Minimal union-find for merging programme prefixes."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


# Pre-defined cluster labels (in preference order for readability)
_CLUSTER_LABELS: dict[frozenset[str], str] = {
    frozenset({"BAR"}): "ARCH",
    frozenset({"BCE"}): "CIVIL",
    frozenset({"BCT", "BEI"}): "IT",
    frozenset({"BAM", "BME", "BIE"}): "MECH",
    frozenset({"MEE", "MIISE", "MMDM"}): "MASTERS",
}


def _label_for_programmes(progs: frozenset[str]) -> str:
    """Return a human-readable label for a set of programme prefixes."""
    if progs in _CLUSTER_LABELS:
        return _CLUSTER_LABELS[progs]
    # fallback: join prefixes alphabetically
    return "+".join(sorted(progs))


def build_clusters(
    ctx: SchedulingContext,
    *,
    min_shared_courses: int = 2,
) -> list[Cluster]:
    """Detect and build clusters by analysing course sharing between programmes.

    Two programmes are merged into the same cluster when they share at least
    *min_shared_courses* course keys.

    Parameters
    ----------
    ctx : SchedulingContext
        Fully linked scheduling context.
    min_shared_courses : int
        Minimum number of ``(course_code, course_type)`` keys shared for two
        programmes to be co-scheduled.  Default ``2``.

    Returns
    -------
    list[Cluster]
        Sorted by cluster_id.
    """
    sgs = build_supergroups(ctx)
    progs = list(sgs.keys())

    # Build coupling edges
    uf = _UnionFind()
    for prog in progs:
        uf.find(prog)  # ensure every prog is registered

    for i in range(len(progs)):
        for j in range(i + 1, len(progs)):
            shared = sgs[progs[i]].course_keys & sgs[progs[j]].course_keys
            if len(shared) >= min_shared_courses:
                uf.union(progs[i], progs[j])

    # Group programmes by component
    components: dict[str, set[str]] = defaultdict(set)
    for prog in progs:
        root = uf.find(prog)
        components[root].add(prog)

    # Build cluster objects
    clusters: list[Cluster] = []
    for component_progs in components.values():
        fset = frozenset(component_progs)
        cid = _label_for_programmes(fset)
        cl = Cluster(cluster_id=cid, programmes=set(component_progs))
        for prog in component_progs:
            sg = sgs[prog]
            cl.group_ids |= sg.group_ids
            cl.course_keys |= sg.course_keys
            cl.instructor_ids |= sg.instructor_ids
        clusters.append(cl)

    clusters.sort(key=lambda c: c.cluster_id)
    return clusters
