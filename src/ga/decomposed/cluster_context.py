"""Cluster Sub-Context: Builds per-cluster SchedulingContext subsets.

Given a full SchedulingContext and a Cluster, extracts only the courses,
groups, instructors, and rooms relevant to that cluster — creating a
self-contained mini-problem that can be solved independently.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.domain.supergroup import Cluster, build_clusters, extract_programme_prefix
from src.domain.types import SchedulingContext

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.gene import SessionGene
    from src.domain.group import Group
    from src.domain.instructor import Instructor
    from src.domain.room import Room

__all__ = [
    "ClusterContext",
    "PartitionResult",
    "build_cluster_contexts",
    "merge_cluster_individuals",
    "partition_individual",
]

logger = logging.getLogger(__name__)


# ── Cluster Context ───────────────────────────────────────────────────────


@dataclass
class ClusterContext:
    """A cluster with its own reduced SchedulingContext and gene mapping.

    Attributes
    ----------
    cluster : Cluster
        The cluster metadata (programmes, group_ids, course_keys, etc.).
    sub_ctx : SchedulingContext
        A self-contained scheduling context with only cluster-relevant entities.
    global_gene_indices : list[int]
        Indices into the global chromosome for genes belonging to this cluster.
    """

    cluster: Cluster
    sub_ctx: SchedulingContext
    global_gene_indices: list[int] = field(default_factory=list)


@dataclass
class PartitionResult:
    """Result of partitioning a global individual into cluster sub-problems.

    Attributes
    ----------
    cluster_contexts : dict[str, ClusterContext]
        Mapping from cluster_id to its context + gene indices.
    bridge_gene_indices : list[int]
        Indices of genes that span multiple clusters (shared instructors/rooms).
    shared_instructor_ids : set[str]
        Instructors teaching in multiple clusters.
    shared_room_ids : set[str]
        Rooms potentially used by multiple clusters.
    """

    cluster_contexts: dict[str, ClusterContext]
    bridge_gene_indices: list[int] = field(default_factory=list)
    shared_instructor_ids: set[str] = field(default_factory=set)
    shared_room_ids: set[str] = field(default_factory=set)


# ── Builder Functions ─────────────────────────────────────────────────────


def build_cluster_contexts(
    ctx: SchedulingContext,
    *,
    min_shared_courses: int = 2,
) -> PartitionResult:
    """Build per-cluster scheduling contexts from a full context.

    This is the main setup function called once at initialization.
    It creates reduced contexts for each cluster, identifies shared resources,
    and provides the mapping infrastructure for gene partitioning.

    Parameters
    ----------
    ctx : SchedulingContext
        The full scheduling context with all entities linked.
    min_shared_courses : int
        Minimum shared course keys to merge programmes into a cluster.

    Returns
    -------
    PartitionResult
        Contains cluster contexts, shared resource info, and mapping data.
    """
    clusters = build_clusters(ctx, min_shared_courses=min_shared_courses)

    # Build group → cluster mapping
    group_to_cluster: dict[str, str] = {}
    for cl in clusters:
        for gid in cl.group_ids:
            group_to_cluster[gid] = cl.cluster_id

    # Detect shared instructors (teach in >1 cluster)
    instructor_clusters: dict[str, set[str]] = {}
    for cl in clusters:
        for iid in cl.instructor_ids:
            instructor_clusters.setdefault(iid, set()).add(cl.cluster_id)
    shared_instructors = {
        iid for iid, cids in instructor_clusters.items() if len(cids) > 1
    }

    # Detect shared rooms: all rooms are potentially shared (capacity pool)
    # We mark rooms as shared that could serve multiple clusters
    # In practice, rooms are a global resource — clusters share the room pool
    all_room_ids = set(ctx.rooms.keys())
    shared_rooms = all_room_ids  # All rooms are shared by default

    # Build per-cluster sub-contexts
    cluster_ctxs: dict[str, ClusterContext] = {}

    for cl in clusters:
        # Filter courses: those enrolled by this cluster's groups
        sub_courses: dict[tuple[str, str], Course] = {
            key: course
            for key, course in ctx.courses.items()
            if any(gid in cl.group_ids for gid in course.enrolled_group_ids)
        }

        # Filter groups: only those in this cluster
        sub_groups: dict[str, Group] = {
            gid: ctx.groups[gid] for gid in cl.group_ids if gid in ctx.groups
        }

        # Filter instructors: those teaching cluster courses
        sub_instructor_ids: set[str] = set()
        for course in sub_courses.values():
            sub_instructor_ids.update(course.qualified_instructor_ids)
        sub_instructors: dict[str, Instructor] = {
            iid: ctx.instructors[iid]
            for iid in sub_instructor_ids
            if iid in ctx.instructors
        }

        # Rooms: include all rooms (they're a shared pool)
        # Each cluster can use any room, but CP-SAT will enforce room exclusivity
        sub_rooms: dict[str, Room] = dict(ctx.rooms)

        # Build cohort pairs relevant to this cluster
        sub_cohort_pairs: list[tuple[str, str]] = []
        if ctx.cohort_pairs:
            for a, b in ctx.cohort_pairs:
                if a in cl.group_ids or b in cl.group_ids:
                    sub_cohort_pairs.append((a, b))

        # Build family map for this cluster
        sub_family_map: dict[str, set[str]] = {}
        for gid in cl.group_ids:
            if gid in ctx.family_map:
                sub_family_map[gid] = ctx.family_map[gid] & cl.group_ids

        sub_ctx = SchedulingContext(
            courses=sub_courses,
            groups=sub_groups,
            instructors=sub_instructors,
            rooms=sub_rooms,
            available_quanta=list(ctx.available_quanta),
            config=ctx.config,
            cohort_pairs=sub_cohort_pairs,
            family_map=sub_family_map,
        )

        cluster_ctxs[cl.cluster_id] = ClusterContext(
            cluster=cl,
            sub_ctx=sub_ctx,
        )

    logger.info(
        "Built %d cluster contexts: %s",
        len(cluster_ctxs),
        {cid: len(cc.sub_ctx.courses) for cid, cc in cluster_ctxs.items()},
    )
    logger.info("Shared instructors: %d", len(shared_instructors))

    return PartitionResult(
        cluster_contexts=cluster_ctxs,
        shared_instructor_ids=shared_instructors,
        shared_room_ids=shared_rooms,
    )


# ── Gene Partitioning ─────────────────────────────────────────────────────


def partition_individual(
    genes: list[SessionGene],
    partition: PartitionResult,
) -> None:
    """Classify each gene into its cluster, updating global_gene_indices.

    Genes whose groups span multiple clusters or whose instructor is shared
    are classified as bridge genes.

    Side effect: populates ``partition.cluster_contexts[cid].global_gene_indices``
    and ``partition.bridge_gene_indices``.
    """
    # Reset
    for cc in partition.cluster_contexts.values():
        cc.global_gene_indices = []
    partition.bridge_gene_indices = []

    # Build group → cluster lookup
    group_to_cluster: dict[str, str] = {}
    for cid, cc in partition.cluster_contexts.items():
        for gid in cc.cluster.group_ids:
            group_to_cluster[gid] = cid

    for idx, gene in enumerate(genes):
        # Determine which clusters the gene's groups belong to
        gene_clusters = set()
        for gid in gene.group_ids:
            maybe_cid = group_to_cluster.get(gid)
            if maybe_cid:
                gene_clusters.add(maybe_cid)

        if len(gene_clusters) == 0:
            # Orphan gene — shouldn't happen but defensive
            logger.warning(
                "Gene %d has no cluster mapping: groups=%s", idx, gene.group_ids
            )
            if partition.cluster_contexts:
                # Assign to first cluster as fallback
                first_cid = next(iter(partition.cluster_contexts))
                partition.cluster_contexts[first_cid].global_gene_indices.append(idx)
        elif len(gene_clusters) == 1:
            cid = next(iter(gene_clusters))
            # Check if instructor is shared across clusters
            if gene.instructor_id in partition.shared_instructor_ids:
                partition.bridge_gene_indices.append(idx)
            else:
                partition.cluster_contexts[cid].global_gene_indices.append(idx)
        else:
            # Gene spans multiple clusters → bridge
            partition.bridge_gene_indices.append(idx)


def extract_cluster_genes(
    genes: list[SessionGene],
    cluster_ctx: ClusterContext,
) -> list[SessionGene]:
    """Extract a sub-chromosome from the global chromosome for a cluster.

    Returns deep copies of the genes so mutations don't affect the global list.
    """
    return [copy.deepcopy(genes[i]) for i in cluster_ctx.global_gene_indices]


def merge_cluster_individuals(
    global_genes: list[SessionGene],
    cluster_results: dict[str, list[SessionGene]],
    partition: PartitionResult,
) -> list[SessionGene]:
    """Merge per-cluster gene lists back into a single chromosome.

    Parameters
    ----------
    global_genes : list[SessionGene]
        The original global chromosome (used as template).
    cluster_results : dict[str, list[SessionGene]]
        Per-cluster optimized gene lists, keyed by cluster_id.
    partition : PartitionResult
        The partition info mapping gene indices.

    Returns
    -------
    list[SessionGene]
        A new chromosome with cluster-optimized genes merged in.
    """
    merged = [copy.deepcopy(g) for g in global_genes]

    for cid, cc in partition.cluster_contexts.items():
        if cid not in cluster_results:
            continue
        cluster_genes = cluster_results[cid]
        if len(cluster_genes) != len(cc.global_gene_indices):
            logger.warning(
                "Cluster %s: gene count mismatch (expected %d, got %d)",
                cid,
                len(cc.global_gene_indices),
                len(cluster_genes),
            )
            continue
        for local_idx, global_idx in enumerate(cc.global_gene_indices):
            merged[global_idx] = cluster_genes[local_idx]

    return merged
