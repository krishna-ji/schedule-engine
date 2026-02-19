"""Cluster Partitioner: Decompose a chromosome into cluster-scoped subproblems.

Given a full individual (list of ``SessionGene``) and a scheduling context, this
module assigns each gene to exactly one cluster and identifies "bridge genes"
whose groups span multiple clusters or whose instructor teaches across clusters.

Bridge genes are solved first in the Global CP Phase so their assignments can be
frozen as hard constraints for the per-cluster CP models.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.domain.supergroup import Cluster, build_clusters

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext

__all__ = ["ClusterPartition", "partition_genes"]

logger = logging.getLogger(__name__)


@dataclass
class ClusterPartition:
    """Result of partitioning a chromosome into cluster subproblems.

    Attributes
    ----------
    clusters : list[Cluster]
        The detected cluster objects.
    cluster_gene_indices : dict[str, list[int]]
        ``{cluster_id: [gene_indices…]}`` — genes belonging exclusively to
        one cluster (after bridge genes are removed).
    bridge_gene_indices : list[int]
        Gene indices that span multiple clusters (cross-programme groups
        or cross-cluster instructors).
    gene_cluster_map : dict[int, str]
        Maps every gene index to its assigned cluster_id (bridge genes are
        mapped to ``"__bridge__"``).
    shared_instructor_ids : set[str]
        Instructors whose teaching load spans multiple clusters.
    group_to_cluster : dict[str, str]
        Maps every group_id to its cluster_id.
    """

    clusters: list[Cluster]
    cluster_gene_indices: dict[str, list[int]] = field(default_factory=dict)
    bridge_gene_indices: list[int] = field(default_factory=list)
    gene_cluster_map: dict[int, str] = field(default_factory=dict)
    shared_instructor_ids: set[str] = field(default_factory=set)
    group_to_cluster: dict[str, str] = field(default_factory=dict)


def partition_genes(
    genes: list[SessionGene],
    ctx: SchedulingContext,
    *,
    min_shared_courses: int = 2,
) -> ClusterPartition:
    """Partition *genes* into cluster-scoped subproblems.

    Parameters
    ----------
    genes : list[SessionGene]
        The full chromosome.
    ctx : SchedulingContext
        Fully linked scheduling context.
    min_shared_courses : int
        Forwarded to :func:`build_clusters`.

    Returns
    -------
    ClusterPartition
    """
    clusters = build_clusters(ctx, min_shared_courses=min_shared_courses)

    # Build group → cluster mapping
    group_to_cluster: dict[str, str] = {}
    for cl in clusters:
        for gid in cl.group_ids:
            group_to_cluster[gid] = cl.cluster_id

    # Identify instructors spanning multiple clusters
    instructor_clusters: dict[str, set[str]] = defaultdict(set)
    for cl in clusters:
        for iid in cl.instructor_ids:
            instructor_clusters[iid].add(cl.cluster_id)
    shared_instructors = {
        iid for iid, cls in instructor_clusters.items() if len(cls) > 1
    }

    # Classify each gene
    cluster_gene_indices: dict[str, list[int]] = {cl.cluster_id: [] for cl in clusters}
    bridge_gene_indices: list[int] = []
    gene_cluster_map: dict[int, str] = {}

    for i, gene in enumerate(genes):
        # Determine which clusters this gene's groups belong to
        gene_clusters: set[str] = set()
        for gid in gene.group_ids:
            cid = group_to_cluster.get(gid)
            if cid:
                gene_clusters.add(cid)

        is_bridge = False
        if len(gene_clusters) > 1:
            # Groups span multiple clusters → bridge gene
            is_bridge = True
        elif gene.instructor_id in shared_instructors:
            # Instructor spans multiple clusters → bridge gene
            is_bridge = True

        if is_bridge:
            bridge_gene_indices.append(i)
            gene_cluster_map[i] = "__bridge__"
        elif len(gene_clusters) == 1:
            cid = next(iter(gene_clusters))
            cluster_gene_indices[cid].append(i)
            gene_cluster_map[i] = cid
        else:
            # Gene has no recognized groups — shouldn't happen in valid data.
            # Assign to bridge for safety.
            bridge_gene_indices.append(i)
            gene_cluster_map[i] = "__bridge__"
            logger.warning(
                "Gene %d has groups %s that don't map to any cluster",
                i,
                gene.group_ids,
            )

    partition = ClusterPartition(
        clusters=clusters,
        cluster_gene_indices=cluster_gene_indices,
        bridge_gene_indices=bridge_gene_indices,
        gene_cluster_map=gene_cluster_map,
        shared_instructor_ids=shared_instructors,
        group_to_cluster=group_to_cluster,
    )

    # Log partition summary
    total = len(genes)
    bridge_pct = 100.0 * len(bridge_gene_indices) / total if total else 0
    logger.info(
        "Partitioned %d genes → %d bridge (%.1f%%) + %d clusters",
        total,
        len(bridge_gene_indices),
        bridge_pct,
        len(clusters),
    )
    for cl in clusters:
        n = len(cluster_gene_indices.get(cl.cluster_id, []))
        logger.info(
            "  Cluster %-8s: %3d genes, %d groups, %d programmes %s",
            cl.cluster_id,
            n,
            len(cl.group_ids),
            len(cl.programmes),
            sorted(cl.programmes),
        )

    return partition
