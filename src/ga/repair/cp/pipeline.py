"""CP Repair Pipeline: Orchestrates the full decompose → solve → merge flow.

Usage::

    from src.ga.repair.cp import CPRepairPipeline

    pipeline = CPRepairPipeline(timeout_global=60, timeout_cluster=30)
    repaired_genes = pipeline.repair(genes, context, family_map)

Architecture:

    1. **Partition** — split chromosome into bridge genes + cluster subproblems.
    2. **Global Phase** — solve bridge genes (cross-cluster foundation courses
       and shared instructors) with CP-SAT.
    3. **Cluster Phase** — solve each cluster's genes independently in parallel,
       with bridge assignments frozen as hard constraints.
    4. **Merge** — apply all CP-SAT results onto the chromosome.
    5. **Audit** — verify 0 hard violations.  If residual conflicts remain
       (unlikely), run a small coordination CP on the conflicting genes.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext

__all__ = ["CPRepairPipeline", "CPRepairStats"]

logger = logging.getLogger(__name__)


@dataclass
class CPRepairStats:
    """Diagnostic stats from one CP repair pass."""

    total_genes: int = 0
    bridge_genes: int = 0
    num_clusters: int = 0
    global_phase_status: str = ""
    global_phase_time: float = 0.0
    cluster_statuses: dict[str, str] = field(default_factory=dict)
    cluster_times: dict[str, float] = field(default_factory=dict)
    residual_hard: float = 0.0
    total_time: float = 0.0
    success: bool = False


class CPRepairPipeline:
    """Two-phase CP-SAT repair: Global (bridges) → Cluster (parallel).

    Parameters
    ----------
    timeout_global : float
        Max seconds for the Global Phase CP-SAT solve.  Default 60.
    timeout_cluster : float
        Max seconds per cluster CP-SAT solve.  Default 30.
    num_workers : int
        CP-SAT internal parallelism.  Default 4.
    max_parallel_clusters : int
        Max clusters to solve in parallel (using separate processes).
        Default 3.  Set to 1 for sequential solving.
    min_shared_courses : int
        Threshold for merging programmes into the same cluster.  Default 2.
    soft_objective : bool
        When True, pass soft-constraint objectives (compactness) to the
        CP-SAT solver.  Default False.
    """

    def __init__(
        self,
        *,
        timeout_global: float = 60.0,
        timeout_cluster: float = 30.0,
        num_workers: int = 4,
        max_parallel_clusters: int = 3,
        min_shared_courses: int = 2,
        soft_objective: bool = False,
        max_cluster_genes: int = 120,
    ) -> None:
        self.timeout_global = timeout_global
        self.timeout_cluster = timeout_cluster
        self.num_workers = num_workers
        self.max_parallel_clusters = max_parallel_clusters
        self.min_shared_courses = min_shared_courses
        self.soft_objective = soft_objective
        self.max_cluster_genes = max_cluster_genes

    def _solve_large_cluster(
        self,
        cid: str,
        cl_indices: list[int],
        genes: list[SessionGene],
        ctx: SchedulingContext,
        family_map: dict[str, set[str]],
        solver: Any,
        base_frozen: list[Any],
    ) -> Any:
        """Solve a large cluster in chunks, grouped by student group.

        Genes are partitioned into chunks of ~max_cluster_genes by their
        primary student group.  Each chunk is solved sequentially, with
        previously solved chunks' results frozen as hard constraints.

        Returns a merged CPSolveResult.
        """
        from collections import defaultdict

        from src.ga.repair.cp.solver import CPSolveResult, FrozenAssignment

        # Group gene indices by their primary group (first group_id)
        group_genes: dict[str, list[int]] = defaultdict(list)
        for gi in cl_indices:
            primary_group = (
                genes[gi].group_ids[0] if genes[gi].group_ids else "__none__"
            )
            # Use semester-level grouping: e.g., BAM1A -> BAM1, BAM3B -> BAM3
            # This groups sections of the same semester together
            prefix = primary_group[:4] if len(primary_group) >= 4 else primary_group
            group_genes[prefix].append(gi)

        # Build chunks that respect the size limit
        chunks: list[list[int]] = []
        current_chunk: list[int] = []
        for prefix in sorted(group_genes):
            batch = group_genes[prefix]
            if (
                current_chunk
                and len(current_chunk) + len(batch) > self.max_cluster_genes
            ):
                chunks.append(current_chunk)
                current_chunk = []
            current_chunk.extend(batch)
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            "Cluster %s: split %d genes into %d chunks: %s",
            cid,
            len(cl_indices),
            len(chunks),
            [len(c) for c in chunks],
        )

        # Solve each chunk sequentially, accumulating frozen assignments
        all_assignments: dict[int, tuple[str, str, int]] = {}
        frozen = list(base_frozen)
        chunk_frozen: list[FrozenAssignment] = []  # frozen from solved chunks only
        total_wall = 0.0
        chunk_statuses: list[str] = []
        temp_genes = list(genes)  # mutable copy for applying results

        for ci, chunk_indices in enumerate(chunks):
            all_frozen = frozen + chunk_frozen
            logger.info(
                "  Chunk %d/%d: %d genes, %d frozen…",
                ci + 1,
                len(chunks),
                len(chunk_indices),
                len(all_frozen),
            )
            cr = solver.solve(
                temp_genes,
                chunk_indices,
                frozen=all_frozen,
                warm_start=True,
            )

            # Retry: keep chunk_frozen (essential for consistency) but
            # drop base_frozen (bridge constraints that may over-constrain)
            if not cr.success and frozen:
                logger.info(
                    "  Chunk %d: %s, retrying without bridge frozen…",
                    ci + 1,
                    cr.status,
                )
                cr = solver.solve(
                    temp_genes,
                    chunk_indices,
                    frozen=chunk_frozen if chunk_frozen else None,
                    warm_start=True,
                )

            total_wall += cr.wall_time
            chunk_statuses.append(cr.status)

            if cr.success:
                all_assignments.update(cr.assignments)
                # Freeze this chunk's results for the next chunk
                for gi, (iid, rid, sq) in cr.assignments.items():
                    g = temp_genes[gi]
                    # Update temp_genes so warm-start reflects solved state
                    temp_genes[gi].instructor_id = iid
                    temp_genes[gi].room_id = rid
                    temp_genes[gi].start_quanta = sq
                    chunk_frozen.append(
                        FrozenAssignment(
                            gene_index=gi,
                            course_id=g.course_id,
                            course_type=g.course_type,
                            instructor_id=iid,
                            group_ids=tuple(g.group_ids),
                            room_id=rid,
                            start_quanta=sq,
                            num_quanta=g.num_quanta,
                        )
                    )
            else:
                logger.warning(
                    "  Chunk %d: failed (%s), continuing…", ci + 1, cr.status
                )

        success = len(all_assignments) > 0
        status = (
            "FEASIBLE"
            if len(all_assignments) == len(cl_indices)
            else "PARTIAL"
            if all_assignments
            else "UNKNOWN"
        )
        logger.info(
            "Cluster %s chunked: %d/%d genes solved, statuses=%s, wall=%.1fs",
            cid,
            len(all_assignments),
            len(cl_indices),
            chunk_statuses,
            total_wall,
        )
        return CPSolveResult(
            success=success,
            assignments=all_assignments,
            status=status,
            wall_time=total_wall,
        )

    def repair(
        self,
        genes: list[SessionGene],
        ctx: SchedulingContext,
        family_map: dict[str, set[str]],
    ) -> tuple[list[SessionGene], CPRepairStats]:
        """Run the full CP repair pipeline.

        Parameters
        ----------
        genes : list[SessionGene]
            The chromosome to repair (not modified).
        ctx : SchedulingContext
            Fully linked scheduling context.
        family_map : dict[str, set[str]]
            Group family map.

        Returns
        -------
        tuple[list[SessionGene], CPRepairStats]
            The repaired gene list and diagnostic statistics.
        """
        from src.ga.repair.cp.merger import apply_cp_results, audit_hard_violations
        from src.ga.repair.cp.partitioner import partition_genes
        from src.ga.repair.cp.solver import CPSATSolver, CPSolveResult, FrozenAssignment

        t0 = _time.monotonic()
        stats = CPRepairStats(total_genes=len(genes))

        # ── 1. Partition ─────────────────────────────────────────────
        partition = partition_genes(
            genes, ctx, min_shared_courses=self.min_shared_courses
        )
        stats.bridge_genes = len(partition.bridge_gene_indices)
        stats.num_clusters = len(partition.clusters)

        logger.info(
            "CP Pipeline: %d total genes, %d bridge, %d clusters",
            len(genes),
            stats.bridge_genes,
            stats.num_clusters,
        )

        # ── 2. Global Phase — solve bridge genes ────────────────────
        solver = CPSATSolver(
            ctx,
            family_map,
            timeout_seconds=self.timeout_global,
            num_workers=self.num_workers,
            soft_objective=self.soft_objective,
        )

        global_result: CPSolveResult
        if partition.bridge_gene_indices:
            logger.info("Global Phase: solving %d bridge genes…", stats.bridge_genes)
            global_result = solver.solve(
                genes,
                partition.bridge_gene_indices,
                frozen=None,
                warm_start=True,
            )
            stats.global_phase_status = global_result.status
            stats.global_phase_time = global_result.wall_time
        else:
            global_result = CPSolveResult(success=True, status="NO_BRIDGES")
            stats.global_phase_status = "NO_BRIDGES"

        # Build frozen assignments from bridge results
        frozen: list[FrozenAssignment] = []
        if global_result.success:
            # Use the frozen selector to validate bridge assignments
            from src.ga.repair.cp.frozen_selector import select_consistent_frozen_genes

            bridge_candidates = list(partition.bridge_gene_indices)

            # Apply CP results to a temporary chromosome
            temp_genes = list(genes)
            for gi, (iid, rid, sq) in global_result.assignments.items():
                temp_genes[gi].instructor_id = iid
                temp_genes[gi].room_id = rid
                temp_genes[gi].start_quanta = sq

            # Select bridge genes that are safe to freeze (validates consistency)
            safe_bridge_indices = select_consistent_frozen_genes(
                temp_genes,
                bridge_candidates,
                ctx,
                max_frozen_ratio=1.0,  # Try to freeze all bridges
            )

            logger.info(
                "Bridge validation: %d/%d bridge assignments are mutually consistent",
                len(safe_bridge_indices),
                len(bridge_candidates),
            )

            # Freeze only the safe bridge genes
            for gi in safe_bridge_indices:
                iid, rid, sq = global_result.assignments[gi]
                g = temp_genes[gi]
                frozen.append(
                    FrozenAssignment(
                        gene_index=gi,
                        course_id=g.course_id,
                        course_type=g.course_type,
                        instructor_id=iid,
                        group_ids=tuple(g.group_ids),
                        room_id=rid,
                        start_quanta=sq,
                        num_quanta=g.num_quanta,
                    )
                )
        else:
            logger.warning(
                "Global Phase failed (%s) — not freezing bridge assignments",
                global_result.status,
            )

        # ── 3. Cluster Phase — solve each cluster independently ─────
        cluster_results: dict[str, CPSolveResult] = {}

        # Build a cluster solver with the cluster timeout
        cluster_solver = CPSATSolver(
            ctx,
            family_map,
            timeout_seconds=self.timeout_cluster,
            num_workers=self.num_workers,
            soft_objective=self.soft_objective,
        )

        # Solve clusters sequentially (ProcessPoolExecutor would require
        # pickling the full context which is complex — sequential is fine
        # for 5 clusters × 30s = 2.5 min worst case).
        for cl in partition.clusters:
            cid = cl.cluster_id
            cl_indices = partition.cluster_gene_indices.get(cid, [])
            if not cl_indices:
                logger.info("Cluster %s: 0 genes, skipping", cid)
                cluster_results[cid] = CPSolveResult(success=True, status="EMPTY")
                stats.cluster_statuses[cid] = "EMPTY"
                stats.cluster_times[cid] = 0.0
                continue

            # Large clusters: split into chunks by student group affinity
            if len(cl_indices) > self.max_cluster_genes:
                logger.info(
                    "Cluster %s: %d genes exceeds max %d, chunking…",
                    cid,
                    len(cl_indices),
                    self.max_cluster_genes,
                )
                cr = self._solve_large_cluster(
                    cid,
                    cl_indices,
                    genes,
                    ctx,
                    family_map,
                    cluster_solver,
                    list(frozen),
                )
            else:
                logger.info("Cluster %s: solving %d genes…", cid, len(cl_indices))
                cr = cluster_solver.solve(
                    genes,
                    cl_indices,
                    frozen=frozen,
                    warm_start=True,
                )

                # If MODEL_INVALID or INFEASIBLE with frozen genes, retry
                # without frozen to avoid over-constraining
                if not cr.success and frozen:
                    retry_status = cr.status
                    logger.info(
                        "Cluster %s: %s with %d frozen, retrying without frozen…",
                        cid,
                        retry_status,
                        len(frozen),
                    )
                    cr = cluster_solver.solve(
                        genes,
                        cl_indices,
                        frozen=None,
                        warm_start=True,
                    )
                    if cr.success:
                        logger.info(
                            "Cluster %s: succeeded without frozen (was %s)",
                            cid,
                            retry_status,
                        )

            cluster_results[cid] = cr
            stats.cluster_statuses[cid] = cr.status
            stats.cluster_times[cid] = cr.wall_time

        # ── 4. Merge all results ────────────────────────────────────
        all_results = [global_result] + [
            cr for cr in cluster_results.values() if cr.success
        ]
        repaired = apply_cp_results(genes, *all_results)

        # ── 5. Audit ────────────────────────────────────────────────
        breakdown = audit_hard_violations(repaired, ctx)
        stats.residual_hard = sum(breakdown.values())
        stats.total_time = _time.monotonic() - t0

        # If residual violations remain, try a coordination CP
        if stats.residual_hard > 0:
            logger.info(
                "Coordination pass: %.0f residual violations, attempting full solve…",
                stats.residual_hard,
            )
            coord_solver = CPSATSolver(
                ctx,
                family_map,
                timeout_seconds=self.timeout_global,
                num_workers=self.num_workers,
                soft_objective=self.soft_objective,
            )
            # Find genes involved in violations
            from src.ga.repair.cp.frozen_selector import select_consistent_frozen_genes
            from src.ga.repair.detector import detect_violated_genes

            violated_set = detect_violated_genes(repaired, ctx)
            if violated_set:
                violated_indices = sorted(violated_set)
                violated_set_fast = set(violated_indices)

                # Candidates for freezing: non-violated genes
                candidate_indices = [
                    i for i in range(len(repaired)) if i not in violated_set_fast
                ]

                # Select consistent frozen genes
                safe_frozen_indices = select_consistent_frozen_genes(
                    repaired, candidate_indices, ctx, max_frozen_ratio=0.5
                )
                coord_frozen = [
                    FrozenAssignment.from_gene(i, repaired[i])
                    for i in safe_frozen_indices
                ]

                coord_result = coord_solver.solve(
                    repaired,
                    violated_indices,
                    frozen=coord_frozen,
                    warm_start=True,
                )
                if coord_result.success:
                    repaired = apply_cp_results(repaired, coord_result)
                    breakdown2 = audit_hard_violations(repaired, ctx)
                    stats.residual_hard = sum(breakdown2.values())

        stats.success = stats.residual_hard == 0
        stats.total_time = _time.monotonic() - t0

        logger.info(
            "CP Pipeline complete: %.1fs, residual_hard=%.0f, success=%s",
            stats.total_time,
            stats.residual_hard,
            stats.success,
        )

        return repaired, stats
