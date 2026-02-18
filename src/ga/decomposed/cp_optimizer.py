"""CP-SAT Optimizer for Decomposed Scheduling.

Provides per-cluster CP-SAT solving that integrates with the GA evolution loop.
This replaces the old repair-based approach with a proper optimization pass.

**Key insight — Iterative Chunked Solving:**

Large clusters (>50 violated genes) are INFEASIBLE in one CP-SAT shot because
the combinatorial explosion of room × instructor × timeslot variables is too
large.  Instead, we group violated genes into *semester-based chunks* of 30-50
genes and solve each chunk sequentially, freezing successful results between
iterations.  This converts one INFEASIBLE 300-gene problem into 6-10 small
FEASIBLE problems.

Two modes:
    1. **Fix Mode** — Fix hard constraint violations only (fast, ~5-15s total).
    2. **Optimize Mode** — Fix hard + minimize soft penalties (slower).
"""

from __future__ import annotations

import copy
import logging
import re
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.ga.repair.cp.frozen_selector import select_consistent_frozen_genes
from src.ga.repair.cp.merger import audit_hard_violations
from src.ga.repair.cp.solver import CPSATSolver, CPSolveResult, FrozenAssignment
from src.ga.repair.detector import detect_violated_genes

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext
    from src.ga.decomposed.cluster_context import ClusterContext, PartitionResult

__all__ = ["CPOptimizeResult", "ClusterCPOptimizer"]

logger = logging.getLogger(__name__)

# ── Semester extraction ───────────────────────────────────────────────────

_SEM_RE = re.compile(r"^[A-Z]{2,5}(\d+)")


def _semester_key(group_id: str) -> str:
    """Extract semester number from group_id for chunking.

    >>> _semester_key("BCE3AB")
    '3'
    >>> _semester_key("BCT1A")
    '1'
    """
    m = _SEM_RE.match(group_id)
    return m.group(1) if m else "0"


def _programme_key(group_id: str) -> str:
    """Extract programme prefix from group_id."""
    m = re.match(r"^([A-Z]{2,5})", group_id)
    return m.group(1) if m else group_id


# Max genes in a single CP-SAT chunk before we split further.
_MAX_CHUNK_SIZE = 50


@dataclass
class CPOptimizeResult:
    """Result of a CP optimization pass.

    Attributes
    ----------
    genes : list[SessionGene]
        The optimized gene list.
    hard_before : float
        Total hard violations before optimization.
    hard_after : float
        Total hard violations after optimization.
    soft_before : float
        Total soft penalty before optimization (if measured).
    soft_after : float
        Total soft penalty after optimization (if measured).
    cluster_stats : dict[str, dict]
        Per-cluster solve statistics.
    bridge_stats : dict
        Bridge solve statistics.
    total_time : float
        Total wall time for the optimization pass.
    """

    genes: list[SessionGene] = field(default_factory=list)
    hard_before: float = 0.0
    hard_after: float = 0.0
    soft_before: float = 0.0
    soft_after: float = 0.0
    cluster_stats: dict[str, dict] = field(default_factory=dict)
    bridge_stats: dict = field(default_factory=dict)
    total_time: float = 0.0


class ClusterCPOptimizer:
    """CP-SAT optimizer that works on cluster subproblems.

    Uses **iterative chunked solving** for large clusters: violated genes
    are grouped by (programme, semester), solved in small batches (~30-50),
    and each batch's results are frozen before the next batch.  This makes
    even 300+ gene clusters tractable.

    Parameters
    ----------
    timeout_bridge : float
        Max seconds for bridge gene CP-SAT solve.
    timeout_cluster : float
        Max seconds per cluster CP-SAT solve.
    timeout_chunk : float
        Max seconds per chunk solve within a large cluster.
    num_workers : int
        CP-SAT internal parallelism.
    soft_objective : bool
        Whether to optimize soft constraints too.
    violation_threshold : int
        Minimum hard violations to trigger CP optimization.
    max_chunk_size : int
        Maximum genes in a single CP-SAT chunk.
    """

    def __init__(
        self,
        *,
        timeout_bridge: float = 30.0,
        timeout_cluster: float = 15.0,
        timeout_chunk: float = 15.0,
        num_workers: int = 4,
        soft_objective: bool = False,
        violation_threshold: int = 0,
        max_chunk_size: int = _MAX_CHUNK_SIZE,
    ) -> None:
        self.timeout_bridge = timeout_bridge
        self.timeout_cluster = timeout_cluster
        self.timeout_chunk = timeout_chunk
        self.num_workers = num_workers
        self.soft_objective = soft_objective
        self.violation_threshold = violation_threshold
        self.max_chunk_size = max_chunk_size

    # ── Main entry point ──────────────────────────────────────────────

    def optimize_individual(
        self,
        genes: list[SessionGene],
        ctx: SchedulingContext,
        partition: PartitionResult,
        *,
        fix_only: bool = True,
    ) -> CPOptimizeResult:
        """Run CP-SAT optimization on a single individual.

        Parameters
        ----------
        genes : list[SessionGene]
            The chromosome to optimize (not modified in-place).
        ctx : SchedulingContext
            Full scheduling context.
        partition : PartitionResult
            Pre-computed cluster partition info.
        fix_only : bool
            If True, only fix violated genes (fast).
            If False, also optimize soft objectives (slower).

        Returns
        -------
        CPOptimizeResult
        """
        from src.ga.decomposed.cluster_context import partition_individual

        t0 = _time.monotonic()
        result = CPOptimizeResult()

        # Measure hard violations before
        hard_breakdown = audit_hard_violations(genes, ctx)
        result.hard_before = sum(hard_breakdown.values())

        # Skip if below threshold
        if result.hard_before <= self.violation_threshold:
            result.genes = list(genes)
            result.hard_after = result.hard_before
            result.total_time = _time.monotonic() - t0
            return result

        # Re-partition genes (assigns gene indices to clusters)
        partition_individual(genes, partition)

        family_map = ctx.family_map
        working = [copy.deepcopy(g) for g in genes]

        # ── Phase 1: Bridge Genes ─────────────────────────────────
        frozen_assignments: list[FrozenAssignment] = []

        if partition.bridge_gene_indices:
            bridge_result = self._solve_bridge(
                working, partition.bridge_gene_indices, ctx, family_map
            )
            result.bridge_stats = {
                "status": bridge_result.status,
                "time": bridge_result.wall_time,
                "solved": len(bridge_result.assignments),
            }

            if bridge_result.success:
                for gi, (iid, rid, sq) in bridge_result.assignments.items():
                    working[gi].instructor_id = iid
                    working[gi].room_id = rid
                    working[gi].start_quanta = sq

                safe_indices = select_consistent_frozen_genes(
                    working,
                    list(partition.bridge_gene_indices),
                    ctx,
                    max_frozen_ratio=1.0,
                )
                for gi in safe_indices:
                    frozen_assignments.append(
                        FrozenAssignment.from_gene(gi, working[gi])
                    )

        # ── Phase 2: Per-Cluster (chunked for large clusters) ─────
        for cid, cc in partition.cluster_contexts.items():
            if not cc.global_gene_indices:
                result.cluster_stats[cid] = {"status": "EMPTY", "time": 0.0}
                continue

            cluster_t0 = _time.monotonic()

            if fix_only:
                violated_set = detect_violated_genes(working, ctx)
                solve_indices = [i for i in cc.global_gene_indices if i in violated_set]
                if not solve_indices:
                    result.cluster_stats[cid] = {
                        "status": "NO_VIOLATIONS",
                        "time": 0.0,
                    }
                    continue
            else:
                solve_indices = list(cc.global_gene_indices)

            # Decide: direct solve vs chunked
            if len(solve_indices) <= self.max_chunk_size:
                # Small enough — single CP-SAT solve
                cr = self._solve_cluster(
                    working, solve_indices, ctx, family_map, frozen_assignments
                )
                if cr.success:
                    for gi, (iid, rid, sq) in cr.assignments.items():
                        working[gi].instructor_id = iid
                        working[gi].room_id = rid
                        working[gi].start_quanta = sq
                result.cluster_stats[cid] = {
                    "status": cr.status,
                    "time": cr.wall_time,
                    "solved": len(cr.assignments),
                }
            else:
                # Large cluster — chunked iterative solving
                chunk_stats = self._solve_chunked(
                    working,
                    solve_indices,
                    cc.global_gene_indices,
                    ctx,
                    family_map,
                    frozen_assignments,
                )
                result.cluster_stats[cid] = {
                    "status": "CHUNKED",
                    "time": _time.monotonic() - cluster_t0,
                    **chunk_stats,
                }

        # ── Measure after ─────────────────────────────────────────
        hard_after = audit_hard_violations(working, ctx)
        result.hard_after = sum(hard_after.values())
        result.genes = working
        result.total_time = _time.monotonic() - t0

        logger.info(
            "CP optimize: hard %.0f → %.0f (Δ%.0f) in %.1fs | clusters: %s",
            result.hard_before,
            result.hard_after,
            result.hard_before - result.hard_after,
            result.total_time,
            {cid: s.get("status", "?") for cid, s in result.cluster_stats.items()},
        )

        return result

    # ── Chunked solving for large clusters ────────────────────────────

    def _solve_chunked(
        self,
        working: list[SessionGene],
        violated_indices: list[int],
        all_cluster_indices: list[int],
        ctx: SchedulingContext,
        family_map: dict[str, set[str]],
        initial_frozen: list[FrozenAssignment],
    ) -> dict:
        """Solve large clusters with sequential chunk freezing + cascade retry.

        Strategy:
        1. Group violated genes by (programme, semester), sorted by semester.
        2. Solve each chunk sequentially, freezing successful results for
           subsequent chunks (provides context, avoids creating new conflicts).
        3. If a chunk is INFEASIBLE, cascade retry:
           a) Retry without frozen constraints (accept cross-chunk conflicts)
           b) Retry with relaxed HC6 (soft instructor availability)
           This handles both over-constraining from frozen genes AND
           structural infeasibility from part-time instructor bottlenecks.

        Returns summary statistics dict.
        """
        # Build chunks grouped by (programme, semester)
        chunks = self._build_chunks(working, violated_indices)

        # Start with initial frozen + non-violated cluster genes as context
        frozen = list(initial_frozen)
        violated_set = set(violated_indices)
        frozen_gene_set = {fa.gene_index for fa in frozen}

        # Freeze non-violated cluster genes as context
        for gi in all_cluster_indices:
            if gi not in violated_set and gi not in frozen_gene_set:
                frozen.append(FrozenAssignment.from_gene(gi, working[gi]))
                frozen_gene_set.add(gi)

        total_solved = 0
        chunk_results: list[dict] = []

        for chunk_label, chunk_indices in chunks:
            if not chunk_indices:
                continue

            cr = self._solve_chunk(working, chunk_indices, ctx, family_map, frozen)

            chunk_info = {
                "label": chunk_label,
                "genes": len(chunk_indices),
                "status": cr.status,
                "time": cr.wall_time,
                "solved": len(cr.assignments),
            }

            if cr.success:
                self._apply_and_freeze(
                    working,
                    cr,
                    frozen,
                    frozen_gene_set,
                    ctx,
                    chunk_label,
                    chunk_indices,
                )
                total_solved += len(cr.assignments)
            else:
                # CASCADE RETRY 1: Keep frozen context + relax HC6.
                # The infeasibility is typically from HC6 (part-time
                # instructor availability), NOT from frozen conflicts.
                # Keeping frozen preserves cross-chunk consistency and
                # prevents room/instructor exclusivity violations.
                cr2 = self._solve_chunk_relaxed(
                    working, chunk_indices, ctx, family_map, frozen
                )
                if cr2.success:
                    # Solution respects frozen context (no new exclusivity
                    # violations), safe to freeze for subsequent chunks.
                    self._apply_and_freeze(
                        working,
                        cr2,
                        frozen,
                        frozen_gene_set,
                        ctx,
                        chunk_label,
                        chunk_indices,
                        status_label=f"RELAXED-{cr2.status}",
                    )
                    total_solved += len(cr2.assignments)
                    chunk_info["status"] = f"RELAXED-{cr2.status}"
                    chunk_info["solved"] = len(cr2.assignments)
                else:
                    # CASCADE RETRY 2: Last resort — no frozen + relaxed HC6
                    cr3 = self._solve_chunk_relaxed(
                        working, chunk_indices, ctx, family_map, []
                    )
                    if cr3.success:
                        # No frozen context: may conflict with other chunks.
                        # Apply only (do NOT freeze).
                        self._apply_results(working, cr3)
                        total_solved += len(cr3.assignments)
                        chunk_info["status"] = f"UNFROZEN-RELAXED-{cr3.status}"
                        chunk_info["solved"] = len(cr3.assignments)
                        logger.info(
                            "  Chunk %s: UNFROZEN-RELAXED-%s "
                            "(%d genes → %d, NOT frozen)",
                            chunk_label,
                            cr3.status,
                            len(chunk_indices),
                            len(cr3.assignments),
                        )
                    else:
                        logger.warning(
                            "  Chunk %s: INFEASIBLE all retries " "(%d genes, %.1fs)",
                            chunk_label,
                            len(chunk_indices),
                            cr.wall_time + cr2.wall_time + cr3.wall_time,
                        )

            chunk_results.append(chunk_info)

        return {
            "chunks": len(chunk_results),
            "total_solved": total_solved,
            "total_violated": len(violated_indices),
            "chunk_details": chunk_results,
        }

    def _apply_results(
        self,
        working: list[SessionGene],
        cr: CPSolveResult,
    ) -> None:
        """Apply solve results WITHOUT freezing (for unfrozen retry)."""
        for gi, (iid, rid, sq) in cr.assignments.items():
            working[gi].instructor_id = iid
            working[gi].room_id = rid
            working[gi].start_quanta = sq

    def _apply_and_freeze(
        self,
        working: list[SessionGene],
        cr: CPSolveResult,
        frozen: list[FrozenAssignment],
        frozen_gene_set: set[int],
        ctx: SchedulingContext,
        chunk_label: str,
        chunk_indices: list[int],
        status_label: str | None = None,
    ) -> None:
        """Apply solve results and freeze for subsequent chunks."""
        for gi, (iid, rid, sq) in cr.assignments.items():
            working[gi].instructor_id = iid
            working[gi].room_id = rid
            working[gi].start_quanta = sq

        safe = select_consistent_frozen_genes(
            working,
            list(cr.assignments.keys()),
            ctx,
            max_frozen_ratio=1.0,
        )
        for gi in safe:
            if gi not in frozen_gene_set:
                frozen.append(FrozenAssignment.from_gene(gi, working[gi]))
                frozen_gene_set.add(gi)

        display_status = status_label or cr.status
        logger.info(
            "  Chunk %s: %s (%d genes → %d solved, frozen=%d)",
            chunk_label,
            display_status,
            len(chunk_indices),
            len(cr.assignments),
            len(frozen),
        )

    def _build_chunks(
        self,
        genes: list[SessionGene],
        indices: list[int],
    ) -> list[tuple[str, list[int]]]:
        """Group gene indices by (programme, semester) for chunked solving.

        Returns list of (label, indices) sorted by semester then SIZE
        (ascending).  Small/hard chunks solve first with less frozen
        context, reducing infeasibility from resource contention.
        Each chunk is capped at max_chunk_size.  Tiny chunks (<5 genes)
        are merged with the next chunk in the same semester.
        """
        buckets: dict[tuple[str, str], list[int]] = defaultdict(list)

        for gi in indices:
            g = genes[gi]
            if g.group_ids:
                gid = g.group_ids[0]
                prog = _programme_key(gid)
                sem = _semester_key(gid)
            else:
                prog, sem = "UNK", "0"
            buckets[(sem, prog)].append(gi)

        # Sort by semester first, then by bucket SIZE ascending
        sorted_keys = sorted(
            buckets.keys(),
            key=lambda sp: (sp[0], len(buckets[sp]), sp[1]),
        )

        result: list[tuple[str, list[int]]] = []
        for sem, prog in sorted_keys:
            bucket = buckets[(sem, prog)]
            label = f"{prog}-sem{sem}"
            if len(bucket) <= self.max_chunk_size:
                result.append((label, bucket))
            else:
                for k in range(0, len(bucket), self.max_chunk_size):
                    sub = bucket[k : k + self.max_chunk_size]
                    part = k // self.max_chunk_size + 1
                    result.append((f"{label}-p{part}", sub))

        # Merge tiny chunks (< 5 genes) with the next same-semester chunk
        merged: list[tuple[str, list[int]]] = []
        for label, idx_list in result:
            if merged and len(merged[-1][1]) < 10:
                prev_label, prev_list = merged[-1]
                merged[-1] = (f"{prev_label}+{label}", prev_list + idx_list)
            else:
                merged.append((label, idx_list))

        return merged

    # ── Solver wrappers ───────────────────────────────────────────────

    def _solve_bridge(
        self,
        genes: list[SessionGene],
        bridge_indices: list[int],
        ctx: SchedulingContext,
        family_map: dict[str, set[str]],
    ) -> CPSolveResult:
        """Solve bridge genes (cross-cluster shared resources)."""
        solver = CPSATSolver(
            ctx,
            family_map,
            timeout_seconds=self.timeout_bridge,
            num_workers=self.num_workers,
            soft_objective=self.soft_objective,
        )
        return solver.solve(
            genes,
            bridge_indices,
            frozen=None,
            warm_start=True,
        )

    def _solve_cluster(
        self,
        genes: list[SessionGene],
        cluster_indices: list[int],
        ctx: SchedulingContext,
        family_map: dict[str, set[str]],
        frozen: list[FrozenAssignment],
    ) -> CPSolveResult:
        """Solve a cluster's genes with bridge assignments frozen."""
        solver = CPSATSolver(
            ctx,
            family_map,
            timeout_seconds=self.timeout_cluster,
            num_workers=self.num_workers,
            soft_objective=self.soft_objective,
        )
        return solver.solve(
            genes,
            cluster_indices,
            frozen=frozen if frozen else None,
            warm_start=True,
        )

    def _solve_chunk(
        self,
        genes: list[SessionGene],
        chunk_indices: list[int],
        ctx: SchedulingContext,
        family_map: dict[str, set[str]],
        frozen: list[FrozenAssignment],
    ) -> CPSolveResult:
        """Solve a single chunk of genes within a large cluster."""
        solver = CPSATSolver(
            ctx,
            family_map,
            timeout_seconds=self.timeout_chunk,
            num_workers=self.num_workers,
            soft_objective=self.soft_objective,
        )
        return solver.solve(
            genes,
            chunk_indices,
            frozen=frozen if frozen else None,
            warm_start=True,
        )

    def _solve_chunk_relaxed(
        self,
        genes: list[SessionGene],
        chunk_indices: list[int],
        ctx: SchedulingContext,
        family_map: dict[str, set[str]],
        frozen: list[FrozenAssignment],
    ) -> CPSolveResult:
        """Solve a chunk with relaxed HC6 (soft instructor availability).

        Used as last-resort retry when the chunk is INFEASIBLE even
        without frozen constraints — typically because part-time
        instructors have availability too limited for their courses.
        """
        solver = CPSATSolver(
            ctx,
            family_map,
            timeout_seconds=self.timeout_chunk,
            num_workers=self.num_workers,
            soft_objective=self.soft_objective,
            relax_availability=True,
        )
        return solver.solve(
            genes,
            chunk_indices,
            frozen=frozen if frozen else None,
            warm_start=True,
        )
