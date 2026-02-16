"""Frozen Gene Selector: Intelligently select genes to freeze for CP-SAT repair.

When repairing a chromosome with CP-SAT, we want to freeze "good" genes (those
without violations) to reduce the search space. However, freezing genes naively
creates infeasibility:

Problem: Two individually "good" genes can conflict with each other:
    - Both assign instructor X to timeslot T
    - Both assign room Y to timeslot T
    - Gene A requires instructor during unavailable time

Solution: Select frozen genes greedily, validating mutual consistency:
    - Check instructor availability BEFORE freezing
    - Check no instructor exclusivity conflicts
    - Check no room exclusivity conflicts
    - Check no group exclusivity conflicts
    - Check room suitability

This ensures the frozen set is internally consistent, preventing INFEASIBLE status.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext

__all__ = ["select_consistent_frozen_genes"]

logger = logging.getLogger(__name__)


def select_consistent_frozen_genes(
    genes: list[SessionGene],
    candidate_indices: list[int],
    ctx: SchedulingContext,
    max_frozen_ratio: float = 0.5,
) -> list[int]:
    """Select genes to freeze, ensuring mutual consistency.

    Parameters
    ----------
    genes : list[SessionGene]
        Full chromosome.
    candidate_indices : list[int]
        Indices of genes that are candidates for freezing (e.g., non-violated).
    ctx : SchedulingContext
        Scheduling context.
    max_frozen_ratio : float
        Maximum fraction of total genes to freeze (default 0.5).

    Returns
    -------
    list[int]
        Indices of genes safe to freeze (guaranteed mutually consistent).
    """
    from src.utils.room_compatibility import is_room_suitable_for_course

    max_frozen = int(len(genes) * max_frozen_ratio)
    frozen_indices: list[int] = []

    # Track resources used by frozen genes
    used_instructor_slots: set[tuple[str, int]] = set()  # (instructor_id, quantum)
    used_room_slots: set[tuple[str, int]] = set()  # (room_id, quantum)
    used_group_slots: set[tuple[str, int]] = set()  # (group_id, quantum)

    for gene_idx in candidate_indices:
        if len(frozen_indices) >= max_frozen:
            break

        gene = genes[gene_idx]

        # ── Validation 1: Instructor availability (CRITICAL) ──
        instructor = ctx.instructors.get(gene.instructor_id)
        if not instructor:
            continue  # Invalid instructor, don't freeze

        if not instructor.is_full_time:
            # Part-time: check ALL quanta are available
            occupied_quanta = range(
                gene.start_quanta, gene.start_quanta + gene.num_quanta
            )
            if not all(q in instructor.available_quanta for q in occupied_quanta):
                logger.debug(
                    "Skipping gene %d: instructor %s unavailable at t=%d",
                    gene_idx,
                    gene.instructor_id,
                    gene.start_quanta,
                )
                continue  # Instructor unavailable, NEVER freeze

        # ── Validation 2: Room suitability ──
        course = ctx.courses.get((gene.course_id, gene.course_type))
        room = ctx.rooms.get(gene.room_id)
        if course and room:
            required_feat = (
                str(getattr(course, "required_room_features", "lecture"))
                .lower()
                .strip()
            )
            room_feat = str(getattr(room, "room_features", "lecture")).lower().strip()
            course_lab = getattr(course, "specific_lab_features", None)
            room_spec = getattr(room, "specific_features", None)

            if not is_room_suitable_for_course(
                required_feat, room_feat, course_lab, room_spec
            ):
                logger.debug(
                    "Skipping gene %d: room %s unsuitable for course %s-%s",
                    gene_idx,
                    gene.room_id,
                    gene.course_id,
                    gene.course_type,
                )
                continue  # Room unsuitable, don't freeze

        # ── Validation 3: No conflicts with already-frozen genes ──
        occupied_quanta = range(gene.start_quanta, gene.start_quanta + gene.num_quanta)

        # Check instructor exclusivity
        inst_keys = {(gene.instructor_id, q) for q in occupied_quanta}
        if inst_keys & used_instructor_slots:
            logger.debug(
                "Skipping gene %d: instructor %s conflict at frozen slots",
                gene_idx,
                gene.instructor_id,
            )
            continue  # Would create instructor conflict

        # Check room exclusivity
        room_keys = {(gene.room_id, q) for q in occupied_quanta}
        if room_keys & used_room_slots:
            logger.debug(
                "Skipping gene %d: room %s conflict at frozen slots",
                gene_idx,
                gene.room_id,
            )
            continue  # Would create room conflict

        # Check group exclusivity
        group_keys = {(gid, q) for gid in gene.group_ids for q in occupied_quanta}
        if group_keys & used_group_slots:
            logger.debug(
                "Skipping gene %d: group conflict at frozen slots",
                gene_idx,
            )
            continue  # Would create group conflict

        # ── SAFE TO FREEZE ──
        frozen_indices.append(gene_idx)
        used_instructor_slots.update(inst_keys)
        used_room_slots.update(room_keys)
        used_group_slots.update(group_keys)

    logger.info(
        "Selected %d/%d candidate genes to freeze (%.1f%% of total)",
        len(frozen_indices),
        len(candidate_indices),
        100.0 * len(frozen_indices) / len(genes) if genes else 0,
    )

    return frozen_indices
