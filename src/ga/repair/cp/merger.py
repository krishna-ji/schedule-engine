"""Solution Merger: Reassemble a full chromosome from solved subproblems.

After the Global Phase and Cluster Phase each produce assignments for their
gene subsets, this module applies those assignments back onto the original
chromosome, then runs a conflict audit to detect any residual violations.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext
    from src.ga.repair.cp.solver import CPSolveResult

__all__ = ["apply_cp_results", "audit_hard_violations"]

logger = logging.getLogger(__name__)


def apply_cp_results(
    genes: list[SessionGene],
    *results: CPSolveResult,
) -> list[SessionGene]:
    """Apply CP-SAT assignments to a (deep-copied) gene list.

    For each result, every ``gene_index`` in ``result.assignments`` gets its
    ``instructor_id``, ``room_id``, and ``start_quanta`` updated.

    Parameters
    ----------
    genes : list[SessionGene]
        The original chromosome (will NOT be modified).
    *results : CPSolveResult
        One or more solve results whose assignments should be merged.

    Returns
    -------
    list[SessionGene]
        A new gene list with the CP-SAT assignments applied.
    """
    new_genes = copy.deepcopy(genes)
    applied = 0

    for result in results:
        if not result.success:
            continue
        for gi, (instr_id, room_id, start_q) in result.assignments.items():
            g = new_genes[gi]
            g.instructor_id = instr_id
            g.room_id = room_id
            g.start_quanta = start_q
            applied += 1

    logger.info(
        "Merger: applied %d gene assignments from %d results", applied, len(results)
    )
    return new_genes


def audit_hard_violations(
    genes: list[SessionGene],
    ctx: SchedulingContext,
) -> dict[str, float]:
    """Run the evaluator on *genes* and return per-hard-constraint penalties.

    Returns
    -------
    dict[str, float]
        ``{constraint_name: weighted_penalty}``.  A fully feasible schedule
        has all values at 0.
    """
    from src.constraints.evaluator import Evaluator
    from src.domain.timetable import Timetable

    tt = Timetable(genes=genes, context=ctx)
    ev = Evaluator()
    breakdown = ev.hard_breakdown(tt)
    total = sum(breakdown.values())

    if total > 0:
        logger.warning(
            "Audit: %.0f residual hard violations after CP repair — %s",
            total,
            {k: v for k, v in breakdown.items() if v > 0},
        )
    else:
        logger.info("Audit: 0 hard violations — fully feasible!")

    return breakdown
