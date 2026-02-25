r"""Symmetric Sub-Cohort Synchronisation — enforce SSCP parallelism.

For each paired practical $(a, b)$, forces
$t_a = t_b \in \mathcal{T}_a \cap \mathcal{T}_b$ and $r_a \neq r_b$.

Delegates to the engine's ``_sync_paired_events`` after domain fixing.

Complexity: $O(N \cdot P)$ where $P$ = number of paired practicals.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class SymmetricSubcohortSync(_AtomicRepairBase):
    """Action 3 — synchronise paired practicals for SSCP."""

    ACTION_NAME: ClassVar[str] = "symmetric_subcohort_sync"

    def _apply(self, X) -> None:
        eng = self.engine
        eng._fix_domains_vec(X)
        if eng._n_pairs > 0:
            eng._sync_paired_events(X)
        logger.debug(
            "SymmetricSubcohortSync: %d pairs synced",
            eng._n_pairs,
        )
