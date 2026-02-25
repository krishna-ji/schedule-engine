r"""Universal Feasibility Projection — full three-stage repair pipeline.

Delegates to ``VectorizedRepair.repair_batch`` with configurable
number of passes.  Serves as the *nuclear option* that the RL agent
can select when the population is heavily infeasible.

Complexity: $O(N \cdot E \cdot \text{passes})$.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class UniversalFeasibilityProjection(_AtomicRepairBase):
    """Action 4 — full three-stage vectorized repair pipeline."""

    ACTION_NAME: ClassVar[str] = "universal_feasibility_projection"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        passes: int = 3,
    ):
        super().__init__(pkl_path)
        self.passes = passes

    def _apply(self, X: np.ndarray) -> None:
        result = self.engine.repair_batch(X, passes=self.passes)
        X[:] = result
        logger.debug(
            "UniversalFeasibilityProjection: %d passes", self.passes,
        )
