r"""Stochastic Spatial Perturbation — random room-swap mutation.

Randomly selects $\sim 5\%$ of $(N, E)$ event assignments and
resamples their **room** from $\mathcal{D}_e^{\text{room}}$.
Pure exploration pressure targeting spatial diversity — no conflict
detection is performed.

Salvaged concept: ``room_shuffle`` from legacy perturbation.py,
translated to batched masked sampling on pre-computed domain arrays.

Complexity: $O(N \cdot E \cdot \rho)$ where $\rho$ = perturbation rate.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class StochasticSpatialPerturbation(_AtomicRepairBase):
    """Action 6 — randomly perturb room assignments for ~5 % of events."""

    ACTION_NAME: ClassVar[str] = "stochastic_spatial_perturbation"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        perturb_rate: float = 0.05,
    ):
        super().__init__(pkl_path)
        self.perturb_rate = perturb_rate

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        rng = np.random.default_rng()

        # Select ~perturb_rate of (individual, event) pairs
        mask = rng.random((N, E)) < self.perturb_rate
        if not mask.any():
            return

        bi, be = np.nonzero(mask)

        # Resample room only
        r_dl = eng.room_dom_len[be]
        r_v = r_dl > 0
        if r_v.any():
            r_bi, r_be, r_dl_v = bi[r_v], be[r_v], r_dl[r_v]
            r_idx = (rng.random(len(r_bi)) * r_dl_v).astype(np.int64)
            r_idx = np.minimum(r_idx, r_dl_v - 1)
            X[r_bi, 3 * r_be + 1] = eng.room_domains[r_be, r_idx]

        logger.debug(
            "StochasticSpatialPerturbation: %d/%d rooms perturbed (rate=%.2f)",
            int(mask.sum()), N * E, self.perturb_rate,
        )
