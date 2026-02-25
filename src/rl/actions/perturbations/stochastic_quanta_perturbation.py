r"""Stochastic Quanta Perturbation — random time-slot mutation.

Randomly selects $\sim 5\%$ of $(N, E)$ event assignments and
resamples their **time quantum** from $\mathcal{D}_e^{\text{time}}$.
Acts as exploration pressure targeting temporal diversity —
no conflict detection is performed.

Salvaged concept: ``temporal_shift`` from legacy perturbation.py,
translated to batched masked sampling on pre-computed domain arrays.

Complexity: $O(N \cdot E \cdot \rho)$ where $\rho$ = perturbation rate.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class StochasticQuantaPerturbation(_AtomicRepairBase):
    """Action 5 — randomly perturb time assignments for ~5 % of events."""

    ACTION_NAME: ClassVar[str] = "stochastic_quanta_perturbation"

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

        # Resample time only
        t_dl = eng.time_dom_len[be]
        t_v = t_dl > 0
        if t_v.any():
            t_bi, t_be, t_dl_v = bi[t_v], be[t_v], t_dl[t_v]
            t_idx = (rng.random(len(t_bi)) * t_dl_v).astype(np.int64)
            t_idx = np.minimum(t_idx, t_dl_v - 1)
            X[t_bi, 3 * t_be + 2] = eng.time_domains[t_be, t_idx]

        logger.debug(
            "StochasticQuantaPerturbation: %d/%d time slots perturbed (rate=%.2f)",
            int(mask.sum()), N * E, self.perturb_rate,
        )
