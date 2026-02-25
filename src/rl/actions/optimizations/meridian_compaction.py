r"""Meridian Compaction Heuristic — soft-constraint optimisation.

Targets two soft constraints simultaneously:

* **MIP** (Meridian Interval Preservation) — events scheduled during
  the lunch window (within-day quanta $\{2, 3, 4\}$) are shifted to
  adjacent non-lunch slots when a valid alternative exists.
* **CSC** (Cohort Schedule Contiguity) — by compacting events towards
  the earliest feasible slot, schedule gaps are reduced.

The operator works in two vectorized passes:

1. **Lunch-window evacuation**: for each event whose time quantum
   falls inside the lunch window, attempt to reassign it to the
   nearest valid time outside the window.
2. **Gap compaction**: for ~10 % of remaining events, shift towards
   the earliest available slot in their domain to reduce CSC gaps.

Salvaged concept: gap-filling logic from legacy ``construction.py``
(earliest-slot-first assignment), translated to batched domain sampling.

Complexity: $O(N \cdot E)$.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)

# Time system constants (must match soft_evaluator_vectorized.py)
_QUANTA_PER_DAY = 7
_LUNCH_WITHIN_DAY = np.array([2, 3, 4], dtype=np.int64)  # within-day indices


class MeridianCompactionHeuristic(_AtomicRepairBase):
    """Action 7 — soft-constraint optimisation targeting MIP and CSC."""

    ACTION_NAME: ClassVar[str] = "meridian_compaction"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        compaction_rate: float = 0.10,
    ):
        super().__init__(pkl_path)
        self.compaction_rate = compaction_rate

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        rng = np.random.default_rng()

        # Current time slots — shape (N, E)
        time = X[:, 2::3].astype(np.int64)

        # ── Pass 1: Lunch-window evacuation ────────────────────────
        # Compute within-day quantum: q_wd = time_slot % QUANTA_PER_DAY
        within_day = time % _QUANTA_PER_DAY  # (N, E)

        # Boolean mask: is this event in the lunch window?
        in_lunch = np.isin(within_day, _LUNCH_WITHIN_DAY)  # (N, E)

        if in_lunch.any():
            bi, be = np.nonzero(in_lunch)

            # For each conflicting event, find a domain time slot
            # that is NOT in the lunch window
            t_dl = eng.time_dom_len[be]
            t_valid = t_dl > 1  # need at least 2 options to relocate
            if t_valid.any():
                t_bi = bi[t_valid]
                t_be = be[t_valid]
                t_dl_v = t_dl[t_valid]

                # Sample a random domain slot and check if it's outside lunch
                # Attempt up to 3 tries to find a non-lunch slot
                for _attempt in range(3):
                    t_idx = (rng.random(len(t_bi)) * t_dl_v).astype(np.int64)
                    t_idx = np.minimum(t_idx, t_dl_v - 1)
                    candidates = eng.time_domains[t_be, t_idx]
                    cand_wd = candidates % _QUANTA_PER_DAY
                    outside = ~np.isin(cand_wd, _LUNCH_WITHIN_DAY)

                    if outside.any():
                        ok_bi = t_bi[outside]
                        ok_be = t_be[outside]
                        ok_val = candidates[outside]
                        X[ok_bi, 3 * ok_be + 2] = ok_val

                        # Remove successfully moved from retry set
                        still_inside = ~outside
                        t_bi = t_bi[still_inside]
                        t_be = t_be[still_inside]
                        t_dl_v = t_dl_v[still_inside]

                    if len(t_bi) == 0:
                        break

            n_evacuated = int(in_lunch.sum()) - len(t_bi) if t_valid.any() else 0
            logger.debug(
                "MeridianCompaction pass 1: %d/%d events evacuated from lunch",
                n_evacuated,
                int(in_lunch.sum()),
            )

        # ── Pass 2: Gap compaction (shift towards earliest slot) ───
        compact_mask = rng.random((N, E)) < self.compaction_rate
        if compact_mask.any():
            c_bi, c_be = np.nonzero(compact_mask)

            # For selected events, assign the EARLIEST domain time slot
            # (index 0 in sorted domain) to compact the schedule
            t_dl = eng.time_dom_len[c_be]
            t_v = t_dl > 0
            if t_v.any():
                c_bi_v = c_bi[t_v]
                c_be_v = c_be[t_v]
                # Domain arrays are typically sorted ascending, so index 0
                # gives the earliest feasible slot
                X[c_bi_v, 3 * c_be_v + 2] = eng.time_domains[c_be_v, 0]

            logger.debug(
                "MeridianCompaction pass 2: %d/%d events compacted (rate=%.2f)",
                int(compact_mask.sum()),
                N * E,
                self.compaction_rate,
            )
