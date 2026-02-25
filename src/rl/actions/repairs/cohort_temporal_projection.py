r"""Cohort Temporal Projection — resolve group double-booking (CTE).

Detects group double-booking via ``np.bincount`` on linearised
group×time keys.  Conflicting events get their **time** resampled;
~30 % also get rooms resampled for structural diversity.

Complexity: $O(N \cdot Q_g)$ where $Q_g$ = group-expanded quanta length.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class CohortTemporalProjection(_AtomicRepairBase):
    """Action 2 — project population onto CTE feasibility surface."""

    ACTION_NAME: ClassVar[str] = "cohort_temporal_projection"

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        T_ = __import__("src.pipeline.bitset_time", fromlist=["T"]).T

        eng._fix_domains_vec(X)

        time = X[:, 2::3].astype(np.int64)
        n_idx = np.arange(N, dtype=np.int64)[:, None]

        grp_starts = time[:, eng.grp_exp_event]
        grp_quanta = np.clip(grp_starts + eng.grp_exp_offset[None, :], 0, T_ - 1)

        nGT = np.int64(eng.n_groups) * np.int64(T_)
        grp_keys = (
            n_idx * nGT + eng.grp_exp_group[None, :].astype(np.int64) * T_ + grp_quanta
        ).ravel()
        grp_cnt = np.bincount(grp_keys, minlength=int(N * nGT))
        grp_conflict = (grp_cnt[grp_keys] > 1).astype(np.float64)

        grp_event_lin = (n_idx * E + eng.grp_exp_event[None, :]).ravel()
        NE = N * E
        scores = np.bincount(grp_event_lin, weights=grp_conflict, minlength=NE)
        scores = scores[:NE].reshape(N, E)

        conflict_mask = scores > 0
        if not conflict_mask.any():
            return

        rng = np.random.default_rng()
        bi, be = np.nonzero(conflict_mask)

        # Resample time
        t_dl = eng.time_dom_len[be]
        t_valid = t_dl > 0
        t_bi, t_be, t_dl_v = bi[t_valid], be[t_valid], t_dl[t_valid]
        t_idx = (rng.random(len(t_bi)) * t_dl_v).astype(np.int64)
        t_idx = np.minimum(t_idx, t_dl_v - 1)
        X[t_bi, 3 * t_be + 2] = eng.time_domains[t_be, t_idx]

        # Resample room for ~30 % to add structural diversity
        do_room = rng.random(len(bi)) < 0.3
        r_bi, r_be = bi[do_room], be[do_room]
        r_dl = eng.room_dom_len[r_be]
        r_v = r_dl > 0
        r_bi, r_be, r_dl_v = r_bi[r_v], r_be[r_v], r_dl[r_v]
        r_idx = (rng.random(len(r_bi)) * r_dl_v).astype(np.int64)
        r_idx = np.minimum(r_idx, r_dl_v - 1)
        X[r_bi, 3 * r_be + 1] = eng.room_domains[r_be, r_idx]

        logger.debug(
            "CohortTemporalProjection: %d/%d events repaired",
            int(conflict_mask.sum()),
            N * E,
        )
