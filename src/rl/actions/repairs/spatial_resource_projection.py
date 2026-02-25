r"""Spatial Resource Projection — resolve room double-booking (SRE).

Detects room double-booking via ``np.bincount`` on linearised
room×time keys $k = n \cdot R \cdot T + r \cdot T + q$.
Conflicting events get their **room** resampled from
$\mathcal{D}_e^{\text{room}}$; ~40 % also get their time resampled
to break structural overlaps.

Complexity: $O(N \cdot Q)$ where $Q$ = total quanta expansion length.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class SpatialResourceProjection(_AtomicRepairBase):
    """Action 0 — project population onto SRE feasibility surface."""

    ACTION_NAME: ClassVar[str] = "spatial_resource_projection"

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        T_ = __import__("src.pipeline.bitset_time", fromlist=["T"]).T

        eng._fix_domains_vec(X)

        room = np.clip(X[:, 1::3], 0, eng.n_rooms - 1).astype(np.int64)
        time = X[:, 2::3].astype(np.int64)
        n_idx = np.arange(N, dtype=np.int64)[:, None]

        starts_exp = time[:, eng.exp_event]
        quanta_exp = np.clip(starts_exp + eng.exp_offset[None, :], 0, T_ - 1)
        rooms_exp = room[:, eng.exp_event]
        event_lin = (n_idx * E + eng.exp_event[None, :]).ravel()
        NE = N * E

        nRT = np.int64(eng.n_rooms) * np.int64(T_)
        room_keys = (n_idx * nRT + rooms_exp * T_ + quanta_exp).ravel()
        room_cnt = np.bincount(room_keys, minlength=int(N * nRT))
        room_conflict = (room_cnt[room_keys] > 1).astype(np.float64)

        scores = np.bincount(event_lin, weights=room_conflict, minlength=NE)
        scores = scores[:NE].reshape(N, E)

        conflict_mask = scores > 0
        if not conflict_mask.any():
            return

        rng = np.random.default_rng()
        bi, be = np.nonzero(conflict_mask)

        # Resample room
        r_dl = eng.room_dom_len[be]
        r_valid = r_dl > 0
        bi_v, be_v, r_dl_v = bi[r_valid], be[r_valid], r_dl[r_valid]
        r_idx = (rng.random(len(bi_v)) * r_dl_v).astype(np.int64)
        r_idx = np.minimum(r_idx, r_dl_v - 1)
        X[bi_v, 3 * be_v + 1] = eng.room_domains[be_v, r_idx]

        # Also resample time for ~40 % to break structural overlaps
        do_time = rng.random(len(bi)) < 0.4
        t_bi, t_be = bi[do_time], be[do_time]
        t_dl = eng.time_dom_len[t_be]
        t_v = t_dl > 0
        t_bi, t_be, t_dl = t_bi[t_v], t_be[t_v], t_dl[t_v]
        t_idx = (rng.random(len(t_bi)) * t_dl).astype(np.int64)
        t_idx = np.minimum(t_idx, t_dl - 1)
        X[t_bi, 3 * t_be + 2] = eng.time_domains[t_be, t_idx]

        logger.debug(
            "SpatialResourceProjection: %d/%d events repaired",
            int(conflict_mask.sum()), N * E,
        )
