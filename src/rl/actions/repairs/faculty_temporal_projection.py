r"""Faculty Temporal Projection — resolve instructor double-booking (FTE).

Detects instructor double-booking via ``np.bincount`` on linearised
instructor×time keys.  Also penalises availability violations with
$5\times$ weight.  Conflicting events get their **instructor** and
**time** resampled from their respective domain arrays.

Complexity: $O(N \cdot Q)$.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class FacultyTemporalProjection(_AtomicRepairBase):
    """Action 1 — project population onto FTE feasibility surface."""

    ACTION_NAME: ClassVar[str] = "faculty_temporal_projection"

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        T_ = __import__("src.pipeline.bitset_time", fromlist=["T"]).T

        eng._fix_domains_vec(X)

        inst = np.clip(X[:, 0::3], 0, eng.n_instructors - 1).astype(np.int64)
        time = X[:, 2::3].astype(np.int64)
        n_idx = np.arange(N, dtype=np.int64)[:, None]

        starts_exp = time[:, eng.exp_event]
        quanta_exp = np.clip(starts_exp + eng.exp_offset[None, :], 0, T_ - 1)
        insts_exp = inst[:, eng.exp_event]
        event_lin = (n_idx * E + eng.exp_event[None, :]).ravel()
        NE = N * E

        nIT = np.int64(eng.n_instructors) * np.int64(T_)
        inst_keys = (n_idx * nIT + insts_exp * T_ + quanta_exp).ravel()
        inst_cnt = np.bincount(inst_keys, minlength=int(N * nIT))
        inst_conflict = (inst_cnt[inst_keys] > 1).astype(np.float64)

        inst_unavail = (~eng.inst_avail[insts_exp.ravel(), quanta_exp.ravel()]).astype(
            np.float64
        ) * 5.0

        q_score = inst_conflict + inst_unavail
        scores = np.bincount(event_lin, weights=q_score, minlength=NE)
        scores = scores[:NE].reshape(N, E)

        conflict_mask = scores > 0
        if not conflict_mask.any():
            return

        rng = np.random.default_rng()
        bi, be = np.nonzero(conflict_mask)

        # Resample instructor
        i_dl = eng.inst_dom_len[be]
        i_valid = i_dl > 1
        i_bi, i_be, i_dl_v = bi[i_valid], be[i_valid], i_dl[i_valid]
        if len(i_bi) > 0:
            i_idx = (rng.random(len(i_bi)) * i_dl_v).astype(np.int64)
            i_idx = np.minimum(i_idx, i_dl_v - 1)
            X[i_bi, 3 * i_be] = eng.inst_domains[i_be, i_idx]

        # Resample time for all conflicting events
        t_dl = eng.time_dom_len[be]
        t_valid = t_dl > 0
        t_bi, t_be, t_dl_v = bi[t_valid], be[t_valid], t_dl[t_valid]
        t_idx = (rng.random(len(t_bi)) * t_dl_v).astype(np.int64)
        t_idx = np.minimum(t_idx, t_dl_v - 1)
        X[t_bi, 3 * t_be + 2] = eng.time_domains[t_be, t_idx]

        logger.debug(
            "FacultyTemporalProjection: %d/%d events repaired",
            int(conflict_mask.sum()),
            N * E,
        )
