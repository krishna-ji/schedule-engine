r"""Potential-Based Reward Shaping (PBRS) for the scheduling hyper-heuristic.

Implements :math:`F(s, s') = \gamma \Phi(s') - \Phi(s)` where
:math:`\Phi(s)` captures the **Bottleneck Density** of the schedule
state.  By Ng et al. (1999), this additive shaping is guaranteed to
preserve the optimal policy while providing dense gradient signal in
local minima where :math:`\Delta_\text{hard} \approx 0`.

Potential Function
------------------

.. math::

    \Phi(s) = -\frac{
        \operatorname{Var}(g_0, \ldots, g_7)
        + \operatorname{Var}(\text{per-inst conflicts})
        + \operatorname{Var}(\text{per-room conflicts})
    }{\text{max\_var}}

Intuition: a state where conflicts are **spread evenly** across
resources (low variance) is easier to fix than one where a single
instructor or room concentrates all violations (high variance →
structural bottleneck).  Higher :math:`\Phi` = better state.

Complexity: :math:`O(E)` per call (one individual, one expansion).
With :math:`E \approx 70` and :math:`Q \approx 140`, this is ~0.01 ms
— invisible next to the ~5 s/step GA overhead.

Phase 62 — Titan V4 SOTA Algorithmic Overhaul
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.pipeline.fast_evaluator_vectorized import VectorizedEvalData

logger = logging.getLogger(__name__)

# Total quanta in the timetable (6 days × 7 slots)
_T: int = 42


class StatePotentialCalculator:
    r"""Compute state potential :math:`\Phi(s)` for PBRS.

    Two-tier bottleneck density:

    **Tier 1** — Constraint-type concentration (:math:`O(8)`):
        Variance across the 8 hard-constraint columns of the best
        individual's G row.  Captures "are violations dominated by
        one constraint type?"

    **Tier 2** — Per-resource concentration (:math:`O(E)`):
        Variance of per-instructor and per-room conflict counts from
        the best individual's chromosome.  Captures "is one instructor
        carrying all the double-bookings?"

    Parameters
    ----------
    gamma : float
        Discount factor for the PBRS shaping term.
    use_chromosome : bool
        If True, compute Tier 2 (per-resource) variance from X.
        If False, use only Tier 1 (fast, from G columns only).
    """

    def __init__(self, gamma: float = 0.99, use_chromosome: bool = True):
        self.gamma = gamma
        self.use_chromosome = use_chromosome

        # Dynamic normalisation — tracks max variance seen so far
        self._max_var: float = 1.0

    def reset(self) -> None:
        """Reset normalisation state for a new training run."""
        self._max_var = 1.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def potential(
        self,
        F: np.ndarray,
        G: np.ndarray,
        X: np.ndarray | None = None,
        vec_data: VectorizedEvalData | None = None,
    ) -> float:
        r"""Compute :math:`\Phi(s)` from population arrays.

        Parameters
        ----------
        F : ndarray (N, 2)
            Objective values. ``F[:, 0]`` = hard penalty.
        G : ndarray (N, 8)
            Per-constraint violation matrix.
        X : ndarray (N, 3E) | None
            Decision variable matrix (for Tier 2).
        vec_data : VectorizedEvalData | None
            Precomputed evaluation data (for Tier 2).

        Returns
        -------
        float
            State potential (higher = better, always ≤ 0).
        """
        # Identify best individual (lowest hard penalty)
        best_idx = int(F[:, 0].argmin())
        G_best = G[best_idx].astype(np.float64)

        # -- Tier 1: constraint-type variance (O(8)) ---------------------
        constraint_var = float(np.var(G_best))

        # -- Tier 2: per-resource variance (O(E)) ------------------------
        resource_var = 0.0
        if self.use_chromosome and X is not None and vec_data is not None:
            X_best = X[best_idx]
            resource_var = self._per_resource_variance(X_best, vec_data)

        total_var = constraint_var + resource_var
        self._max_var = max(self._max_var, total_var + 1e-12)

        # Negative normalised variance: Φ ∈ [-1, 0]
        # Higher (closer to 0) = conflicts spread evenly = easier to fix
        return -total_var / self._max_var

    def shaping_reward(self, phi_prev: float, phi_new: float) -> float:
        r"""Compute the PBRS additive term.

        .. math::

            F(s, s') = \gamma \Phi(s') - \Phi(s)

        Returns
        -------
        float
            Shaping reward (positive when moving to higher-potential state).
        """
        return self.gamma * phi_new - phi_prev

    # ------------------------------------------------------------------
    # Tier 2: Per-resource conflict variance
    # ------------------------------------------------------------------

    @staticmethod
    def _per_resource_variance(
        x: np.ndarray,
        vec_data: VectorizedEvalData,
    ) -> float:
        r"""Per-instructor + per-room conflict density from chromosome.

        For each resource type, expand the best individual's assignments
        to quanta, count double-bookings per resource, and return the
        variance of those counts.

        Complexity: :math:`O(Q)` where :math:`Q = \sum d_e \approx 140`.

        Parameters
        ----------
        x : ndarray (3E,)
            Single individual's chromosome.
        vec_data : VectorizedEvalData
            Precomputed expansion/membership arrays.

        Returns
        -------
        float
            ``Var(per-instructor conflicts) + Var(per-room conflicts)``.
        """
        n_inst = vec_data.n_instructors
        n_rooms = vec_data.n_rooms
        T = _T

        # Chromosome slicing: [I0, R0, T0, I1, R1, T1, ...]
        inst = x[0::3].astype(np.int64)  # (E,)
        room = x[1::3].astype(np.int64)  # (E,)
        time = x[2::3].astype(np.int64)  # (E,)

        # Expansion arrays (event → quanta)
        exp_event = vec_data.exp_event  # (Q,)
        exp_offset = vec_data.exp_offset  # (Q,)

        # Expand to quanta for the best individual
        starts = time[exp_event]  # (Q,)
        quanta = starts + exp_offset  # (Q,)

        # Clip quanta to valid range [0, T-1] for safety
        np.clip(quanta, 0, T - 1, out=quanta)

        insts = inst[exp_event]  # (Q,)
        rooms = room[exp_event]  # (Q,)

        # -- Per-instructor conflict variance ----------------------------
        flat_inst = insts * T + quanta
        minlen_inst = n_inst * T
        cnt_inst = np.bincount(flat_inst, minlength=minlen_inst)[:minlen_inst]
        # Conflicts = slots where count > 1
        per_inst_conflicts = np.maximum(cnt_inst.reshape(n_inst, T) - 1, 0).sum(
            axis=1
        )  # (n_inst,)
        inst_var = float(np.var(per_inst_conflicts))

        # -- Per-room conflict variance ----------------------------------
        flat_room = rooms * T + quanta
        minlen_room = n_rooms * T
        cnt_room = np.bincount(flat_room, minlength=minlen_room)[:minlen_room]
        per_room_conflicts = np.maximum(cnt_room.reshape(n_rooms, T) - 1, 0).sum(
            axis=1
        )  # (n_rooms,)
        room_var = float(np.var(per_room_conflicts))

        return inst_var + room_var
