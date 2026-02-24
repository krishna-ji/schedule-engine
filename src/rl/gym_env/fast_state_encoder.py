r"""Vectorized state encoder operating directly on Pymoo population arrays.

Extracts a **39-dimensional** normalised observation vector from the
algorithm's ``pop.F``, ``pop.G``, and ``pop.X`` matrices using pure
NumPy/SciPy operations.  **Zero Python loops over individuals.**

Feature layout (39-D)
---------------------

+-------+---------+---------------------------------------------------+
| Start | Count   | Description                                       |
+=======+=========+===================================================+
|   0   |   5     | **Fitness stats** on ``pop.F``                    |
|       |         | (min, max, mean, std, ptp — column 0)             |
+-------+---------+---------------------------------------------------+
|   5   |   3     | **Constraint violation stats** on ``pop.G``       |
|       |         | (mean total, max total, frac feasible)            |
+-------+---------+---------------------------------------------------+
|   8   |   5     | **Diversity** — pairwise distances on ``pop.X``   |
|       |         | via ``scipy.spatial.distance.pdist``              |
|       |         | (mean, std, min, max, ptp)                        |
+-------+---------+---------------------------------------------------+
|  13   |  12     | **Constraint breakdown** (mean across pop)        |
|       |         | 8 hard (CTE…ICTD) + 4 soft (CSC, FSC, MIP, SSCP) |
+-------+---------+---------------------------------------------------+
|  25   |   4     | **Progress** (gen / max_gen, stagnation,          |
|       |         | convergence rate, feasibility gain)                |
+-------+---------+---------------------------------------------------+
|  29   |  10     | **Heuristic history** (last 10 action IDs)        |
+-------+---------+---------------------------------------------------+

All features are clipped to $[0, 1]$.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Try to import scipy for fast pairwise distance
try:
    from scipy.spatial.distance import pdist
except ImportError:
    pdist = None  # type: ignore[assignment]

# Canonical constraint names (must match SchedulingProblem's G columns)
HARD_CONSTRAINT_NAMES = [
    "CTE",  # 0  Cohort Temporal Exclusivity
    "FTE",  # 1  Faculty Temporal Exclusivity
    "SRE",  # 2  Spatial Resource Exclusivity
    "FPC",  # 3  Faculty Pedagogical Congruence
    "FFC",  # 4  Facility Feature Congruence
    "FCA",  # 5  Faculty Chronological Availability
    "CQF",  # 6  Curriculum Quanta Fulfillment
    "ICTD",  # 7  Intra-Course Temporal Dispersion
]

SOFT_CONSTRAINT_NAMES = [
    "CSC",  # Cohort Schedule Contiguity
    "FSC",  # Faculty Schedule Contiguity
    "MIP",  # Meridian Interval Preservation
    "SSCP",  # Symmetric Sub-Cohort Parallelism
]

OBS_DIM: int = 39
_HISTORY_SIZE: int = 10
_MAX_PDIST_POP: int = 200  # subsample for pdist if pop > this


class VectorizedStateEncoder:
    r"""Extract a 39-D $[0,1]$ observation from a Pymoo population.

    All computations use NumPy broadcasting / aggregation and
    ``scipy.spatial.distance.pdist`` — no per-individual Python loops.

    Parameters
    ----------
    max_generations : int
        Maximum generation budget (for normalising progress features).
    history_size : int
        How many recent action IDs to keep in the history ring.
    n_events : int
        Number of scheduling events $E$ (for normalising diversity).
    """

    def __init__(
        self,
        max_generations: int = 500,
        history_size: int = _HISTORY_SIZE,
        n_events: int | None = None,
    ):
        self.max_generations = max(max_generations, 1)
        self.history_size = history_size
        self.n_events = n_events

        # Internal mutable state (progress tracking)
        self._gen: int = 0
        self._stagnation: int = 0
        self._prev_best_hard: float = np.inf
        self._prev_frac_feasible: float = 0.0
        self._history: deque[int] = deque(
            [0] * history_size, maxlen=history_size
        )

        # Normalisation constants (updated lazily from observed ranges)
        self._hard_max: float = 1.0  # will be updated from first obs
        self._soft_max: float = 1.0
        self._div_max: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal counters for a new episode."""
        self._gen = 0
        self._stagnation = 0
        self._prev_best_hard = np.inf
        self._prev_frac_feasible = 0.0
        self._history = deque(
            [0] * self.history_size, maxlen=self.history_size
        )
        self._hard_max = 1.0
        self._soft_max = 1.0
        self._div_max = 1.0

    def encode(
        self,
        F: np.ndarray,
        G: np.ndarray,
        X: np.ndarray,
        soft_breakdown: dict[str, np.ndarray] | None = None,
        action_taken: int | None = None,
    ) -> NDArray[np.float32]:
        r"""Encode population matrices into a 39-D observation.

        Parameters
        ----------
        F : ndarray, shape ``(N, 2)``
            Objective values.  ``F[:, 0]`` = hard, ``F[:, 1]`` = soft.
        G : ndarray, shape ``(N, 8)``
            Hard-constraint violation matrix.
        X : ndarray, shape ``(N, 3E)``
            Decision variable matrix.
        soft_breakdown : dict[str, ndarray(N,)] | None
            Per-soft-constraint penalty arrays (keys: CSC, FSC, MIP, SSCP).
        action_taken : int | None
            Action ID that was taken this step (pushed to history).

        Returns
        -------
        obs : ndarray, shape ``(39,)``, float32
            Normalised observation vector in $[0, 1]$.
        """
        obs = np.zeros(OBS_DIM, dtype=np.float64)

        # -- Fitness features (5) ----------------------------------------
        hard = F[:, 0]
        self._hard_max = max(self._hard_max, float(hard.max()) + 1e-12)
        obs[0] = hard.min() / self._hard_max
        obs[1] = hard.max() / self._hard_max
        obs[2] = hard.mean() / self._hard_max
        obs[3] = hard.std() / self._hard_max if hard.std() > 0 else 0.0
        obs[4] = float(hard.max() - hard.min()) / self._hard_max

        # -- Constraint violation stats (3) --------------------------------
        total_viol = G.sum(axis=1)  # (N,)
        viol_max = max(float(total_viol.max()), 1e-12)
        obs[5] = total_viol.mean() / viol_max
        obs[6] = total_viol.max() / viol_max  # always 1.0 but useful when 0
        frac_feasible = float((total_viol == 0).mean())
        obs[7] = frac_feasible

        # -- Diversity features (5) ----------------------------------------
        obs[8:13] = self._compute_diversity(X)

        # -- Constraint breakdown (12) -------------------------------------
        obs[13:25] = self._compute_constraint_breakdown(G, soft_breakdown)

        # -- Progress features (4) ----------------------------------------
        obs[25:29] = self._compute_progress(hard, frac_feasible)

        # -- Heuristic history (10) ----------------------------------------
        if action_taken is not None:
            self._history.append(action_taken)
        from src.rl.actions.vectorized_ops import NUM_ACTIONS

        max_act = max(NUM_ACTIONS, 1)
        for i, act_id in enumerate(self._history):
            obs[29 + i] = act_id / max_act

        # Clip to [0, 1] and cast
        np.clip(obs, 0.0, 1.0, out=obs)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Private feature extractors
    # ------------------------------------------------------------------

    def _compute_diversity(self, X: np.ndarray) -> NDArray[np.float64]:
        """5-D diversity features via pairwise Hamming-style distances."""
        feats = np.zeros(5, dtype=np.float64)
        N = X.shape[0]

        if N < 2:
            return feats  # not enough for pairwise

        # Subsample if population is large (pdist is O(N^2))
        if N > _MAX_PDIST_POP:
            idx = np.random.default_rng(0).choice(N, _MAX_PDIST_POP, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        if pdist is not None:
            # Use normalised Hamming distance (fraction of differing genes)
            dists = pdist(X_sub, metric="hamming")
        else:
            # Fallback: unique-row ratio as a scalar diversity proxy
            unique_ratio = len(np.unique(X_sub, axis=0)) / len(X_sub)
            feats[0] = unique_ratio
            return feats

        if len(dists) == 0:
            return feats

        self._div_max = max(self._div_max, float(dists.max()) + 1e-12)
        feats[0] = dists.mean() / self._div_max
        feats[1] = dists.std() / self._div_max if dists.std() > 0 else 0.0
        feats[2] = dists.min() / self._div_max
        feats[3] = dists.max() / self._div_max
        feats[4] = float(dists.max() - dists.min()) / self._div_max
        return feats

    def _compute_constraint_breakdown(
        self,
        G: np.ndarray,
        soft_breakdown: dict[str, np.ndarray] | None,
    ) -> NDArray[np.float64]:
        """12-D per-constraint violation means (normalised)."""
        feats = np.zeros(12, dtype=np.float64)

        # Hard constraint columns (8)
        for i in range(min(G.shape[1], 8)):
            col = G[:, i]
            col_max = max(float(col.max()), 1.0)
            feats[i] = float(col.mean()) / col_max

        # Soft constraint breakdown (4)
        if soft_breakdown is not None:
            for j, name in enumerate(SOFT_CONSTRAINT_NAMES):
                arr = soft_breakdown.get(name)
                if arr is not None:
                    arr = np.asarray(arr, dtype=np.float64)
                    if arr.ndim == 0:
                        # scalar → broadcast
                        val = float(arr)
                        self._soft_max = max(self._soft_max, abs(val) + 1e-12)
                        feats[8 + j] = abs(val) / self._soft_max
                    else:
                        arr_max = max(float(np.abs(arr).max()), 1e-12)
                        self._soft_max = max(self._soft_max, arr_max)
                        feats[8 + j] = float(np.abs(arr).mean()) / self._soft_max

        return feats

    def _compute_progress(
        self,
        hard: np.ndarray,
        frac_feasible: float,
    ) -> NDArray[np.float64]:
        """4-D progress features."""
        feats = np.zeros(4, dtype=np.float64)

        self._gen += 1
        feats[0] = min(self._gen / self.max_generations, 1.0)

        # Stagnation counter
        best_hard = float(hard.min())
        if best_hard < self._prev_best_hard - 1e-8:
            self._stagnation = 0
        else:
            self._stagnation += 1
        feats[1] = min(self._stagnation / max(self.max_generations * 0.1, 1.0), 1.0)

        # Convergence rate = improvement per gen
        if self._prev_best_hard > 0 and self._prev_best_hard < np.inf:
            delta = (self._prev_best_hard - best_hard) / self._prev_best_hard
            feats[2] = float(np.clip(delta, -1.0, 1.0) * 0.5 + 0.5)
        else:
            feats[2] = 0.5

        # Feasibility gain
        feats[3] = float(
            np.clip(frac_feasible - self._prev_frac_feasible + 0.5, 0.0, 1.0)
        )

        self._prev_best_hard = best_hard
        self._prev_frac_feasible = frac_feasible

        return feats
