"""Canonical batch API for the scheduling pipeline.

All public functions operate on **numpy arrays only** — no OOP wrappers,
no per-individual Python loops in the caller.

Shapes & dtypes
~~~~~~~~~~~~~~~
    X : int64, (N, 3*E)    — population matrix (interleaved chromosomes)
    G : int64, (N, 8)      — hard constraint violation counts (8 constraints)
    F : float64, (N, 2)    — objectives [hard_total, soft_total]
    S : float64, (N,)      — soft penalty per individual
    hv, igd, spacing : float64 scalars

Runtime asserts verify shapes/dtypes at every entry point.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np

from .fast_evaluator_vectorized import (
    VectorizedEvalData,
    fast_evaluate_hard_vectorized,
    prepare_vectorized_data,
)
from .soft_evaluator_vectorized import (
    SoftVectorizedData,
    eval_soft_vectorized,
    prepare_soft_vectorized_data,
)

if TYPE_CHECKING:
    from .repair_operator_bitset import BitsetSchedulingRepair

__all__ = [
    "BatchContext",
    "eval_hard_batch",
    "eval_soft_batch",
    "metrics_batch",
    "repair_batch",
]


# ------------------------------------------------------------------
# Shared context (avoids repeated pkl loading)
# ------------------------------------------------------------------


class BatchContext:
    """Pre-computed data for batch evaluation.

    Construct once, pass to all ``eval_*_batch`` and ``repair_batch`` calls.
    """

    def __init__(self, pkl_path: str = ".cache/events_with_domains.pkl"):
        with open(pkl_path, "rb") as f:
            self.pkl_data: dict = pickle.load(f)

        self.vdata: VectorizedEvalData = prepare_vectorized_data(self.pkl_data)
        self.sdata: SoftVectorizedData = prepare_soft_vectorized_data(self.pkl_data)
        self.events: list[dict] = self.pkl_data["events"]
        self.n_events: int = len(self.events)
        self.pkl_path: str = pkl_path

        # Lazy-init
        self._repairer: BitsetSchedulingRepair | None = None

    @property
    def repairer(self) -> BitsetSchedulingRepair:
        if self._repairer is None:
            from .repair_operator_bitset import BitsetSchedulingRepair

            self._repairer = BitsetSchedulingRepair(self.pkl_path)
        return self._repairer


# ------------------------------------------------------------------
# Shape / dtype assertions
# ------------------------------------------------------------------


def _assert_population(X: np.ndarray, n_events: int) -> np.ndarray:
    """Validate and coerce population matrix."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    assert X.ndim == 2, f"X must be 2-D, got shape {X.shape}"
    assert X.shape[1] == 3 * n_events, f"X.shape[1]={X.shape[1]} != 3*E={3 * n_events}"
    return X.astype(np.int64, copy=False)


# ------------------------------------------------------------------
# eval_hard_batch
# ------------------------------------------------------------------


def eval_hard_batch(
    X: np.ndarray,
    ctx: BatchContext,
) -> np.ndarray:
    """Evaluate hard constraints for the full population.

    Parameters
    ----------
    X : ndarray, shape (N, 3*E), int
    ctx : BatchContext

    Returns
    -------
    G : ndarray, shape (N, 8), int64
        Per-constraint violation counts.
    """
    X = _assert_population(X, ctx.n_events)
    G = fast_evaluate_hard_vectorized(X, ctx.vdata)
    assert G.shape == (X.shape[0], 8), f"G shape mismatch: {G.shape}"
    assert G.dtype == np.int64, f"G dtype mismatch: {G.dtype}"
    return G


# ------------------------------------------------------------------
# eval_soft_batch  (Phase B will replace the stub)
# ------------------------------------------------------------------


def eval_soft_batch(
    X: np.ndarray,
    ctx: BatchContext,
) -> np.ndarray:
    """Evaluate soft constraints for the full population.

    Parameters
    ----------
    X : ndarray, shape (N, 3*E), int
    ctx : BatchContext

    Returns
    -------
    S : ndarray, shape (N,), float64
        Total soft penalty per individual.

    Notes
    -----
    Phase B will add a true vectorized implementation.
    Currently returns zeros (soft eval requires OOP timetable construction).
    """
    X = _assert_population(X, ctx.n_events)
    S = eval_soft_vectorized(X, ctx.sdata)
    assert S.shape == (X.shape[0],), f"S shape mismatch: {S.shape}"
    assert S.dtype == np.float64, f"S dtype mismatch: {S.dtype}"
    return S


# ------------------------------------------------------------------
# repair_batch
# ------------------------------------------------------------------


def repair_batch(
    X: np.ndarray,
    ctx: BatchContext,
) -> np.ndarray:
    """Repair a batch of chromosomes.

    Applies vectorized domain fixing across the population first,
    then per-individual conflict repair.

    Parameters
    ----------
    X : ndarray, shape (N, 3*E), int
    ctx : BatchContext

    Returns
    -------
    X_repaired : ndarray, shape (N, 3*E), int64
    """
    from .repair_analysis_vectorized import fix_domains_batch

    X = _assert_population(X, ctx.n_events)
    N = X.shape[0]
    out = X.copy()

    # Stage 1: vectorized domain fix across all individuals
    repairer = ctx.repairer
    fix_domains_batch(
        out,
        repairer.allowed_instructors,
        repairer.allowed_rooms,
        repairer.allowed_starts,
        inst_avail=repairer.inst_avail,
        durations=repairer.durations,
    )

    # Stages 2-3: per-individual conflict repair
    for i in range(N):
        out[i] = repairer.repair(out[i])

    assert out.shape == X.shape
    return out


# ------------------------------------------------------------------
# metrics_batch
# ------------------------------------------------------------------


def metrics_batch(
    F: np.ndarray,
    ref_point: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute multi-objective quality metrics from an objective matrix.

    Parameters
    ----------
    F : ndarray, shape (N, n_obj), float64
        Objective values for each individual.
    ref_point : ndarray, shape (n_obj,), optional
        Reference point for HV.  Auto-computed if None.

    Returns
    -------
    dict with keys:
        hv : float — hypervolume indicator
        igd : float — inverted generational distance (NaN if no ref front)
        spacing : float — spacing indicator
        n_fronts : int — number of non-dominated fronts
    """
    F = np.asarray(F, dtype=np.float64)
    assert F.ndim == 2, f"F must be 2-D, got {F.shape}"
    N, n_obj = F.shape

    result: dict[str, float] = {}

    # Non-dominated sorting
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    nds = NonDominatedSorting()
    fronts = nds.do(F)
    result["n_fronts"] = len(fronts)
    pf = F[fronts[0]]  # Pareto front

    # Hypervolume
    if ref_point is None:
        ref_point = F.max(axis=0) * 1.1 + 1.0
    ref_point = np.asarray(ref_point, dtype=np.float64)
    assert ref_point.shape == (n_obj,), f"ref_point shape {ref_point.shape}"

    from pymoo.indicators.hv import HV

    hv_ind = HV(ref_point=ref_point)
    result["hv"] = float(hv_ind(pf))

    # Spacing
    if len(pf) > 1:
        from scipy.spatial.distance import pdist

        dists = pdist(pf)
        nn_dists = []
        from scipy.spatial.distance import squareform

        dm = squareform(dists)
        np.fill_diagonal(dm, np.inf)
        nn_dists = dm.min(axis=1)
        result["spacing"] = float(np.std(nn_dists))
    else:
        result["spacing"] = 0.0

    # IGD — use Pareto front as reference (self-IGD, baseline)
    from pymoo.indicators.igd import IGD

    igd_ind = IGD(pf)
    result["igd"] = float(igd_ind(F))

    return result
