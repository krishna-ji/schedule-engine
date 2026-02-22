"""Per-generation MOEA metric computation for callbacks.

Computes hypervolume, spacing, diversity, feasibility rate, and IGD
from pymoo population data.  All functions are pure (no side effects)
and work with numpy arrays directly.

Metric policy
-------------
- **Feasible-only**: HV, spacing, diversity are computed on the
  *feasible* subset of the population (``G.sum(axis=1) <= 0``).
  When no feasible solutions exist, ``float('nan')`` is returned.
- **Adaptive reference point**: The caller should maintain a running
  element-wise max of ``F`` across generations and pass
  ``ref_point = 1.1 * max_F_seen`` so that HV is comparable across
  generations.
- **IGD** is optional — only computed when a reference Pareto front
  is supplied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────


def _feasible_mask(G: np.ndarray) -> np.ndarray:
    """Boolean mask for feasible individuals (all constraints ≤ 0)."""
    return G.sum(axis=1) <= 0


def filter_feasible(F: np.ndarray, G: np.ndarray) -> np.ndarray | None:
    """Return F rows for feasible individuals, or *None* if none."""
    mask = _feasible_mask(G)
    if not mask.any():
        return None
    return F[mask]


def update_ref_point_max(
    running_max: np.ndarray | None,
    F: np.ndarray,
    margin: float = 1.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Update element-wise running max and return (running_max, ref_point).

    Parameters
    ----------
    running_max : ndarray or None
        Previous element-wise max across generations.  ``None`` → first call.
    F : ndarray, shape (N, n_obj)
        Current population objectives.
    margin : float
        Multiplicative margin for the reference point (default 1.1 = 10 %).

    Returns
    -------
    running_max : ndarray
        Updated element-wise max.
    ref_point : ndarray
        ``margin * running_max``.
    """
    cur_max = F.max(axis=0)
    if running_max is None:
        running_max = cur_max.copy()
    else:
        running_max = np.maximum(running_max, cur_max)
    # Avoid zero ref_point (causes HV = 0 trivially)
    safe_max = np.where(running_max == 0, 1.0, running_max)
    return running_max, safe_max * margin


# ── Core metrics ─────────────────────────────────────────────────────


def compute_hypervolume(F: np.ndarray, ref_point: np.ndarray | None = None) -> float:
    """Hypervolume indicator for a population objective matrix.

    Parameters
    ----------
    F : ndarray, shape (N, n_obj)
        Objective values — should already be filtered to *feasible* rows
        by the caller.
    ref_point : ndarray, shape (n_obj,), optional
        Reference point.  If ``None``, falls back to ``1.1 * max(F, axis=0)``
        (per-call, NOT adaptive across generations — prefer passing an
        adaptive ref_point from ``update_ref_point_max``).

    Returns
    -------
    float
        Hypervolume value.  Returns 0.0 if pymoo's HV fails.
    """
    try:
        from pymoo.indicators.hv import HV

        if ref_point is None:
            ref_point = F.max(axis=0) * 1.1
            ref_point = np.where(ref_point == 0, 1.0, ref_point)
        hv = HV(ref_point=ref_point)
        return float(hv(F))
    except Exception:
        return 0.0


def compute_spacing(F: np.ndarray) -> float:
    """Spacing indicator — uniformity of the Pareto approximation.

    Parameters
    ----------
    F : ndarray, shape (N, n_obj)
        Should be feasible-only.

    Returns
    -------
    float
        Spacing value (lower = more uniform). Returns 0.0 on error.
    """
    if F.shape[0] < 2:
        return 0.0
    try:
        from scipy.spatial.distance import cdist

        D = cdist(F, F)
        np.fill_diagonal(D, np.inf)
        min_dists = D.min(axis=1)
        return float(np.std(min_dists))
    except Exception:
        return 0.0


def compute_diversity(F: np.ndarray) -> float:
    """Population diversity — average std of objective columns.

    Parameters
    ----------
    F : ndarray, shape (N, n_obj)
        Should be feasible-only.

    Returns
    -------
    float
        Mean standard deviation across objectives.
    """
    if F.shape[0] < 2:
        return 0.0
    return float(np.std(F, axis=0).mean())


def compute_feasibility_rate(G: np.ndarray) -> float:
    """Fraction of population that is feasible (all G <= 0).

    Parameters
    ----------
    G : ndarray, shape (N, n_constr)
        Constraint violation matrix.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    return float(_feasible_mask(G).mean())


def compute_igd(F: np.ndarray, ref_front: np.ndarray) -> float:
    """Inverted Generational Distance (IGD).

    Parameters
    ----------
    F : ndarray, shape (N, n_obj)
        Current approximation set (should be feasible-only).
    ref_front : ndarray, shape (M, n_obj)
        Known / approximate true Pareto front.

    Returns
    -------
    float
        IGD value (lower = better).  Returns ``nan`` on error.
    """
    try:
        from pymoo.indicators.igd import IGD

        igd = IGD(ref_front)
        return float(igd(F))
    except Exception:
        return float("nan")


def load_reference_front(path: str | Path) -> np.ndarray | None:
    """Load a reference Pareto front from disk.

    Supports ``.npy`` and ``.csv`` (no header, columns = objectives).
    Returns ``None`` if the file does not exist or cannot be read.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix == ".npy":
            return np.load(p)
        if p.suffix == ".csv":
            return np.loadtxt(p, delimiter=",")
        return None
    except Exception:
        return None
