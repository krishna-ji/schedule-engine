"""Shared callback infrastructure for GA experiment modes.

Provides ``GACallbackBase`` — a pymoo ``Callback`` subclass that handles
common per-generation bookkeeping:

- timing (``gen_times``)
- best-individual tracking (``best_hards``, ``best_softs``, ``best_breakdowns``)
- MOEA quality metrics (hypervolume, spacing, diversity, feasibility, IGD)
- compact console logging

Mode-specific behaviour lives in ``_on_generation()`` which subclasses
override.  The base implementation is a no-op (suitable for baseline mode).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.core.callback import Callback

logger = logging.getLogger(__name__)

# ── Short constraint labels (matches HARD_CONSTRAINT_NAMES order) ────
_SHORT = ["grp", "inst", "room", "qual", "suit", "iAvl", "rAvl", "comp", "sib"]

# ── Short labels for soft constraint components ──────────────────────
_SHORT_SOFT = ["sComp", "iComp", "lunch", "paired"]

# ── MOEA metrics computed every K generations ────────────────────────
_METRICS_INTERVAL = 10


# =====================================================================
#  Helper functions (pure / minimal side-effects)
# =====================================================================


def _init_moea_lists(cb: Any) -> None:
    """Attach empty MOEA metric lists to a callback instance."""
    cb.hypervolumes = []
    cb.spacings = []
    cb.diversities = []
    cb.feasibility_rates = []
    cb.igds = []
    # Running element-wise max of F for adaptive HV reference point
    cb._hv_running_max = None
    # Optional reference front for IGD (loaded lazily once)
    cb._ref_front = None
    cb._ref_front_checked = False


def _record_moea_metrics(cb: Any, algorithm: Any, F: np.ndarray, G: np.ndarray) -> None:
    """Record MOEA metrics every ``_METRICS_INTERVAL`` generations.

    Policy:
    - HV / spacing / diversity computed on *feasible-only* subset.
    - ``nan`` stored when no feasible solutions exist.
    - HV uses an *adaptive* reference point: ``1.1 × element-wise max``
      of F across all generations seen so far.
    - IGD recorded only if a reference front file is present.
    """
    if algorithm.n_gen % _METRICS_INTERVAL != 0:
        return
    from src.experiments.moea_metrics import (compute_diversity,
                                              compute_feasibility_rate,
                                              compute_hypervolume, compute_igd,
                                              compute_spacing, filter_feasible,
                                              load_reference_front,
                                              update_ref_point_max)

    # Feasibility rate uses ALL individuals
    cb.feasibility_rates.append(compute_feasibility_rate(G))

    # Update adaptive reference point from ALL F (not just feasible)
    cb._hv_running_max, ref_point = update_ref_point_max(cb._hv_running_max, F)

    # Feasible-only subset for quality metrics
    F_feas = filter_feasible(F, G)
    if F_feas is None or F_feas.shape[0] == 0:
        cb.hypervolumes.append(float("nan"))
        cb.spacings.append(float("nan"))
        cb.diversities.append(float("nan"))
        cb.igds.append(float("nan"))
        return

    cb.hypervolumes.append(compute_hypervolume(F_feas, ref_point=ref_point))
    cb.spacings.append(compute_spacing(F_feas))
    cb.diversities.append(compute_diversity(F_feas))

    # IGD — lazy-load reference front once
    if not cb._ref_front_checked:
        cb._ref_front_checked = True
        root = Path(__file__).resolve().parent.parent.parent
        for ext in (".npy", ".csv"):
            rf = load_reference_front(root / f"reference_front{ext}")
            if rf is not None:
                cb._ref_front = rf
                break
    if cb._ref_front is not None:
        cb.igds.append(compute_igd(F_feas, cb._ref_front))
    else:
        cb.igds.append(float("nan"))


# G columns treated as tolerated (soft) — must match scheduling_problem._TOLERATED_HARD_COLS
_TOLERATED_COLS = frozenset({5})  # iAvl


def _progress_payload(algorithm: Any) -> tuple:
    """Extract F, G, cv, best_idx from current population."""
    F = algorithm.pop.get("F")
    G = algorithm.pop.get("G")
    # cv uses only strict hard columns (excludes tolerated like iAvl)
    strict_cols = [i for i in range(G.shape[1]) if i not in _TOLERATED_COLS]
    cv = G[:, strict_cols].sum(axis=1).clip(0)
    best_idx = int(np.argmin(cv))
    return F, G, cv, best_idx


def _constraint_breakdown(G_row: np.ndarray) -> dict[str, int]:
    """Return {short_name: violation_count} for one individual."""
    return {n: int(v) for n, v in zip(_SHORT, G_row, strict=False)}


def _soft_breakdown(problem: Any, best_idx: int) -> dict[str, int]:
    """Return {short_name: penalty} for best individual's soft constraints."""
    bd = getattr(problem, "_last_soft_breakdown", None)
    if bd is None:
        return {}
    keys = [
        "student_compactness",
        "instructor_compactness",
        "lunch_break",
        "paired_cohort",
    ]
    labels = _SHORT_SOFT
    result = {}
    for key, label in zip(keys, labels):
        arr = bd.get(key)
        if arr is not None:
            result[label] = (
                int(arr[best_idx]) if hasattr(arr, "__getitem__") else int(arr)
            )
    return result


def _log_gen(algorithm: Any, log_interval: int) -> tuple:
    """Log generation summary and return (F, G, cv, best_idx)."""
    F, G, cv, best_idx = _progress_payload(algorithm)
    if algorithm.n_gen == 1 or algorithm.n_gen % log_interval == 0:
        bd = G[best_idx]
        # Mark tolerated columns with ~ prefix so logs show e.g. ~iAvl=15
        parts = " ".join(
            f"{'~' if i in _TOLERATED_COLS else ''}{n}={int(v)}"
            for i, (n, v) in enumerate(zip(_SHORT, bd, strict=False))
        )
        # Soft breakdown from problem
        sbd = _soft_breakdown(algorithm.problem, best_idx)
        soft_parts = " ".join(f"{k}={v}" for k, v in sbd.items()) if sbd else ""
        logger.info(
            "Gen %4d: hard=%.0f  (%s)  soft=%.0f  (%s)  cv=%.0f  feasible=%d/%d",
            algorithm.n_gen,
            F[best_idx, 0],
            parts,
            F[best_idx, 1],
            soft_parts,
            cv.min(),
            int((cv == 0).sum()),
            len(algorithm.pop),
        )
    return F, G, cv, best_idx


# =====================================================================
#  GACallbackBase — shared callback for all GA modes
# =====================================================================


class GACallbackBase(Callback):
    """Shared pymoo callback with common per-generation tracking.

    Handles:
    - per-generation timing
    - best hard/soft/breakdown recording
    - MOEA metric recording (HV, spacing, diversity, feasibility, IGD)
    - compact console logging

    Subclasses override ``_on_generation()`` for mode-specific logic
    (repair, escalation, CP polish, etc.).  The base implementation is
    a no-op, suitable for baseline mode.

    Parameters
    ----------
    log_interval : int
        Generations between detailed console log lines.
    """

    def __init__(self, log_interval: int) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.best_hards: list[float] = []
        self.best_softs: list[float] = []
        self.best_breakdowns: list[dict[str, int]] = []
        self.best_soft_breakdowns: list[dict[str, int]] = []
        self.gen_times: list[float] = []
        self._gen_t0: float = time.time()
        _init_moea_lists(self)

    def notify(self, algorithm: Any) -> None:
        """Per-generation hook: track metrics, then delegate to mode hook."""
        now = time.time()
        gen_dt = now - self._gen_t0
        self.gen_times.append(gen_dt)
        self._gen_t0 = now

        F, G, cv, best_idx = _log_gen(algorithm, self.log_interval)
        cur_hard = float(F[best_idx, 0])
        cur_soft = float(F[best_idx, 1])
        self.best_hards.append(cur_hard)
        self.best_softs.append(cur_soft)
        self.best_breakdowns.append(_constraint_breakdown(G[best_idx]))
        self.best_soft_breakdowns.append(_soft_breakdown(algorithm.problem, best_idx))
        _record_moea_metrics(self, algorithm, F, G)

        # ── Debug: improvement / stagnation detection ────────────
        gen = algorithm.n_gen
        n_hards = len(self.best_hards)
        if n_hards >= 2:
            prev_hard = self.best_hards[-2]
            delta = prev_hard - cur_hard
            if delta > 0:
                logger.debug(
                    "Gen %4d: IMPROVED  hard %.0f -> %.0f (delta=%.0f)  dt=%.2fs",
                    gen,
                    prev_hard,
                    cur_hard,
                    delta,
                    gen_dt,
                )
            elif n_hards >= 10 and cur_hard == self.best_hards[-10]:
                logger.debug(
                    "Gen %4d: STAGNANT  hard=%.0f unchanged for 10 gens  dt=%.2fs",
                    gen,
                    cur_hard,
                    gen_dt,
                )
        else:
            logger.debug(
                "Gen %4d: INIT  hard=%.0f  soft=%.0f  dt=%.2fs",
                gen,
                cur_hard,
                cur_soft,
                gen_dt,
            )

        # ── Debug: population diversity snapshot ─────────────────
        if gen % self.log_interval == 0:
            unique_hard = len(np.unique(F[:, 0]))
            mean_cv = float(cv.mean())
            logger.debug(
                "Gen %4d: pop diversity=%d unique hard values, mean_cv=%.1f, "
                "min_cv=%.0f, feasible=%d/%d",
                gen,
                unique_hard,
                mean_cv,
                cv.min(),
                int((cv == 0).sum()),
                len(cv),
            )

        self._on_generation(algorithm, F, G, cv, best_idx)

    def _on_generation(
        self,
        algorithm: Any,
        F: np.ndarray,
        G: np.ndarray,
        cv: np.ndarray,
        best_idx: int,
    ) -> None:
        """Override in mode subclasses for per-generation behaviour.

        Called after common tracking is complete.  ``self.best_hards[-1]``
        is already the current generation's best hard violation.

        Parameters
        ----------
        algorithm : pymoo Algorithm
        F : ndarray, shape (pop_size, n_obj)
        G : ndarray, shape (pop_size, n_constr)
        cv : ndarray, shape (pop_size,) — clipped constraint violation sum
        best_idx : int — index of best individual (lowest cv)
        """
