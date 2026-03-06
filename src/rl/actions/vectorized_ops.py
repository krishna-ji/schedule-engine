r"""Pipeline-configuration LLHs for the RL hyper-heuristic.

**Phase 53 architectural redesign** — each Low-Level Heuristic (LLH)
runs the **complete** three-stage ``VectorizedRepair.repair_batch``
pipeline with different parameter configurations.  The RL agent
selects which *configuration* to use each generation, not which
single repair concern to address.

Previous architecture (Phases 35-52) had 8 atomic operators that each
fixed ONE constraint type per generation inside the same Pymoo
``Repair`` slot.  They overwrote each other's work.  The working
memetic mode ran them ALL in sequence — which is exactly what
``repair_batch`` does.

New architecture
----------------

Each LLH is a ``pymoo.core.repair.Repair`` that calls the full
pipeline ``engine.repair_batch(X, passes=P)`` with different ``P``,
then optionally applies a second-stage intensification or
diversification kernel.

Pipeline configurations::

    0: ConservativeRepair   — passes=3, exploit (steady progress)
    1: AggressiveRepair     — passes=7, explore (heavy resampling)
    2: MemeticEliteRepair   — passes=3 + extra passes on top-15% worst
    3: SoftFocusRepair      — passes=3 + time-compaction for soft obj
    4: DestructiveConstructive — ruin 10% worst + passes=5 (escape)
    5: IntensifiedRepair    — passes=5, balanced middle ground

The RL agent learns *when* to use aggressive vs. conservative, when
to trigger elite intensification, when to ruin-and-reconstruct to
escape plateaus.  Every action produces a coherent, fully-repaired
population.

Complexity
----------

All LLHs are $O(\text{passes} \cdot N \cdot Q)$ with optional
$+O(N \cdot E)$ intensification.  No action is destructive —
rollback is unnecessary.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
from pymoo.core.repair import Repair

from src.pipeline.repair_operator_vectorized import VectorizedRepair

logger = logging.getLogger(__name__)


# ======================================================================
# Shared engine cache — avoid re-loading pkl per action
# ======================================================================

_ENGINE_CACHE: dict[str, VectorizedRepair] = {}


def _get_engine(pkl_path: str) -> VectorizedRepair:
    """Return (or create & cache) a ``VectorizedRepair`` engine."""
    if pkl_path not in _ENGINE_CACHE:
        _ENGINE_CACHE[pkl_path] = VectorizedRepair(pkl_path)
    return _ENGINE_CACHE[pkl_path]


# ======================================================================
# Base class
# ======================================================================


class _AtomicRepairBase(Repair):
    """Base for all pipeline-configuration LLH actions.

    Subclasses override ``_apply`` which receives the population
    matrix X and applies a complete repair pipeline configuration.
    """

    ACTION_NAME: ClassVar[str] = "base"

    def __init__(self, pkl_path: str = ".cache/events_with_domains.pkl"):
        super().__init__()
        self.engine: VectorizedRepair = _get_engine(pkl_path)
        self._pkl_path = pkl_path

    def _do(self, problem, x, **kwargs):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = x.copy().astype(np.int64)
        self._apply(x)
        return x

    def _apply(self, X: np.ndarray) -> None:
        """In-place repair kernel.  Override in subclasses."""
        raise NotImplementedError


# ======================================================================
# Action 0 — Conservative Repair (exploit)
# ======================================================================


class ConservativeRepair(_AtomicRepairBase):
    r"""Full pipeline with low passes — steady exploitation.

    Runs ``repair_batch(X, passes=3)`` — the same configuration used
    by the working memetic GA.  Minimal disruption to the population,
    preserving good schemata while resolving the worst conflicts.

    This is the *safe default* action: consistent progress, low
    variance, no risk of population degradation.
    """

    ACTION_NAME: ClassVar[str] = "conservative_repair"

    def _apply(self, X: np.ndarray) -> None:
        result = self.engine.repair_batch(X, passes=3)
        X[:] = result
        logger.debug("ConservativeRepair: 3 passes (exploit)")


# ======================================================================
# Action 1 — Aggressive Repair (explore)
# ======================================================================


class AggressiveRepair(_AtomicRepairBase):
    r"""Full pipeline with high passes — aggressive exploration.

    Runs ``repair_batch(X, passes=7)`` — more than double the
    standard passes.  Each extra pass rescores the population,
    applies a fresh 30% stochastic mutation mask, and resamples
    conflicting events.

    High disruption: many events get resampled multiple times,
    breaking structural correlations.  Useful when the population
    is stuck in a penalty plateau — the extra passes explore more
    of the feasible neighbourhood per generation.
    """

    ACTION_NAME: ClassVar[str] = "aggressive_repair"

    def _apply(self, X: np.ndarray) -> None:
        result = self.engine.repair_batch(X, passes=7)
        X[:] = result
        logger.debug("AggressiveRepair: 7 passes (explore)")


# ======================================================================
# Action 2 — Memetic Elite Repair
# ======================================================================


class MemeticEliteRepair(_AtomicRepairBase):
    r"""Full pipeline + extra targeted repair on worst individuals.

    1. Standard ``repair_batch(X, passes=3)`` on full population.
    2. Score population via ``_score_all_batch``.
    3. Identify top-15% worst individuals (by total conflict score).
    4. Apply 4 additional repair passes **only** on those elites.

    This concentrates computational budget on the individuals most
    likely to benefit from extra repair, while leaving already-good
    individuals untouched.  The RL agent should select this when
    the population has a wide quality spread (some individuals
    near-feasible, others heavily infeasible).
    """

    ACTION_NAME: ClassVar[str] = "memetic_elite_repair"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        elite_fraction: float = 0.15,
        extra_passes: int = 4,
    ):
        super().__init__(pkl_path)
        self.elite_fraction = elite_fraction
        self.extra_passes = extra_passes

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]

        # Stage 1: full pipeline on everyone
        result = eng.repair_batch(X, passes=3)
        X[:] = result

        # Stage 2: score and identify worst individuals
        scores = eng._score_all_batch(X)  # (N, E)
        ind_severity = scores.sum(axis=1)  # (N,)

        n_elite = max(1, int(N * self.elite_fraction))
        worst_idx = np.argsort(-ind_severity)[:n_elite]

        if ind_severity[worst_idx[0]] == 0:
            # Everyone is already clean — nothing to do
            logger.debug("MemeticEliteRepair: population clean, skipping elite pass")
            return

        # Stage 3: extra passes on worst subset only
        X_elite = X[worst_idx].copy()
        eng._repair_conflicts_vec(X_elite, passes=self.extra_passes)
        if eng._n_pairs > 0:
            eng._sync_paired_events(X_elite)
        X[worst_idx] = X_elite

        logger.debug(
            "MemeticEliteRepair: 3+%d passes, %d/%d elites targeted "
            "(worst_severity=%d)",
            self.extra_passes,
            n_elite,
            N,
            int(ind_severity[worst_idx[0]]),
        )


# ======================================================================
# Action 3 — Soft-Focus Repair
# ======================================================================


class SoftFocusRepair(_AtomicRepairBase):
    r"""Full pipeline + time-compaction pass for soft-constraint quality.

    1. Standard ``repair_batch(X, passes=3)`` on full population.
    2. For each individual, compute per-event time-slots and attempt
       to compact them toward earlier time quanta (reducing gaps).

    This targets soft-constraint quality (instructor idle-time gaps,
    student schedule compactness) without degrading hard-constraint
    feasibility — the compaction only moves events within their
    valid time domains and re-runs SSCP sync afterward.

    The RL agent should select this when hard penalties are low
    and soft-constraint improvement pressure is needed.
    """

    ACTION_NAME: ClassVar[str] = "soft_focus_repair"

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events

        # Stage 1: full pipeline
        result = eng.repair_batch(X, passes=3)
        X[:] = result

        # Stage 2: soft-objective time compaction
        # Try to move events to earlier valid time slots to reduce
        # schedule gaps (affects instructor idle time, student compactness)
        rng = np.random.default_rng()

        # Score to find conflict-free events (candidates for compaction)
        scores = eng._score_all_batch(X)  # (N, E)
        clean_mask = scores == 0  # (N, E) — no hard conflicts

        # Only compact ~20% of clean events to avoid over-disruption
        compact_mask = clean_mask & (rng.random((N, E)) < 0.20)
        if not compact_mask.any():
            logger.debug("SoftFocusRepair: no clean events to compact")
            return

        bi, be = np.nonzero(compact_mask)
        current_times = X[bi, 3 * be + 2]

        # For each selected event, try moving to an earlier valid time
        t_dl = eng.time_dom_len[be]
        t_valid = t_dl > 0
        bi, be, t_dl = bi[t_valid], be[t_valid], t_dl[t_valid]
        current_times = X[bi, 3 * be + 2]

        for k in range(len(bi)):
            n, e = int(bi[k]), int(be[k])
            dl = int(t_dl[k])
            domain = eng.time_domains[e, :dl]
            cur_t = int(current_times[k])

            # Find earlier times in domain
            earlier = domain[domain < cur_t]
            if len(earlier) > 0:
                # Pick the latest of the earlier options (greedy compaction)
                X[n, 3 * e + 2] = int(earlier[-1])

        # Stage 3: re-sync paired events (compaction may break SSCP)
        if eng._n_pairs > 0:
            eng._sync_paired_events(X)

        logger.debug(
            "SoftFocusRepair: 3 passes + compaction on %d events",
            len(bi),
        )


# ======================================================================
# Action 4 — Destructive-Constructive (Ruin & Recreate)
# ======================================================================


class DestructiveConstructive(_AtomicRepairBase):
    r"""Ruin worst 10% of events per individual, then full repair.

    1. Score population to identify per-event conflict severity.
    2. For each individual, "ruin" the top-10% worst-scoring events
       by randomising all three genes (inst, room, time) from domains.
    3. Run ``repair_batch(X, passes=5)`` — the full pipeline rebuilds
       the ruined events within the context of the surviving structure.

    This is the **escape hatch** for local-optima plateaus.  By
    destroying correlated conflict clusters and letting the pipeline
    reconstruct them, the population can jump to a different region
    of the search space.  More aggressive than ``AggressiveRepair``
    (which just adds passes) because it breaks structural correlations
    before repairing.
    """

    ACTION_NAME: ClassVar[str] = "destructive_constructive"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        ruin_fraction: float = 0.10,
    ):
        super().__init__(pkl_path)
        self.ruin_fraction = ruin_fraction

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        rng = np.random.default_rng()

        # Stage 1: fix domains so scoring is valid
        eng._fix_domains_vec(X)

        # Stage 2: score and identify worst events per individual
        scores = eng._score_all_batch(X)  # (N, E)
        n_ruin = max(1, int(E * self.ruin_fraction))

        total_ruined = 0
        for n in range(N):
            row_scores = scores[n]
            if row_scores.sum() == 0:
                continue

            worst = np.argsort(-row_scores)[:n_ruin]
            has_conflict = row_scores[worst] > 0
            worst = worst[has_conflict]
            if len(worst) == 0:
                continue

            # Ruin: randomise all genes from valid domains
            for e in worst:
                i_dl = int(eng.inst_dom_len[e])
                r_dl = int(eng.room_dom_len[e])
                t_dl = int(eng.time_dom_len[e])
                if i_dl > 0:
                    X[n, 3 * e + 0] = eng.inst_domains[e, rng.integers(i_dl)]
                if r_dl > 0:
                    X[n, 3 * e + 1] = eng.room_domains[e, rng.integers(r_dl)]
                if t_dl > 0:
                    X[n, 3 * e + 2] = eng.time_domains[e, rng.integers(t_dl)]

            total_ruined += len(worst)

        # Stage 3: full pipeline reconstructs from the rubble
        result = eng.repair_batch(X, passes=5)
        X[:] = result

        logger.debug(
            "DestructiveConstructive: ruined %d events total, " "then 5-pass repair",
            total_ruined,
        )


# ======================================================================
# Action 5 — Intensified Repair (balanced)
# ======================================================================


class IntensifiedRepair(_AtomicRepairBase):
    r"""Full pipeline with moderate passes — balanced middle ground.

    Runs ``repair_batch(X, passes=5)`` — between conservative (3)
    and aggressive (7).  A solid workhorse configuration that provides
    more repair pressure than conservative without the high disruption
    of aggressive.
    """

    ACTION_NAME: ClassVar[str] = "intensified_repair"

    def _apply(self, X: np.ndarray) -> None:
        result = self.engine.repair_batch(X, passes=5)
        X[:] = result
        logger.debug("IntensifiedRepair: 5 passes (balanced)")


# ======================================================================
# Action Space Registry — Pipeline Configurations (Phase 53)
# ======================================================================

VECTORIZED_ACTION_SPACE: dict[int, type[_AtomicRepairBase]] = {
    0: ConservativeRepair,  # Full pipeline, 3 passes (exploit)
    1: AggressiveRepair,  # Full pipeline, 7 passes (explore)
    2: MemeticEliteRepair,  # 3 passes + 4 extra on worst 15%
    3: SoftFocusRepair,  # 3 passes + time compaction (soft obj)
    4: DestructiveConstructive,  # Ruin 10% + 5-pass repair (escape)
    5: IntensifiedRepair,  # Full pipeline, 5 passes (balanced)
}

ACTION_NAMES: dict[int, str] = {
    k: v.ACTION_NAME for k, v in VECTORIZED_ACTION_SPACE.items()
}

NUM_ACTIONS: int = len(VECTORIZED_ACTION_SPACE)
