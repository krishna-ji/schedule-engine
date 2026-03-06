r"""Atomic vectorized repair actions for the RL hyper-heuristic.

Each class isolates **one** repair concern from the monolithic
``VectorizedRepair.repair_batch`` pipeline.  The RL agent picks one
action per generation; the Pymoo algorithm injects it as
``algorithm.mating.repair`` and calls ``algorithm.next()``.

Architecture
------------

All operators inherit from ``pymoo.core.repair.Repair`` and share a
common ``VectorizedRepair`` engine (loaded once via the pkl path).
The heavy precomputation (domain matrices, expansion arrays, boolean
availability tensors) happens **once** at construction; each
``_do`` call is a thin, purely-vectorised NumPy kernel.

Action registry
---------------

``VECTORIZED_ACTION_SPACE`` maps integer action IDs to classes::

    {0: ActionRepairRoomClash,
     1: ActionRepairInstructorClash,
     2: ActionSyncSSCP,
     3: ActionRandomPerturb,
     4: ActionRepairGroupClash,
     5: ActionFullRepair}

Complexity
----------

Each atomic action is $O(N \cdot Q)$ or better.  ``ActionFullRepair``
runs the original three-stage pipeline as a fallback.
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
    """Base for all atomic RL actions.

    Subclasses override ``_apply`` which receives the engine and X.
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
# Action 0 — Repair Room Clashes (SRE)
# ======================================================================


class ActionRepairRoomClash(_AtomicRepairBase):
    r"""Resolve Spatial Resource Exclusivity (room double-booking).

    Detects room double-booking via ``np.bincount`` on linearised
    room-time keys $k = n \cdot R \cdot T + r \cdot T + q$.
    Conflicting events get their **room** resampled from
    $\mathcal{D}_e^{\text{room}}$.

    Domain domains are *always* fixed first to avoid sampling from
    invalid indices.
    """

    ACTION_NAME: ClassVar[str] = "repair_room_clash"

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        T_ = __import__("src.pipeline.bitset_time", fromlist=["T"]).T

        # Fix domains first (prerequisite)
        eng._fix_domains_vec(X)

        inst = np.clip(X[:, 0::3], 0, eng.n_instructors - 1).astype(np.int64)
        room = np.clip(X[:, 1::3], 0, eng.n_rooms - 1).astype(np.int64)
        time = X[:, 2::3].astype(np.int64)

        n_idx = np.arange(N, dtype=np.int64)[:, None]

        # Expand to quantum level using engine's expansion arrays
        starts_exp = time[:, eng.exp_event]
        quanta_exp = np.clip(starts_exp + eng.exp_offset[None, :], 0, T_ - 1)
        rooms_exp = room[:, eng.exp_event]
        event_lin = (n_idx * E + eng.exp_event[None, :]).ravel()
        NE = N * E

        # Room occupancy histogram
        nRT = np.int64(eng.n_rooms) * np.int64(T_)
        room_keys = (n_idx * nRT + rooms_exp * T_ + quanta_exp).ravel()
        room_cnt = np.bincount(room_keys, minlength=int(N * nRT))
        room_conflict = (room_cnt[room_keys] > 1).astype(np.float64)

        # Aggregate to per-event score
        scores = np.bincount(event_lin, weights=room_conflict, minlength=NE)
        scores = scores[:NE].reshape(N, E)

        conflict_mask = scores > 0
        if not conflict_mask.any():
            return

        rng = np.random.default_rng()

        # Resample room for conflicting events
        bi, be = np.nonzero(conflict_mask)
        r_dl = eng.room_dom_len[be]
        r_valid = r_dl > 0
        bi, be, r_dl = bi[r_valid], be[r_valid], r_dl[r_valid]
        r_idx = (rng.random(len(bi)) * r_dl).astype(np.int64)
        r_idx = np.minimum(r_idx, r_dl - 1)
        X[bi, 3 * be + 1] = eng.room_domains[be, r_idx]

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
            "ActionRepairRoomClash: %d/%d events resampled",
            int(conflict_mask.sum()),
            N * E,
        )


# ======================================================================
# Action 1 — Repair Instructor Clashes (FTE)
# ======================================================================


class ActionRepairInstructorClash(_AtomicRepairBase):
    r"""Resolve Faculty Temporal Exclusivity (instructor double-booking).

    Detects instructor double-booking via ``np.bincount`` on linearised
    instructor-time keys.  Conflicting events get their **instructor**
    (and optionally time) resampled.
    """

    ACTION_NAME: ClassVar[str] = "repair_instructor_clash"

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

        # Instructor occupancy histogram
        nIT = np.int64(eng.n_instructors) * np.int64(T_)
        inst_keys = (n_idx * nIT + insts_exp * T_ + quanta_exp).ravel()
        inst_cnt = np.bincount(inst_keys, minlength=int(N * nIT))
        inst_conflict = (inst_cnt[inst_keys] > 1).astype(np.float64)

        # Availability violations
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

        # Also resample time for all conflicting events
        t_dl = eng.time_dom_len[be]
        t_valid = t_dl > 0
        t_bi, t_be, t_dl_v = bi[t_valid], be[t_valid], t_dl[t_valid]
        t_idx = (rng.random(len(t_bi)) * t_dl_v).astype(np.int64)
        t_idx = np.minimum(t_idx, t_dl_v - 1)
        X[t_bi, 3 * t_be + 2] = eng.time_domains[t_be, t_idx]

        logger.debug(
            "ActionRepairInstructorClash: %d/%d events resampled",
            int(conflict_mask.sum()),
            N * E,
        )


# ======================================================================
# Action 2 — Sync SSCP (Paired-Event Synchronisation)
# ======================================================================


class ActionSyncSSCP(_AtomicRepairBase):
    r"""Enforce Symmetric Sub-Cohort Parallelism.

    For each pair $(a, b)$, forces
    $t_a = t_b \in \mathcal{T}_a \cap \mathcal{T}_b$ and $r_a \neq r_b$.

    Delegates to ``VectorizedRepair._sync_paired_events``.
    """

    ACTION_NAME: ClassVar[str] = "sync_sscp"

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        eng._fix_domains_vec(X)
        if eng._n_pairs > 0:
            eng._sync_paired_events(X)
        logger.debug("ActionSyncSSCP: %d pairs synced", eng._n_pairs)


# ======================================================================
# Action 3 — Random Perturbation (Exploration)
# ======================================================================


class ActionRandomPerturb(_AtomicRepairBase):
    r"""Vectorized random perturbation (mutation for escaping local optima).

    Randomly selects $\sim 5\%$ of $(N, E)$ assignments and resamples
    **all three genes** (instructor, room, time) from their domains.
    Acts as pure exploration pressure — no conflict detection is
    performed.
    """

    ACTION_NAME: ClassVar[str] = "random_perturb"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        perturb_rate: float = 0.05,
    ):
        super().__init__(pkl_path)
        self.perturb_rate = perturb_rate

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        rng = np.random.default_rng()

        # Select ~perturb_rate of (individual, event) assignments
        mask = rng.random((N, E)) < self.perturb_rate
        if not mask.any():
            return

        bi, be = np.nonzero(mask)

        # Resample instructor
        i_dl = eng.inst_dom_len[be]
        i_v = i_dl > 0
        if i_v.any():
            i_bi, i_be, i_dl_v = bi[i_v], be[i_v], i_dl[i_v]
            i_idx = (rng.random(len(i_bi)) * i_dl_v).astype(np.int64)
            i_idx = np.minimum(i_idx, i_dl_v - 1)
            X[i_bi, 3 * i_be] = eng.inst_domains[i_be, i_idx]

        # Resample room
        r_dl = eng.room_dom_len[be]
        r_v = r_dl > 0
        if r_v.any():
            r_bi, r_be, r_dl_v = bi[r_v], be[r_v], r_dl[r_v]
            r_idx = (rng.random(len(r_bi)) * r_dl_v).astype(np.int64)
            r_idx = np.minimum(r_idx, r_dl_v - 1)
            X[r_bi, 3 * r_be + 1] = eng.room_domains[r_be, r_idx]

        # Resample time
        t_dl = eng.time_dom_len[be]
        t_v = t_dl > 0
        if t_v.any():
            t_bi, t_be, t_dl_v = bi[t_v], be[t_v], t_dl[t_v]
            t_idx = (rng.random(len(t_bi)) * t_dl_v).astype(np.int64)
            t_idx = np.minimum(t_idx, t_dl_v - 1)
            X[t_bi, 3 * t_be + 2] = eng.time_domains[t_be, t_idx]

        logger.debug(
            "ActionRandomPerturb: %d/%d assignments perturbed (rate=%.2f)",
            int(mask.sum()),
            N * E,
            self.perturb_rate,
        )


# ======================================================================
# Action 4 — Repair Group Clashes (CTE)
# ======================================================================


class ActionRepairGroupClash(_AtomicRepairBase):
    r"""Resolve Cohort Temporal Exclusivity (group double-booking).

    Detects group double-booking via ``np.bincount`` on linearised
    group-time keys.  Conflicting events get their **time** resampled.
    """

    ACTION_NAME: ClassVar[str] = "repair_group_clash"

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        T_ = __import__("src.pipeline.bitset_time", fromlist=["T"]).T

        eng._fix_domains_vec(X)

        time = X[:, 2::3].astype(np.int64)
        n_idx = np.arange(N, dtype=np.int64)[:, None]

        # Group-expanded quanta
        grp_starts = time[:, eng.grp_exp_event]
        grp_quanta = np.clip(grp_starts + eng.grp_exp_offset[None, :], 0, T_ - 1)

        nGT = np.int64(eng.n_groups) * np.int64(T_)
        grp_keys = (
            n_idx * nGT + eng.grp_exp_group[None, :].astype(np.int64) * T_ + grp_quanta
        ).ravel()
        grp_cnt = np.bincount(grp_keys, minlength=int(N * nGT))
        grp_conflict = (grp_cnt[grp_keys] > 1).astype(np.float64)

        # Aggregate to per-event via group expansion event indices
        grp_event_lin = (n_idx * E + eng.grp_exp_event[None, :]).ravel()
        NE = N * E
        scores = np.bincount(grp_event_lin, weights=grp_conflict, minlength=NE)
        scores = scores[:NE].reshape(N, E)

        conflict_mask = scores > 0
        if not conflict_mask.any():
            return

        rng = np.random.default_rng()
        bi, be = np.nonzero(conflict_mask)

        # Resample time for conflicting events
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
            "ActionRepairGroupClash: %d/%d events resampled",
            int(conflict_mask.sum()),
            N * E,
        )


# ======================================================================
# Action 5 — Full Pipeline Repair (Fallback)
# ======================================================================


class ActionFullRepair(_AtomicRepairBase):
    r"""Execute the complete three-stage repair pipeline.

    Delegates to ``VectorizedRepair.repair_batch`` with configurable
    number of passes.  Serves as a *safe* fallback action that the RL
    agent can select when the population is heavily infeasible.
    """

    ACTION_NAME: ClassVar[str] = "full_repair"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        passes: int = 3,
    ):
        super().__init__(pkl_path)
        self.passes = passes

    def _apply(self, X: np.ndarray) -> None:
        result = self.engine.repair_batch(X, passes=self.passes)
        X[:] = result
        logger.debug("ActionFullRepair: %d passes", self.passes)


# ======================================================================
# Action Space Registry — Elite 8
# ======================================================================

# Repairs (hard-constraint feasibility projections)
# Optimizations (soft-constraint quality-of-life)

# Perturbations (stochastic neighbourhood exploration)

# ======================================================================
# Elite 8 — Full Micro-Memetic Action Space (RESTORED)
# ======================================================================
# Imports from academic-taxonomy modules with micro-memetic upgrades

from src.rl.actions.optimizations.meridian_compaction import MeridianCompactionHeuristic
from src.rl.actions.perturbations.stochastic_quanta_perturbation import (
    StochasticQuantaPerturbation,
)
from src.rl.actions.perturbations.stochastic_spatial_perturbation import (
    StochasticSpatialPerturbation,
)
from src.rl.actions.repairs.cohort_temporal_projection import CohortTemporalProjection
from src.rl.actions.repairs.faculty_temporal_projection import FacultyTemporalProjection
from src.rl.actions.repairs.spatial_resource_projection import SpatialResourceProjection
from src.rl.actions.repairs.symmetric_subcohort_sync import SymmetricSubcohortSync
from src.rl.actions.repairs.universal_feasibility_projection import (
    UniversalFeasibilityProjection,
)

VECTORIZED_ACTION_SPACE: dict[int, type[_AtomicRepairBase]] = {
    0: SpatialResourceProjection,  # Conflict-directed room sniper (micro-memetic)
    1: FacultyTemporalProjection,  # Instructor clash repair
    2: CohortTemporalProjection,  # Group clash repair
    3: SymmetricSubcohortSync,  # SSCP paired-practical sync (soft)
    4: (
        UniversalFeasibilityProjection
    ),  # Bounded ejection chain bulldozer (micro-memetic)
    5: StochasticQuantaPerturbation,  # Time-slot exploration
    6: StochasticSpatialPerturbation,  # Room exploration
    7: MeridianCompactionHeuristic,  # Feasibility-gated soft optimizer (soft)
}

ACTION_NAMES: dict[int, str] = {
    k: v.ACTION_NAME for k, v in VECTORIZED_ACTION_SPACE.items()
}

NUM_ACTIONS: int = len(VECTORIZED_ACTION_SPACE)
