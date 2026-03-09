r"""Pipeline-configuration LLHs for the RL hyper-heuristic.

**Phase 55b fix** — two-phase repair architecture mirroring the
memetic GA (``ga_02_memetic.py``):

Phase 1 — Mating repair (on OFFSPRING, inside ``algorithm.next()``):
    Fast vectorized domain fix + SSCP sync.  Applied to ALL offspring.
    No BitsetRepair here — offspring are too many and too raw.

Phase 2 — Post-generation repair (on SURVIVORS, after ``algorithm.next()``):
    ``BitsetSchedulingRepair`` on the BEST K% of the surviving
    population (lowest hard penalty).  This directly improves the
    elite individuals, mirroring the memetic GA's callback.

Root cause of Phase 55a failure: BitsetRepair was in the mating
pipeline (offspring only), never touching surviving parents.  The
memetic GA achieves hard=72 because its callback repairs SURVIVING
elites.  Moving BitsetRepair to post-gen on survivors reproduces
this: hard < 100 at gen 2, stabilising at ~68-72.

Pipeline per LLH action::

    Mating (all offspring):
        1. Domain fix (vectorized)              — O(N·E), ~1ms
        2. SSCP paired-event sync               — O(N·P), ~1ms

    Post-gen (BEST K% survivors):
        3. BitsetRepair (greedy cost-minimising) — O(K·E²), ~K·400ms

Different LLH configurations vary K%, number of BitsetRepair passes,
deterministic vs stochastic, and optional mating-level kernels.

    0: ConservativeRepair   — 10% best, 2 passes alt
    1: AggressiveRepair     — 25% best, 3 passes alt
    2: MemeticEliteRepair   — 15% best, 4 passes alt (memetic clone)
    3: SoftFocusRepair      — 8% best, 2 passes + time compaction
    4: DestructiveConstructive — ruin events + 20% best, 2 passes
    5: IntensifiedRepair    — 20% best, 3 passes deterministic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from pymoo.core.repair import Repair

from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair
from src.pipeline.repair_operator_vectorized import VectorizedRepair

logger = logging.getLogger(__name__)


# ======================================================================
# Post-generation repair parameters (per LLH action)
# ======================================================================


@dataclass(frozen=True)
class PostGenConfig:
    """Parameters for post-generation BitsetRepair on survivors."""

    elite_fraction: float = 0.15  # Fraction of BEST survivors to repair
    passes: int = 4  # Number of BitsetRepair passes
    stochastic_alternate: bool = True  # Alternate det/stoch passes
    # Optional mating-level extras (domain fix + SSCP sync always run)
    ruin_fraction: float = 0.0  # Fraction of events to ruin before repair
    compact_soft: bool = False  # Time compaction for soft objectives


# ======================================================================
# Shared engine caches — avoid re-loading pkl per action
# ======================================================================

_VEC_ENGINE_CACHE: dict[str, VectorizedRepair] = {}
_BITSET_ENGINE_CACHE: dict[str, BitsetSchedulingRepair] = {}


def _get_vec_engine(pkl_path: str) -> VectorizedRepair:
    """Return (or create & cache) a ``VectorizedRepair`` engine."""
    if pkl_path not in _VEC_ENGINE_CACHE:
        # Auto-build the pkl if it doesn't exist yet
        from src.pipeline.build_events import ensure_pkl

        ensure_pkl(pkl_path)
        _VEC_ENGINE_CACHE[pkl_path] = VectorizedRepair(pkl_path)
    return _VEC_ENGINE_CACHE[pkl_path]


def _get_bitset_engine(pkl_path: str) -> BitsetSchedulingRepair:
    """Return (or create & cache) a ``BitsetSchedulingRepair`` engine."""
    if pkl_path not in _BITSET_ENGINE_CACHE:
        from src.pipeline.build_events import ensure_pkl

        ensure_pkl(pkl_path)
        _BITSET_ENGINE_CACHE[pkl_path] = BitsetSchedulingRepair(pkl_path)
    return _BITSET_ENGINE_CACHE[pkl_path]


# ======================================================================
# Base class
# ======================================================================


class _AtomicRepairBase(Repair):
    """Base for all pipeline-configuration LLH actions.

    Phase 55b: two-phase architecture:
    - ``_do()`` (mating repair): fast domain fix + SSCP sync on ALL offspring
    - ``POST_GEN``: parameters for post-gen BitsetRepair on BEST survivors
      (applied by ``env.step()`` after ``algorithm.next()``)

    Subclasses override ``_apply_mating`` for any mating-level extras
    (e.g., ruin & recreate, time compaction) and set ``POST_GEN`` for
    post-gen repair intensity.
    """

    ACTION_NAME: ClassVar[str] = "base"
    POST_GEN: ClassVar[PostGenConfig] = PostGenConfig()

    def __init__(self, pkl_path: str = ".cache/events_with_domains.pkl"):
        super().__init__()
        self.vec_engine: VectorizedRepair = _get_vec_engine(pkl_path)
        self.bitset_engine: BitsetSchedulingRepair = _get_bitset_engine(pkl_path)
        self._pkl_path = pkl_path

    def _do(self, problem, x, **kwargs):
        """Mating repair: domain fix + SSCP sync + optional extras."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = x.copy().astype(np.int64)

        eng = self.vec_engine

        # Stage 1: fast domain fix (vectorized over full pop)
        eng._fix_domains_vec(x)

        # Stage 2: subclass-specific mating extras (default: nothing)
        self._apply_mating(x)

        # Stage 3: SSCP sync
        if eng._n_pairs > 0:
            eng._sync_paired_events(x)

        return x

    def _apply_mating(self, X: np.ndarray) -> None:
        """Optional mating-level extras.  Override in subclasses."""


# ======================================================================
# Action 0 — Conservative Repair (exploit)
# ======================================================================


class ConservativeRepair(_AtomicRepairBase):
    r"""Domain fix + post-gen BitsetRepair on best 10% — steady exploitation.

    Minimal computational budget per generation.  Good schemata are
    preserved; only the best (near-feasible) survivors get polished.
    ~10% × 120 = 12 individuals × 400ms × 2 passes ≈ 9.6s/gen.
    """

    ACTION_NAME: ClassVar[str] = "conservative_repair"
    POST_GEN: ClassVar[PostGenConfig] = PostGenConfig(
        elite_fraction=0.10,
        passes=2,
        stochastic_alternate=True,
    )


# ======================================================================
# Action 1 — Aggressive Repair (explore)
# ======================================================================


class AggressiveRepair(_AtomicRepairBase):
    r"""Domain fix + post-gen BitsetRepair on best 25% — aggressive exploration.

    High computational budget: quarter of survivors get 3-pass repair
    with alternating deterministic/stochastic placement.  Use when
    stuck in a penalty plateau.
    ~25% × 120 = 30 × 400ms × 3 ≈ 36s/gen.
    """

    ACTION_NAME: ClassVar[str] = "aggressive_repair"
    POST_GEN: ClassVar[PostGenConfig] = PostGenConfig(
        elite_fraction=0.25,
        passes=3,
        stochastic_alternate=True,
    )


# ======================================================================
# Action 2 — Memetic Elite Repair
# ======================================================================


class MemeticEliteRepair(_AtomicRepairBase):
    r"""Domain fix + post-gen BitsetRepair on best 15% — memetic GA clone.

    Directly mirrors ``ga_02_memetic.py``'s callback: repair the top
    15% of survivors with 4 alternating passes.  This is the
    configuration that achieves hard=72 at gen 6 in the memetic GA.
    ~15% × 120 = 18 × 400ms × 4 ≈ 28.8s/gen.
    """

    ACTION_NAME: ClassVar[str] = "memetic_elite_repair"
    POST_GEN: ClassVar[PostGenConfig] = PostGenConfig(
        elite_fraction=0.15,
        passes=4,
        stochastic_alternate=True,
    )


# ======================================================================
# Action 3 — Soft-Focus Repair
# ======================================================================


class SoftFocusRepair(_AtomicRepairBase):
    r"""Domain fix + BitsetRepair on best 8% + time compaction.

    Post-gen: repair top 8% (conservative hard-fix budget).
    Mating extra: time compaction on conflict-free events toward earlier
    quanta (reduces gaps → better soft constraints).

    The RL agent should select this when hard penalties are low.
    """

    ACTION_NAME: ClassVar[str] = "soft_focus_repair"
    POST_GEN: ClassVar[PostGenConfig] = PostGenConfig(
        elite_fraction=0.08,
        passes=2,
        stochastic_alternate=True,
        compact_soft=True,
    )

    def _apply_mating(self, X: np.ndarray) -> None:
        """Time compaction: move conflict-free events to earlier quanta."""
        eng = self.vec_engine
        N, E = X.shape[0], eng.n_events
        rng = np.random.default_rng()

        scores = eng._score_all_batch(X)  # (N, E)
        clean_mask = scores == 0
        compact_mask = clean_mask & (rng.random((N, E)) < 0.20)

        if compact_mask.any():
            bi, be = np.nonzero(compact_mask)
            t_dl = eng.time_dom_len[be]
            t_valid = t_dl > 0
            bi, be, t_dl = bi[t_valid], be[t_valid], t_dl[t_valid]

            for k in range(len(bi)):
                n, e = int(bi[k]), int(be[k])
                dl = int(t_dl[k])
                domain = eng.time_domains[e, :dl]
                cur_t = int(X[n, 3 * e + 2])
                earlier = domain[domain < cur_t]
                if len(earlier) > 0:
                    X[n, 3 * e + 2] = int(earlier[-1])


# ======================================================================
# Action 4 — Destructive-Constructive (Ruin & Recreate)
# ======================================================================


class DestructiveConstructive(_AtomicRepairBase):
    r"""Ruin worst 10% of events per individual, then post-gen BitsetRepair.

    Mating extra: score per-event conflict severity, "ruin" the top
    10% worst events by randomising all three genes.
    Post-gen: BitsetRepair on best 20% with 2 passes.

    This is the **escape hatch** for local-optima plateaus.
    """

    ACTION_NAME: ClassVar[str] = "destructive_constructive"
    POST_GEN: ClassVar[PostGenConfig] = PostGenConfig(
        elite_fraction=0.20,
        passes=2,
        stochastic_alternate=True,
        ruin_fraction=0.10,
    )

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        ruin_fraction: float = 0.10,
    ):
        super().__init__(pkl_path)
        self.ruin_fraction = ruin_fraction

    def _apply_mating(self, X: np.ndarray) -> None:
        """Ruin worst events per individual."""
        eng = self.vec_engine
        N, E = X.shape[0], eng.n_events
        rng = np.random.default_rng()

        scores = eng._score_all_batch(X)
        n_ruin = max(1, int(E * self.ruin_fraction))

        for n in range(N):
            row_scores = scores[n]
            if row_scores.sum() == 0:
                continue
            worst = np.argsort(-row_scores)[:n_ruin]
            has_conflict = row_scores[worst] > 0
            worst = worst[has_conflict]
            if len(worst) == 0:
                continue
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


# ======================================================================
# Action 5 — Intensified Repair (balanced)
# ======================================================================


class IntensifiedRepair(_AtomicRepairBase):
    r"""Domain fix + post-gen BitsetRepair on best 20% — balanced workhorse.

    Three alternating passes on 20% of survivors.  More coverage
    than Conservative (10%, 2 pass) with moderate stochastic
    exploration.  Sits between Conservative and Aggressive in both
    repair budget and convergence speed.
    """

    ACTION_NAME: ClassVar[str] = "intensified_repair"
    POST_GEN: ClassVar[PostGenConfig] = PostGenConfig(
        elite_fraction=0.20,
        passes=3,
        stochastic_alternate=True,
    )


# ======================================================================
# Action Space Registry — Pipeline Configurations (Phase 55b)
# ======================================================================

VECTORIZED_ACTION_SPACE: dict[int, type[_AtomicRepairBase]] = {
    0: ConservativeRepair,  # 10% best, 2 passes alt
    1: AggressiveRepair,  # 25% best, 3 passes alt
    2: MemeticEliteRepair,  # 15% best, 4 passes alt (memetic clone)
    3: SoftFocusRepair,  # 8% best, 2 passes + compaction
    4: DestructiveConstructive,  # Ruin events + 20% best, 2 passes
    5: IntensifiedRepair,  # 20% best, 3 passes alt
}

ACTION_NAMES: dict[int, str] = {
    k: v.ACTION_NAME for k, v in VECTORIZED_ACTION_SPACE.items()
}

NUM_ACTIONS: int = len(VECTORIZED_ACTION_SPACE)
