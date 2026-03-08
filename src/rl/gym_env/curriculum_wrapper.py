r"""Constraint-Curriculum Gym Wrapper with integrated PBRS.

Implements a three-phase training curriculum that gradually increases
the complexity of the reward signal seen by the RL agent, plus
Potential-Based Reward Shaping (PBRS) for dense gradients.

Design Decision — Reward-Weight Modulation (not Constraint Masking)
-------------------------------------------------------------------

The GA **always evaluates all 8 hard constraints** — its population
evolution depends on the full ``F[:, 0] = sum(G[:, strict_cols])``
fitness.  We cannot mask constraints inside
``SchedulingProblem._evaluate()`` without corrupting NSGA-II selection.

Instead, we modify the **RL reward** to emphasise different constraint
groups per phase.  The GA evolves normally; the agent learns which LLH
helps *the currently-rewarded constraint subset* most.

Curriculum Phases
-----------------

**Phase 1** (episodes 0 → ``phase1_end``):
    Reward only counts **Spatial** constraint improvements:
    SRE (Room Exclusivity), FFC (Facility Feature Congruence).
    Agent learns: "which LLH is good for fixing room conflicts?"

**Phase 2** (episodes ``phase1_end`` → ``phase2_end``):
    Add **Instructor** constraints to reward:
    FTE (Faculty Temporal Exclusivity), FPC (Faculty Pedagogical
    Congruence), FCA (Faculty Chronological Availability).
    Agent learns spatio-temporal resolution.

**Phase 3** (episodes > ``phase2_end``):
    All constraints active in reward.  Full NP-hard complexity.
    Adds CTE (Cohort Temporal Exclusivity), CQF (Curriculum Quanta
    Fulfillment), ICTD (Intra-Course Temporal Dispersion).

Smooth Blending
---------------

Phase transitions use a 5-episode linear blend to avoid reward
distribution discontinuities that could destabilise PPO.

PBRS Integration
----------------

Each ``step()`` also adds the PBRS term
:math:`\gamma \Phi(s') - \Phi(s)` from :class:`StatePotentialCalculator`.

Phase 62 — Titan V4 SOTA Algorithmic Overhaul
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from src.rl.gym_env.reward_shaper import StatePotentialCalculator

logger = logging.getLogger(__name__)


# ======================================================================
# Constraint column definitions (index into G matrix / info dict keys)
# ======================================================================

# Column name → G index mapping
_CONSTRAINT_INDEX = {
    "CTE": 0,  # Cohort Temporal Exclusivity     (Group)
    "FTE": 1,  # Faculty Temporal Exclusivity     (Instructor)
    "SRE": 2,  # Spatial Resource Exclusivity     (Room)
    "FPC": 3,  # Faculty Pedagogical Congruence   (Instructor)
    "FFC": 4,  # Facility Feature Congruence      (Room)
    "FCA": 5,  # Faculty Chronological Avail.     (Instructor)
    "CQF": 6,  # Curriculum Quanta Fulfillment    (Completeness)
    "ICTD": 7,  # Intra-Course Temporal Dispersion (Sibling)
}

ALL_CONSTRAINT_NAMES = list(_CONSTRAINT_INDEX.keys())

# Phase groupings (by resource type)
_SPATIAL_COLS = ["SRE", "FFC"]
_INSTRUCTOR_COLS = ["FTE", "FPC", "FCA"]
_GROUP_COLS = ["CTE", "CQF", "ICTD"]

_PHASE_1_COLS = frozenset(_SPATIAL_COLS)
_PHASE_2_COLS = frozenset(_SPATIAL_COLS + _INSTRUCTOR_COLS)
_PHASE_3_COLS = frozenset(ALL_CONSTRAINT_NAMES)

# Transition blend window (episodes)
_BLEND_WINDOW: int = 5


class ConstraintCurriculumWrapper(gym.Wrapper):
    r"""Gym Wrapper that applies curriculum reward shaping + PBRS.

    Wraps a :class:`PymooHyperHeuristicEnv` to:

    1. Compute per-constraint-column deltas from the ``info`` dict.
    2. Weight those deltas based on the current curriculum phase.
    3. Add PBRS term :math:`\gamma \Phi(s') - \Phi(s)`.
    4. Return ``shaped_reward = base_reward + curriculum_bonus + pbrs``.

    The wrapper maintains its own episode counter.  When used inside
    ``SubprocVecEnv``, each worker tracks episodes independently.

    Parameters
    ----------
    env : gym.Env
        Underlying PymooHyperHeuristicEnv.
    phase1_episodes : int
        Per-worker episodes for Phase 1 (spatial only).
    phase2_episodes : int
        Per-worker episodes for Phase 2 (spatial + instructor).
    gamma : float
        Discount factor for PBRS.
    curriculum_weight : float
        Scaling factor for the per-constraint curriculum bonus.
    use_chromosome_potential : bool
        If True, compute Tier 2 per-resource potential from X.
    """

    def __init__(
        self,
        env: gym.Env,
        phase1_episodes: int = 21,
        phase2_episodes: int = 63,
        gamma: float = 0.99,
        curriculum_weight: float = 0.5,
        use_chromosome_potential: bool = True,
    ):
        super().__init__(env)

        self.phase1_episodes = phase1_episodes
        self.phase2_episodes = phase2_episodes
        self.curriculum_weight = curriculum_weight

        # PBRS calculator
        self._shaper = StatePotentialCalculator(
            gamma=gamma,
            use_chromosome=use_chromosome_potential,
        )

        # Episode tracking (per-worker)
        self._episode_count: int = 0

        # PBRS state
        self._prev_phi: float = 0.0

        # Per-constraint tracking for curriculum bonus
        self._prev_cv: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Phase determination
    # ------------------------------------------------------------------

    def _get_phase(self) -> int:
        """Return current curriculum phase (1, 2, or 3)."""
        if self._episode_count <= self.phase1_episodes:
            return 1
        if self._episode_count <= self.phase2_episodes:
            return 2
        return 3

    def _active_columns(self, phase: int) -> frozenset[str]:
        """Return the set of constraint names active in the given phase."""
        if phase == 1:
            return _PHASE_1_COLS
        if phase == 2:
            return _PHASE_2_COLS
        return _PHASE_3_COLS

    def _blend_factor(self) -> float:
        r"""Smooth blending at phase transitions.

        Returns a factor in [0, 1] that linearly ramps up the new
        phase's constraint contributions over ``_BLEND_WINDOW`` episodes.
        """
        for boundary in [self.phase1_episodes, self.phase2_episodes]:
            dist = self._episode_count - boundary
            if 0 < dist <= _BLEND_WINDOW:
                return dist / _BLEND_WINDOW
        return 1.0

    # ------------------------------------------------------------------
    # Population access helpers
    # ------------------------------------------------------------------

    def _get_population(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract (F, G, X) from the underlying env's current population."""
        unwrapped = self.env.unwrapped
        pop = unwrapped._algorithm.pop
        return unwrapped._extract_pop(pop)

    def _get_vec_data(self):
        """Get VectorizedEvalData from the underlying problem."""
        unwrapped = self.env.unwrapped
        if unwrapped._problem is not None:
            return unwrapped._problem._vec_data
        return None

    # ------------------------------------------------------------------
    # Gym API overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment and initialise PBRS + curriculum state."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_count += 1

        # Compute initial potential Φ(s₀)
        try:
            F, G, X = self._get_population()
            vec_data = self._get_vec_data()
            self._prev_phi = self._shaper.potential(F, G, X, vec_data)
        except Exception:
            self._prev_phi = 0.0

        # Initialise per-constraint tracking
        self._prev_cv = {}
        for name in ALL_CONSTRAINT_NAMES:
            self._prev_cv[name] = info.get(f"cv_{name}", 0.0)

        # Augment info
        info["curriculum_phase"] = self._get_phase()
        info["episode_count"] = self._episode_count
        info["phi"] = self._prev_phi

        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Run one step with curriculum shaping + PBRS.

        Returns
        -------
        obs : ndarray(39,)
        shaped_reward : float
            ``base_reward + curriculum_bonus + PBRS_term``
        terminated : bool
        truncated : bool
        info : dict
            Augmented with ``original_reward``, ``pbrs``,
            ``curriculum_bonus``, ``curriculum_phase``, ``phi``.
        """
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # -- PBRS term ---------------------------------------------------
        pbrs = 0.0
        phi_new = self._prev_phi  # fallback
        try:
            F, G, X = self._get_population()
            vec_data = self._get_vec_data()
            phi_new = self._shaper.potential(F, G, X, vec_data)
            pbrs = self._shaper.shaping_reward(self._prev_phi, phi_new)
        except Exception as exc:
            logger.debug("PBRS computation failed: %s", exc)
        self._prev_phi = phi_new

        # -- Curriculum bonus --------------------------------------------
        phase = self._get_phase()
        active = self._active_columns(phase)
        blend = self._blend_factor()

        curriculum_bonus = 0.0
        for name in ALL_CONSTRAINT_NAMES:
            current_val = info.get(f"cv_{name}", 0.0)
            prev_val = self._prev_cv.get(name, current_val)

            # Delta: positive = improvement (violation decreased)
            delta = prev_val - current_val

            if name in active:
                # Active constraint — full weight
                curriculum_bonus += delta * self.curriculum_weight
            else:
                # Check if this constraint is in the NEXT phase's set
                # and apply blended weight during transition
                next_phase = min(phase + 1, 3)
                next_active = self._active_columns(next_phase)
                if name in next_active and blend < 1.0:
                    # Transition blending: ramp up from 0 to full
                    curriculum_bonus += delta * self.curriculum_weight * blend

            self._prev_cv[name] = current_val

        # -- Combine -----------------------------------------------------
        shaped_reward = base_reward + pbrs + curriculum_bonus

        # Clip to [-15, 15] (wider than base ±10 to accommodate shaping)
        shaped_reward = float(np.clip(shaped_reward, -15.0, 15.0))

        # -- Augment info ------------------------------------------------
        info["original_reward"] = base_reward
        info["pbrs"] = pbrs
        info["curriculum_bonus"] = curriculum_bonus
        info["shaped_reward"] = shaped_reward
        info["curriculum_phase"] = phase
        info["episode_count"] = self._episode_count
        info["phi"] = phi_new

        return obs, shaped_reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Delegate action masks to the underlying environment."""
        return self.env.unwrapped.action_masks()
