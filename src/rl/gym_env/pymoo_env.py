r"""Pymoo-native Gymnasium environment for RL hyper-heuristic control.

The RL agent selects **which pipeline configuration** (action) is
injected into the Pymoo algorithm each generation.  One ``step()``
$\equiv$ one call to ``algorithm.next()``.

**Phase 53 redesign**: each action runs the COMPLETE repair pipeline
(``repair_batch``) with different parameters.  No rollback is needed
because every action produces a coherent, fully-repaired population.

Key design invariants
---------------------

1. **Zero DEAP** — everything operates on ``pop.F``, ``pop.G``,
   ``pop.X`` NumPy matrices.
2. **No per-individual loops** — state extraction is $O(1)$ matrix ops
   (via ``VectorizedStateEncoder``).
3. **Composable** — any ``pymoo.core.repair.Repair`` can be hot-swapped
   into ``algorithm.mating.repair`` at each step.
4. **No rollback** — all LLHs run the complete pipeline, so no action
   is destructive.  The lexicographic safety rail is removed.

Usage
-----

.. code-block:: python

    import gymnasium as gym
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    env = PymooHyperHeuristicEnv(
        pkl_path=".cache/events_with_domains.pkl",
        max_generations=300,
        pop_size=100,
    )
    obs, info = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.rl.actions.vectorized_ops import (
    NUM_ACTIONS,
    VECTORIZED_ACTION_SPACE,
    PostGenConfig,
    _AtomicRepairBase,
)
from src.rl.gym_env.fast_state_encoder import OBS_DIM, VectorizedStateEncoder

logger = logging.getLogger(__name__)


class PymooHyperHeuristicEnv(gym.Env):
    r"""Gymnasium environment that wraps a Pymoo scheduling algorithm.

    **Observation** (39-D Box $[0,1]$):
        Fitness stats, constraint violations, diversity, progress,
        heuristic history — all extracted via NumPy matrix ops on
        ``pop.F``, ``pop.G``, ``pop.X``.

    **Actions** (Discrete):
        Integer indexing into ``VECTORIZED_ACTION_SPACE``.  Each action
        is a complete pipeline configuration (``repair_batch`` with
        different passes + optional intensification/diversification).

    **Reward**:
        Pure delta-based: hard-penalty improvement + 0.1 × soft-penalty
        improvement + one-time feasibility bonus.  No time penalty.

    **Termination**:
        ``done = True`` when the best individual is fully feasible
        (``F[best, 0] == 0``) or when ``max_generations`` is reached.

    Parameters
    ----------
    pkl_path : str
        Path to ``events_with_domains.pkl``.
    max_generations : int
        Episode budget.
    pop_size : int
        Pymoo population size.
    algorithm_name : str
        ``"nsga2"`` or ``"ga"``.
    seed : int
        Random seed for the Pymoo algorithm.
    reward_scale : float
        Multiplicative scale for the reward signal.
    feasibility_bonus : float
        One-time bonus added when first feasible solution is found.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        max_generations: int = 300,
        pop_size: int = 100,
        n_offsprings: int | None = None,
        algorithm_name: str = "nsga2",
        seed: int = 42,
        reward_scale: float = 1.0,
        feasibility_bonus: float = 10.0,
        acceptance_tolerance: float = 0.0,  # DEPRECATED (Phase 53: no rollback)
    ):
        super().__init__()

        self.pkl_path = pkl_path
        self.max_generations = max(max_generations, 1)
        self.pop_size = pop_size
        self.n_offsprings = n_offsprings or pop_size
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.reward_scale = reward_scale
        self.feasibility_bonus = feasibility_bonus
        # acceptance_tolerance silently ignored — retained for backward compat

        # -- Gym spaces ---------------------------------------------------
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # -- Pre-instantiate action operators (shared engine) -------------
        self._action_operators: dict[int, _AtomicRepairBase] = {
            aid: cls(pkl_path) for aid, cls in VECTORIZED_ACTION_SPACE.items()
        }

        # -- State encoder ------------------------------------------------
        self._encoder = VectorizedStateEncoder(
            max_generations=max_generations,
        )

        # -- Pymoo objects (initialised in reset()) -----------------------
        # Cache the SchedulingProblem across resets — it's stateless
        # (evaluations are pure functions on population data) and
        # recreating it triggers expensive DataStore.from_json() +
        # feasibility checks every episode.
        self._problem = None
        self._problem_cached = False
        self._algorithm = None
        self._gen: int = 0
        self._prev_best_hard: float = np.inf
        self._prev_best_soft: float = np.inf
        self._found_feasible: bool = False
        self._episode_start: float = 0.0

        # -- State-Conditioned Action Masking -----------------------------
        self._current_best_hard: float = np.inf  # Track for action masking

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment: create a fresh problem + algorithm.

        Returns
        -------
        obs : ndarray(39,)
        info : dict
        """
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self.seed

        # Lazy imports to keep module-level fast
        from src.pipeline.pymoo_operators import (
            EventBlockCrossover,
            EventLocalMutation,
            RandomDomainSampling,
        )

        # Cache the SchedulingProblem across resets — it's stateless
        # (pure function evaluator on population data).  Avoids
        # DataStore.from_json() + feasibility checks every episode.
        if not self._problem_cached:
            from src.pipeline.scheduling_problem import create_problem

            self._problem = create_problem(self.pkl_path)
            self._problem_cached = True

        self._encoder = VectorizedStateEncoder(
            max_generations=self.max_generations,
            n_events=self._problem.spec.n_events,
        )
        self._encoder.reset()

        # Create algorithm — start with conservative (safe default)
        default_repair = self._action_operators[0]  # ConservativeRepair

        if self.algorithm_name.lower() == "nsga2":
            from pymoo.algorithms.moo.nsga2 import NSGA2

            self._algorithm = NSGA2(
                pop_size=self.pop_size,
                n_offsprings=self.n_offsprings,
                sampling=RandomDomainSampling(self.pkl_path),
                crossover=EventBlockCrossover(prob=0.5),
                mutation=EventLocalMutation(pkl_path=self.pkl_path, event_prob=0.05),
                repair=default_repair,
                seed=effective_seed,
            )
        else:
            from pymoo.algorithms.soo.nonconvex.ga import GA

            self._algorithm = GA(
                pop_size=self.pop_size,
                n_offsprings=self.n_offsprings,
                sampling=RandomDomainSampling(self.pkl_path),
                crossover=EventBlockCrossover(prob=0.5),
                mutation=EventLocalMutation(pkl_path=self.pkl_path, event_prob=0.05),
                repair=default_repair,
                seed=effective_seed,
            )

        # Setup the algorithm with the problem
        self._algorithm.setup(
            self._problem, termination=("n_gen", self.max_generations)
        )

        # Run the first generation to initialise the population
        self._algorithm.next()

        # Extract initial state
        self._gen = 1
        self._prev_best_hard = np.inf
        self._prev_best_soft = np.inf
        self._found_feasible = False
        self._episode_start = _time.perf_counter()

        pop = self._algorithm.pop
        F, G, X = self._extract_pop(pop)
        soft_bd = getattr(self._problem, "_last_soft_breakdown", None)

        obs = self._encoder.encode(F, G, X, soft_breakdown=soft_bd)

        self._prev_best_hard = float(F[:, 0].min())
        self._prev_best_soft = float(F[:, 1].min())
        self._current_best_hard = self._prev_best_hard  # Update for masking

        info = self._build_info(F, G)
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one generation with the selected pipeline configuration.

        **Phase 55b**: two-phase repair architecture:
        1. Mating repair (domain fix + SSCP sync) on offspring
        2. Post-gen BitsetRepair on BEST K% of surviving population

        Parameters
        ----------
        action : int
            Index into ``VECTORIZED_ACTION_SPACE``.

        Returns
        -------
        obs : ndarray(39,)
        reward : float
        terminated : bool  (fully feasible found)
        truncated : bool   (max generations reached)
        info : dict
        """
        assert self._algorithm is not None, "Call reset() first."

        # -- Inject the chosen mating repair (domain fix + SSCP) ---------
        operator = self._action_operators[action]
        self._algorithm.mating.repair = operator

        prev_best_hard = self._prev_best_hard
        prev_best_soft = self._prev_best_soft

        # -- Phase 1: Run one generational step (mating + survival) ------
        t0 = _time.perf_counter()
        self._algorithm.next()

        # -- Phase 2: Post-gen BitsetRepair on BEST survivors ------------
        cfg = operator.POST_GEN
        n_repaired = self._post_gen_repair(operator, cfg)

        step_time = _time.perf_counter() - t0

        self._gen += 1

        # -- Extract population state ------------------------------------
        pop = self._algorithm.pop
        F, G, X = self._extract_pop(pop)

        best_hard = float(F[:, 0].min())
        best_soft = float(F[:, 1].min())

        # -- Encode observation and compute reward -----------------------
        soft_bd = getattr(self._problem, "_last_soft_breakdown", None)
        obs = self._encoder.encode(F, G, X, soft_breakdown=soft_bd, action_taken=action)
        reward = self._compute_reward(F, G)

        # -- Update tracking ---------------------------------------------
        self._prev_best_hard = best_hard
        self._prev_best_soft = best_soft
        self._current_best_hard = best_hard

        delta_hard = best_hard - prev_best_hard
        delta_soft = best_soft - prev_best_soft

        logger.debug(
            "Gen %d | action=%d (%s) | hard=%.1f (Δ%.1f) | R=%.4f | %.2fs | repaired=%d",
            self._gen,
            action,
            operator.ACTION_NAME,
            best_hard,
            delta_hard,
            reward,
            step_time,
            n_repaired,
        )

        # -- Termination checks ------------------------------------------
        terminated = False  # Let episodes run full max_generations
        truncated = self._gen >= self.max_generations

        info = self._build_info(*self._extract_pop(self._algorithm.pop)[:2])
        info["step_time_s"] = step_time
        info["action"] = action
        info["action_name"] = operator.ACTION_NAME
        info["rejected"] = False  # No rollback in Phase 53
        info["delta_hard"] = delta_hard
        info["delta_soft"] = delta_soft
        info["n_repaired"] = n_repaired

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Post-generation BitsetRepair on survivors
    # ------------------------------------------------------------------

    def _post_gen_repair(
        self,
        operator: _AtomicRepairBase,
        cfg: PostGenConfig,
    ) -> int:
        """Apply BitsetRepair to the BEST K% of the surviving population.

        Mirrors the memetic GA's callback: select the elite survivors
        (lowest hard penalty), repair them with ``BitsetSchedulingRepair``,
        and force re-evaluation so NSGA-II sees updated fitness.

        Parameters
        ----------
        operator : _AtomicRepairBase
            The active LLH operator (provides ``bitset_engine``).
        cfg : PostGenConfig
            Post-gen repair parameters from the LLH action.

        Returns
        -------
        int
            Number of individuals repaired.
        """
        from pymoo.core.evaluator import Evaluator
        from pymoo.core.population import Population

        pop = self._algorithm.pop
        F = pop.get("F")
        hard_vals = F[:, 0]

        # Select BEST (lowest hard) individuals
        n_elite = max(1, int(len(pop) * cfg.elite_fraction))
        elite_idx = np.argsort(hard_vals)[:n_elite]
        # Only repair those with remaining violations
        elite_idx = elite_idx[hard_vals[elite_idx] > 0]

        if len(elite_idx) == 0:
            return 0

        repairer = operator.bitset_engine
        modified = []

        for idx in elite_idx:
            xi = pop[idx].get("X").copy()
            for p in range(cfg.passes):
                if cfg.stochastic_alternate and p % 2 == 0:
                    rng = np.random.default_rng()
                else:
                    rng = None
                xi_new = repairer.repair(xi, rng=rng)
                if np.array_equal(xi_new, xi):
                    break  # converged
                xi = xi_new

            pop[idx].set("X", xi)
            # Clear stale fitness for re-evaluation
            pop[idx].set("F", None)
            pop[idx].set("G", None)
            pop[idx].set("CV", None)
            for tag in ["F", "G", "CV"]:
                if tag in pop[idx].evaluated:
                    pop[idx].evaluated.remove(tag)
            modified.append(pop[idx])

        if modified:
            eval_pop = Population.create(*modified)
            Evaluator().eval(self._problem, eval_pop)

        return len(modified)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, F: np.ndarray, G: np.ndarray) -> float:
        r"""Pure delta-based reward from F and G matrices.

        .. math::

            R_t = \Delta_{\text{hard}} + 0.1 \cdot \Delta_{\text{soft}}
                  + \text{feasibility\_bonus}

        where $\Delta = \text{prev\_best} - \text{best}$ (positive means
        improvement).  No time penalty — the agent is rewarded strictly
        for **making things better**.
        """
        best_hard = float(F[:, 0].min())
        best_soft = float(F[:, 1].min())

        # -- Component 1: absolute hard-penalty improvement --------------
        delta_hard = self._prev_best_hard - best_hard  # >0 = got better
        # Normalise by initial magnitude so reward scale stays ~O(1)
        norm_hard = max(self._prev_best_hard, 1.0)
        hard_reward = delta_hard / norm_hard

        # -- Component 2: absolute soft-penalty improvement --------------
        delta_soft = self._prev_best_soft - best_soft
        norm_soft = max(self._prev_best_soft, 1.0)
        soft_reward = delta_soft / norm_soft

        # -- Component 3: one-time feasibility bonus ---------------------
        first_feasible_bonus = 0.0
        if best_hard == 0.0 and not self._found_feasible:
            first_feasible_bonus = self.feasibility_bonus
            self._found_feasible = True

        # -- Combine (no time penalty) ------------------------------------
        reward = (
            hard_reward + 0.1 * soft_reward + first_feasible_bonus
        ) * self.reward_scale

        # Clip to [-5, 5]  (feasibility bonus handled separately)
        return float(np.clip(reward, -5.0, 5.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pop(pop) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract (F, G, X) from a Pymoo Population object."""
        F = pop.get("F")  # (N, 2)
        G = pop.get("G")  # (N, 8)
        X = pop.get("X")  # (N, 3E)
        # Defensive: ensure G is never None
        if G is None:
            G = np.zeros((F.shape[0], 8), dtype=np.float64)
        return F, G, X

    def _build_info(self, F: np.ndarray, G: np.ndarray) -> dict[str, Any]:
        """Build the info dict returned by step/reset.

        Includes per-constraint violation means (8 hard + 4 soft) so
        that evaluation scripts can export a full constraint breakdown
        without re-parsing the observation vector.
        """
        total_viol = G.sum(axis=1)
        info: dict[str, Any] = {
            "generation": self._gen,
            "best_hard": float(F[:, 0].min()),
            "mean_hard": float(F[:, 0].mean()),
            "best_soft": float(F[:, 1].min()),
            "mean_soft": float(F[:, 1].mean()),
            "feasible_count": int((total_viol == 0).sum()),
            "feasible_frac": float((total_viol == 0).mean()),
            "pop_size": F.shape[0],
            "elapsed_s": _time.perf_counter() - self._episode_start,
        }

        # -- Per-hard-constraint mean violations (8 columns) -------------
        from src.rl.gym_env.fast_state_encoder import (
            HARD_CONSTRAINT_NAMES,
            SOFT_CONSTRAINT_NAMES,
        )

        for i, name in enumerate(HARD_CONSTRAINT_NAMES):
            if i < G.shape[1]:
                info[f"cv_{name}"] = float(G[:, i].mean())
            else:
                info[f"cv_{name}"] = 0.0

        # -- Per-soft-constraint mean penalties (4 soft) -----------------
        soft_bd = getattr(self._problem, "_last_soft_breakdown", None)
        for name in SOFT_CONSTRAINT_NAMES:
            if soft_bd and name in soft_bd:
                info[f"cv_{name}"] = float(np.asarray(soft_bd[name]).mean())
            else:
                info[f"cv_{name}"] = 0.0

        return info

    # ------------------------------------------------------------------
    # State-Conditioned Action Masking
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return boolean mask for valid actions based on current state.

        **Phase 53 action space** (6 pipeline configurations):

        - Action 3 (SoftFocusRepair) is masked out when hard constraints
          are violated — soft optimisation is wasteful when hard
          penalties dominate.
        - All other actions are always available since they all run
          the complete repair pipeline.

        Returns
        -------
        mask : ndarray(NUM_ACTIONS,), bool
            True means action is available.
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)

        # Block soft-focus optimiser when hard constraints are violated
        if self._current_best_hard > 0.0:
            mask[3] = False  # SoftFocusRepair (soft obj only)

        return mask

    def render(self) -> None:
        """No rendering — headless environment."""

    def close(self) -> None:
        """Clean up."""
        self._algorithm = None
        self._problem = None
