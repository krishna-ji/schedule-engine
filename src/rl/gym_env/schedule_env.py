"""
Gymnasium environment for schedule optimization.

Wraps the GA scheduler as an RL environment where the agent learns to select
effective heuristics at each step.
"""

import copy
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces

from src.core.types import Individual, SchedulingContext
from src.rl.gym_env.state_encoder import StateEncoder
from src.rl.gym_env.action_space import ActionMapper
from src.rl.gym_env.reward_calculator import RewardCalculator


logger = logging.getLogger(__name__)


class ScheduleEnv(gym.Env):
    """
    Gymnasium environment for schedule optimization.

    Observation Space:
        Box(39,) - normalized features [0, 1]:
        - Fitness metrics (5): best, avg, worst, std, range
        - Diversity metrics (5): population, genotype, phenotype, fitness, unique_ratio
        - Progress metrics (4): generation, stagnation, convergence, improvement
        - Violation metrics (3): hard, soft, std
        - ENHANCEMENT #2: Per-constraint breakdown (12): 8 hard + 4 soft constraints
        - Heuristic history (10): recent heuristic applications

    Action Space:
        Discrete(20) - heuristic selection:
        - 0: No-op
        - 1-19: Heuristic operators

    Reward:
        Continuous [-1, 1] - fitness improvement + diversity - time penalty
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        initial_population: List[Individual],
        context: SchedulingContext,
        max_generations: int = 2000,
        max_steps_per_episode: int = 20,
        render_mode: Optional[str] = None,
        fast_evaluation: bool = True,
        debug_logging: bool = False,
        env_rank: int = 0,
        debug_log_interval: int = 25,
    ):
        """
        Initialize RL environment.

        Args:
            initial_population: Initial GA population
            context: Scheduling context (courses, rooms, etc.)
            max_generations: Maximum GA generations
            max_steps_per_episode: Maximum RL steps per episode (default: 20 for speed)
            render_mode: Rendering mode ("human", "ansi", None)
            fast_evaluation: Use cached fitness when possible (10x faster)
            debug_logging: Enable verbose instrumentation output
            env_rank: Environment index when running in parallel
            debug_log_interval: Step interval between debug log entries
        """
        super().__init__()

        self.context = context
        self.max_generations = max_generations
        self.max_steps_per_episode = max_steps_per_episode
        self.render_mode = render_mode
        self.fast_evaluation = fast_evaluation
        self.debug_logging = debug_logging
        self.env_rank = env_rank
        self.debug_log_interval = max(1, debug_log_interval)

        # Initialize components
        self.state_encoder = StateEncoder(
            max_generations=max_generations, history_size=10
        )
        self.action_mapper = ActionMapper(use_config=True)
        self.reward_calculator = RewardCalculator(
            fitness_weight=1.0, diversity_weight=0.1, time_weight=0.01
        )

        # Define spaces
        obs_dim = self.state_encoder.observation_dim
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        n_actions = self.action_mapper.n_actions
        self.action_space = spaces.Discrete(n_actions)

        # Episode state
        # CRITICAL: Deep copy individuals to avoid shared references across episodes!
        # shallow copy() would share Individual objects = memory corruption
        self.population: List[Individual] = [
            self._clone_individual(ind) for ind in initial_population
        ]
        self._initial_population = [
            self._clone_individual(ind) for ind in initial_population
        ]  # Store for reset
        self.current_generation = 0
        self.current_step = 0
        self.generations_without_improvement = 0
        self.best_fitness_ever = float("inf")
        self.episode_heuristic_counts: Dict[int, int] = {}

        # Render buffer
        self.render_buffer: List[str] = []
        self._fitness_evaluator: Optional[Callable] = None
        self._last_debug_step = -1

        # Step counter for debug logging
        self._total_steps_taken = 0

        self._debug_log(
            "Environment initialized (population=%s, max_steps=%s, max_generations=%s)",
            len(self.population),
            self.max_steps_per_episode,
            self.max_generations,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            (initial_observation, info)
        """
        reset_start = time.perf_counter()

        super().reset(seed=seed)

        # Reset episode counters
        self.current_generation = 0
        self.current_step = 0
        self.generations_without_improvement = 0
        self.best_fitness_ever = float("inf")
        self.episode_heuristic_counts = {}
        self.render_buffer = []
        self._last_debug_step = -1
        self._total_steps_taken = 0

        if self.debug_logging:
            logger.info(
                f"[ENV {self.env_rank}] Reset called (total steps so far: {getattr(self, '_total_steps_taken', 0)})"
            )

        # Reset components
        self.state_encoder.reset()
        self.reward_calculator.reset()

        # Re-initialize population (if provided in options)
        # CRITICAL: Deep copy to avoid shared references!
        if options and "initial_population" in options:
            self.population = [
                self._clone_individual(ind) for ind in options["initial_population"]
            ]
        else:
            # Reset to fresh copy of initial population
            self.population = [
                self._clone_individual(ind) for ind in self._initial_population
            ]

        # Get initial observation
        observation = self.state_encoder.encode(
            self.population,
            self.current_generation,
            self.generations_without_improvement,
        )

        info = self._get_info()

        duration_ms = (time.perf_counter() - reset_start) * 1000
        self._debug_log(
            "Environment reset complete (seed=%s, pop=%s, duration=%.2f ms)",
            seed,
            len(self.population),
            duration_ms,
        )

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Heuristic action to apply [0-19]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        step_start = time.perf_counter()

        # Track total steps for debugging
        self._total_steps_taken += 1

        # Log FREQUENTLY at first to show it's working, then reduce frequency
        log_freq = 5 if self._total_steps_taken < 50 else 25
        if self.debug_logging and self._total_steps_taken % log_freq == 0:
            logger.info(
                f"[ENV {self.env_rank}] Step {self._total_steps_taken} - action={action}"
            )

        # Convert numpy array to int (SB3 returns actions as arrays)
        if isinstance(action, np.ndarray):
            action = int(action.item())

        # Validate action
        if not self.action_mapper.is_valid_action(action):
            self._debug_log(
                "Invalid action %s received; applying penalty (population=%s)",
                action,
                len(self.population),
            )
            # Invalid action - penalize and return
            obs = self.state_encoder.encode(
                self.population,
                self.current_generation,
                self.generations_without_improvement,
            )
            self._maybe_log_step_summary(
                action_label=f"invalid_{action}",
                reward=-0.1,
                success=False,
                population_diversity=self.state_encoder._calculate_diversity(
                    self.population
                ),
                duration_ms=(time.perf_counter() - step_start) * 1000,
                improvement_note="",
                force=True,
            )
            return obs, -0.1, False, False, {"error": "invalid_action"}

        # Record action
        self.episode_heuristic_counts[action] = (
            self.episode_heuristic_counts.get(action, 0) + 1
        )
        self.state_encoder.record_heuristic_application(action)
        action_info = self.action_mapper.get_action_info(action)
        action_label = action_info.name if action_info else f"action_{action}"

        # Apply action to best individual
        best_individual = min(self.population, key=lambda ind: ind.fitness.values[0])
        prev_fitness = best_individual.fitness.values
        prev_individual = self._clone_individual(best_individual)
        working_individual = self._clone_individual(best_individual)

        prev_hard = prev_fitness[0] if len(prev_fitness) > 0 else float("inf")
        prev_soft = prev_fitness[1] if len(prev_fitness) > 1 else 0.0

        self._debug_log(
            "Selected action=%s (%s); generation=%s, prev_fitness=(%.2f, %.2f)",
            action_label,
            action,
            self.current_generation,
            prev_hard,
            prev_soft,
        )

        modified_individual, success = self.action_mapper.apply_action(
            action,
            working_individual,
            self.context,
            population=self.population,
            generation=self.current_generation,
        )

        self._debug_log("Action %s success=%s", action_label, success)

        candidate = (
            modified_individual
            if isinstance(modified_individual, list)
            else working_individual
        )

        evaluated_candidate: Optional[Individual] = None

        if success:
            # Convert plain list to DEAP Individual if needed (for diversity operators)
            candidate = self._ensure_deap_individual(candidate)
            # Ensure mutated individual keeps a valid fitness tuple for downstream consumers
            success = self._ensure_individual_fitness(candidate, action_label)

        if success:
            evaluated_candidate = self._clone_individual(candidate)
            # Replace worst individual with modified copy
            worst_idx = max(
                range(len(self.population)),
                key=lambda i: self.population[i].fitness.values[0],
            )
            self.population[worst_idx] = self._clone_individual(evaluated_candidate)
            result_individual = self._clone_individual(evaluated_candidate)
        else:
            result_individual = prev_individual

        # Calculate population metrics
        population_diversity = self.state_encoder._calculate_diversity(self.population)

        # Calculate reward
        reward, _ = self.reward_calculator.calculate_reward(
            prev_individual,
            result_individual,
            population_diversity,
            self.current_generation,
        )

        # Update progress
        current_best_fitness = self._get_best_fitness()
        improvement = ""
        if current_best_fitness < self.best_fitness_ever:
            improvement = (
                f"IMPROVED by {self.best_fitness_ever - current_best_fitness:.2f}"
            )
            self.best_fitness_ever = current_best_fitness
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        duration_ms = (time.perf_counter() - step_start) * 1000

        self._maybe_log_step_summary(
            action_label=action_label,
            reward=reward,
            success=success,
            population_diversity=population_diversity,
            duration_ms=duration_ms,
            improvement_note=improvement,
            force=bool(improvement),
        )

        self.current_generation += 1
        self.current_step += 1

        # Get new observation
        observation = self.state_encoder.encode(
            self.population,
            self.current_generation,
            self.generations_without_improvement,
        )

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _is_terminated(self) -> bool:
        """Check if episode is terminated (goal reached)."""
        # Terminate if perfect solution found (0 violations)
        return self.best_fitness_ever == 0.0

    def _is_truncated(self) -> bool:
        """Check if episode is truncated (max steps reached)."""
        return (
            self.current_step >= self.max_steps_per_episode
            or self.current_generation >= self.max_generations
        )

    def _get_best_fitness(self) -> float:
        """Get best fitness in current population."""
        if not self.population:
            return float("inf")
        best_ind = min(self.population, key=lambda ind: ind.fitness.values[0])
        hard, soft = best_ind.fitness.values
        return abs(hard) * 100 + abs(soft)

    def _get_info(self) -> Dict[str, Any]:
        """Get episode info dictionary."""
        return {
            "generation": self.current_generation,
            "step": self.current_step,
            "best_fitness": self._get_best_fitness(),
            "generations_without_improvement": self.generations_without_improvement,
            "heuristic_counts": self.episode_heuristic_counts.copy(),
            "population_size": len(self.population),
        }

    def _debug_log(self, message: str, *args: Any) -> None:
        """Emit prefixed debug logs when instrumentation is enabled."""

        if not self.debug_logging:
            return

        logger.info(
            "[env=%s gen=%s step=%s] " + message,
            self.env_rank,
            self.current_generation,
            self.current_step,
            *args,
        )

    def _maybe_log_step_summary(
        self,
        *,
        action_label: str,
        reward: float,
        success: bool,
        population_diversity: float,
        duration_ms: float,
        improvement_note: str,
        force: bool = False,
    ) -> None:
        """Log periodic step summaries to avoid overwhelming the console."""

        if not self.debug_logging:
            return

        if (
            not force
            and self.current_step - self._last_debug_step < self.debug_log_interval
        ):
            return

        self._last_debug_step = self.current_step
        self._debug_log(
            "Step summary: action=%s reward=%.4f success=%s best=%.2f diversity=%.4f "
            "stagnation=%s duration=%.2f ms %s",
            action_label,
            reward,
            success,
            self._get_best_fitness(),
            population_diversity,
            self.generations_without_improvement,
            duration_ms,
            improvement_note,
        )

    def render(self) -> Optional[str]:
        """Render environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Render state as ANSI string."""
        lines = []
        lines.append(f"=== Schedule Optimization Environment ===")
        lines.append(f"Generation: {self.current_generation}/{self.max_generations}")
        lines.append(f"Step: {self.current_step}/{self.max_steps_per_episode}")
        lines.append(f"Best Fitness: {self._get_best_fitness():.2f}")
        lines.append(f"Stagnation: {self.generations_without_improvement} generations")
        lines.append(f"\nHeuristic Usage:")
        for action_id, count in sorted(self.episode_heuristic_counts.items()):
            action_info = self.action_mapper.get_action_info(action_id)
            if action_info:
                lines.append(f"  [{action_id:2d}] {action_info.name:25s}: {count:3d}x")
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    def _ensure_deap_individual(self, individual: Individual) -> Individual:
        """Convert plain list to DEAP Individual if needed."""
        if hasattr(individual, "fitness"):
            return individual

        # Import DEAP creator (already initialized by GAScheduler)
        from deap import creator

        # Create new DEAP Individual from plain list
        deap_individual = creator.Individual(individual)
        return deap_individual

    def _ensure_individual_fitness(
        self, individual: Individual, action_label: str
    ) -> bool:
        """Recompute fitness when heuristics invalidate it (RL requires dense rewards)."""

        if not hasattr(individual, "fitness"):
            logger.warning(
                "Heuristic candidate (%s) is missing fitness metadata after conversion; skipping.",
                action_label,
            )
            return False

        values = getattr(individual.fitness, "values", ())
        values_valid = (
            getattr(individual.fitness, "valid", False)
            and len(values) == 2
            and all(isinstance(value, (int, float)) for value in values)
        )

        # Fast evaluation mode: trust cached fitness if valid
        if values_valid and self.fast_evaluation:
            return True

        if values_valid:
            return True

        if self._fitness_evaluator is None:
            from src.ga.evaluator.fitness import evaluate as evaluate_fitness

            self._fitness_evaluator = evaluate_fitness

        try:
            fitness = self._fitness_evaluator(
                individual,
                courses=self.context.courses,
                instructors=self.context.instructors,
                groups=self.context.groups,
                rooms=self.context.rooms,
            )
        except Exception as exc:  # pragma: no cover - defensive log path
            logger.warning(
                "Failed to evaluate heuristic candidate (%s): %s",
                action_label,
                exc,
            )
            return False

        individual.fitness.values = fitness
        return True

    def _clone_individual(self, individual: Individual) -> Individual:
        """
        Return a copy so mutations don't alias population references.

        Uses shallow copy + manual list copy for 10-50x speedup vs deepcopy.
        Safe because SessionGene objects are immutable after creation.
        """
        # Shallow copy the individual (copies DEAP metadata)
        cloned = copy.copy(individual)

        # Manually copy the chromosome list (list of SessionGene objects)
        # SessionGene objects themselves don't need deep copy - they're effectively immutable
        cloned[:] = individual[:]

        # Copy fitness (shallow copy is sufficient - tuples are immutable)
        if hasattr(individual, "fitness") and hasattr(individual.fitness, "values"):
            cloned.fitness.values = individual.fitness.values

        return cloned


def create_schedule_env(
    initial_population: List[Individual],
    context: SchedulingContext,
    max_generations: int = 2000,
    max_steps_per_episode: int = 20,
    render_mode: Optional[str] = None,
    fast_evaluation: bool = True,
    debug_logging: bool = False,
    env_rank: int = 0,
    debug_log_interval: int = 25,
) -> ScheduleEnv:
    """
    Factory function to create ScheduleEnv.

    Args:
        initial_population: Initial GA population
        context: Scheduling context
        max_generations: Maximum generations
        max_steps_per_episode: Maximum RL steps (default: 20 for speed)
        render_mode: Rendering mode
        fast_evaluation: Use cached fitness (10x faster)
        debug_logging: Enable verbose instrumentation output
        env_rank: Environment index when running in parallel
        debug_log_interval: Interval between debug log lines

    Returns:
        Configured ScheduleEnv instance
    """
    return ScheduleEnv(
        initial_population=initial_population,
        context=context,
        max_generations=max_generations,
        max_steps_per_episode=max_steps_per_episode,
        render_mode=render_mode,
        fast_evaluation=fast_evaluation,
        debug_logging=debug_logging,
        env_rank=env_rank,
        debug_log_interval=debug_log_interval,
    )
