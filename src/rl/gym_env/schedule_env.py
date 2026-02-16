"""
Gymnasium environment for schedule optimization.

Wraps the GA scheduler as an RL environment where the agent learns to select
effective heuristics at each step.
"""

import copy
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.domain.gene import SessionGene
from src.domain.types import Individual, SchedulingContext
from src.rl.gym_env.action_space import ActionMapper
from src.rl.gym_env.reward_calculator import RewardCalculator
from src.rl.gym_env.state_encoder import StateEncoder
from src.utils.logging_config import get_logger
from src.utils.performance_profiler import PerformanceProfiler

logger = get_logger(__name__)


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

    metadata = {"render_modes": ["human", "ansi"]}  # noqa: RUF012

    def __init__(
        self,
        initial_population: list[Individual],
        context: SchedulingContext,
        max_generations: int = 2000,
        max_steps_per_episode: int = 20,
        render_mode: str | None = None,
        fast_evaluation: bool = True,
        debug_logging: bool = True,
        env_rank: int = 0,
        debug_log_interval: int = 25,
        diversity_update_interval: int = 1,
        diversity_sample_size: int | None = None,
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
            max_generations=max_generations,
            history_size=10,
            diversity_update_interval=diversity_update_interval,
            diversity_sample_size=diversity_sample_size,
        )
        self.action_mapper = ActionMapper(use_config=True, timeout_seconds=30.0)
        self.reward_calculator = RewardCalculator(
            fitness_weight=1.0, diversity_weight=0.1, time_weight=0.01
        )

        # Initialize profiler (enabled when debug logging is on, but not verbose to avoid progress bar interference)
        self.profiler = PerformanceProfiler(enabled=debug_logging, verbose=False)

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
        self.population: list[Individual] = [
            self._clone_individual(ind) for ind in initial_population
        ]
        self._initial_population = [
            self._clone_individual(ind) for ind in initial_population
        ]  # Store for reset
        self.current_generation = 0
        self.current_step = 0
        self.generations_without_improvement = 0
        self.best_fitness_ever = float("inf")
        self.episode_heuristic_counts: dict[int, int] = {}

        # Render buffer
        self.render_buffer: list[str] = []
        self._fitness_evaluator: Callable | None = None
        self._last_debug_step = -1

        # Step counter for debug logging
        self._total_steps_taken = 0

        # Set initial logging context
        logger.set_context(env_rank=self.env_rank, generation=0, step=0)

        if self.debug_logging:
            logger.info(
                "Environment initialized",
                population=len(self.population),
                max_steps=self.max_steps_per_episode,
                max_generations=self.max_generations,
            )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
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

        # Update logging context
        logger.set_context(env_rank=self.env_rank, generation=0, step=0)

        if self.debug_logging:
            logger.info("Environment reset", total_steps_so_far=self._total_steps_taken)

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

        if self.debug_logging:
            logger.debug(
                "Reset complete",
                seed=seed,
                population=len(self.population),
                duration_ms=f"{duration_ms:.2f}",
            )

        return observation, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Heuristic action to apply [0-19]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        step_start = time.perf_counter()
        self.profiler.start_generation(self.current_step)

        # Track total steps for debugging
        self._total_steps_taken += 1

        # Update logging context
        logger.set_context(
            env_rank=self.env_rank,
            generation=self.current_generation,
            step=self.current_step,
        )

        # Log FREQUENTLY at first to show it's working, then reduce frequency
        log_freq = 5 if self._total_steps_taken < 50 else 25
        if self.debug_logging and self._total_steps_taken % log_freq == 0:
            logger.debug(
                "Step start", total_steps=self._total_steps_taken, action=action
            )

        # Convert numpy array to int (SB3 returns actions as arrays)
        if isinstance(action, np.ndarray):
            action = int(action.item())

        # Validate action
        if not self.action_mapper.is_valid_action(action):
            if self.debug_logging:
                logger.warning(
                    "Invalid action received",
                    action=action,
                    population=len(self.population),
                )
            # Invalid action - penalize and return
            obs = self.state_encoder.encode(
                self.population,
                self.current_generation,
                self.generations_without_improvement,
            )
            duration_ms = (time.perf_counter() - step_start) * 1000
            if self.debug_logging:
                logger.step_summary(
                    action=f"invalid_{action}",
                    reward=-0.1,
                    success=False,
                    best_fitness=self._get_best_fitness(),
                    diversity=self.state_encoder._calculate_diversity(
                        self.population, self.current_generation
                    ),
                    stagnation=self.generations_without_improvement,
                    duration_ms=duration_ms,
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
        best_individual = min(self.population, key=self._get_combined_fitness_value)
        prev_fitness = best_individual.fitness.values  # type: ignore[attr-defined]
        prev_individual = self._clone_individual(best_individual)
        working_individual = self._clone_individual(best_individual)

        prev_hard = prev_fitness[0] if len(prev_fitness) > 0 else float("inf")
        prev_soft = prev_fitness[1] if len(prev_fitness) > 1 else 0.0

        if self.debug_logging:
            logger.debug(
                "Applying action",
                action=action_label,
                action_id=action,
                prev_fitness=f"({prev_hard:.2f}, {prev_soft:.2f})",
            )

        # Phase 2: Apply heuristic action (main computational work)
        self.profiler.start_phase(
            f"apply_{action_label}", items_to_process=len(self.population)
        )
        modified_individual, success = self.action_mapper.apply_action(
            action,
            working_individual,
            self.context,
            population=self.population,
            generation=self.current_generation,
        )
        self.profiler.end_phase()

        if self.debug_logging:
            logger.action(action_label, success=success)

        candidate = (
            modified_individual
            if isinstance(modified_individual, list)
            else working_individual
        )

        evaluated_candidate: Individual | None = None

        if success:
            # Phase 4: Ensure DEAP individual format
            self.profiler.start_phase("ensure_deap_individual")
            candidate = self._ensure_deap_individual(candidate)
            self.profiler.end_phase()

            # Phase 5: Evaluate fitness
            self.profiler.start_phase("evaluate_fitness")
            success = self._ensure_individual_fitness(candidate, action_label)
            self.profiler.end_phase()

        if success:
            # Phase 6: Update population
            self.profiler.start_phase("update_population", len(self.population))
            evaluated_candidate = self._clone_individual(candidate)
            # Replace worst individual with modified copy
            worst_idx = max(
                range(len(self.population)),
                key=lambda i: self._get_combined_fitness_value(self.population[i]),
            )
            self.population[worst_idx] = self._clone_individual(evaluated_candidate)
            result_individual = self._clone_individual(evaluated_candidate)
            self.profiler.end_phase()
        else:
            result_individual = prev_individual

        # Phase 7: Calculate diversity metrics
        self.profiler.start_phase("calculate_diversity", len(self.population))
        population_diversity = self.state_encoder._calculate_diversity(
            self.population, self.current_generation
        )
        self.profiler.end_phase()

        # Phase 8: Calculate reward
        self.profiler.start_phase("calculate_reward")
        reward, _ = self.reward_calculator.calculate_reward(
            prev_individual,
            result_individual,
            population_diversity,
            self.current_generation,
            population=self.population,
        )
        self.profiler.end_phase()

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

        # Log step summary using structured logger
        if self.debug_logging and (
            improvement or self.current_step % self.debug_log_interval == 0
        ):
            logger.step_summary(
                action=action_label,
                reward=reward,
                success=success,
                best_fitness=self._get_best_fitness(),
                diversity=population_diversity,
                stagnation=self.generations_without_improvement,
                duration_ms=duration_ms,
                improvement=float(improvement.split()[-1]) if improvement else None,
            )

        self.current_generation += 1
        self.current_step += 1

        # Phase 9: Encode new state
        self.profiler.start_phase("encode_observation", len(self.population))
        observation = self.state_encoder.encode(
            self.population,
            self.current_generation,
            self.generations_without_improvement,
        )
        self.profiler.end_phase()

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = self._get_info()

        # End profiling and add breakdown to info
        self.profiler.end_generation()
        if (
            self.debug_logging
            and hasattr(self.profiler, "generation_profiles")
            and self.profiler.generation_profiles
        ):
            last_profile = self.profiler.generation_profiles[-1]
            info["profile"] = last_profile.get_summary()

        # Print profiling summary at episode end
        if (terminated or truncated) and self.debug_logging:
            logger.info("Episode complete", total_steps=self.current_step)
            self.print_profiling_summary()

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
        best_ind = min(self.population, key=self._get_combined_fitness_value)
        return self._get_combined_fitness_value(best_ind)

    @staticmethod
    def _get_combined_fitness_value(individual: Individual) -> float:
        """Compute combined fitness value for comparisons."""
        if not hasattr(individual, "fitness") or not individual.fitness.valid:
            return float("inf")
        hard, soft = individual.fitness.values
        return float(abs(hard) * 100 + abs(soft))

    def _get_info(self) -> dict[str, Any]:
        """Get episode info dictionary."""
        return {
            "generation": self.current_generation,
            "step": self.current_step,
            "best_fitness": self._get_best_fitness(),
            "generations_without_improvement": self.generations_without_improvement,
            "heuristic_counts": self.episode_heuristic_counts.copy(),
            "population_size": len(self.population),
        }

    def render(self) -> str | None:  # type: ignore[override]
        """Render environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Render state as ANSI string."""
        lines = []
        lines.append("=== Schedule Optimization Environment ===")
        lines.append(f"Generation: {self.current_generation}/{self.max_generations}")
        lines.append(f"Step: {self.current_step}/{self.max_steps_per_episode}")
        lines.append(f"Best Fitness: {self._get_best_fitness():.2f}")
        lines.append(f"Stagnation: {self.generations_without_improvement} generations")
        lines.append("\nHeuristic Usage:")
        for action_id, count in sorted(self.episode_heuristic_counts.items()):
            action_info = self.action_mapper.get_action_info(action_id)
            if action_info:
                lines.append(f"  [{action_id:2d}] {action_info.name:25s}: {count:3d}x")
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up environment resources."""

    def _ensure_deap_individual(self, individual: Individual) -> Individual:
        """Convert plain list to DEAP Individual if needed."""
        if hasattr(individual, "fitness"):
            return individual

        # Import DEAP creator (already initialized by GAScheduler)
        from deap import creator

        # Create new DEAP Individual from plain list
        deap_individual: Individual = creator.Individual(individual)
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
            and all(isinstance(value, int | float) for value in values)
        )

        # Fast evaluation mode: trust cached fitness if valid
        if values_valid and self.fast_evaluation:
            return True

        if values_valid:
            return True

        if self._fitness_evaluator is None:
            from src.ga.core.evaluator import evaluate as evaluate_fitness

            self._fitness_evaluator = evaluate_fitness

        try:
            fitness = self._fitness_evaluator(  # type: ignore[operator]
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

        Uses shallow copy for DEAP metadata and deep-copies genes to avoid
        cross-episode or cross-env mutation leakage.
        """
        # Shallow copy the individual (copies DEAP metadata)
        cloned = copy.copy(individual)

        # Copy the chromosome list and its genes (SessionGene is mutable)
        cloned[:] = [
            SessionGene(
                course_id=gene.course_id,
                course_type=gene.course_type,
                instructor_id=gene.instructor_id,
                group_ids=list(gene.group_ids),
                room_id=gene.room_id,
                start_quanta=gene.start_quanta,
                num_quanta=gene.num_quanta,
            )
            for gene in individual
        ]

        # Copy fitness so mutations don't alias original individuals
        if hasattr(individual, "fitness") and hasattr(individual.fitness, "values"):
            cloned.fitness = copy.copy(individual.fitness)  # type: ignore[attr-defined]
            cloned.fitness.values = individual.fitness.values  # type: ignore[attr-defined]

        return cloned

    def get_profiling_summary(self) -> dict[str, Any]:
        """
        Get comprehensive profiling statistics for this environment.

        Returns:
            Dictionary with profiling statistics per phase
        """
        if not hasattr(self.profiler, "get_statistics"):
            return {}

        return self.profiler.get_statistics()

    def print_profiling_summary(self) -> None:
        """Print profiling summary table at end of episode."""
        if not self.debug_logging:
            return

        if hasattr(self.profiler, "print_summary_table"):
            logger.info("Profiling summary")
            self.profiler.print_summary_table()


def create_schedule_env(
    initial_population: list[Individual],
    context: SchedulingContext,
    max_generations: int = 2000,
    max_steps_per_episode: int = 20,
    render_mode: str | None = None,
    fast_evaluation: bool = True,
    debug_logging: bool = False,
    env_rank: int = 0,
    debug_log_interval: int = 25,
    diversity_update_interval: int | None = None,
    diversity_sample_size: int | None = None,
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
        diversity_update_interval: Compute diversity metrics every N generations
        diversity_sample_size: Optional subsample size for diversity metrics

    Returns:
        Configured ScheduleEnv instance
    """
    if diversity_update_interval is None:
        from src.config import get_config

        env_config = get_config().rl.environment
        diversity_update_interval = env_config.diversity_update_interval
        diversity_sample_size = env_config.diversity_sample_size

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
        diversity_update_interval=diversity_update_interval,
        diversity_sample_size=diversity_sample_size,
    )
