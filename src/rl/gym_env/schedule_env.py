"""
Gymnasium environment for schedule optimization.

Wraps the GA scheduler as an RL environment where the agent learns to select
effective heuristics at each step.
"""

import copy
import logging
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
        """
        super().__init__()

        self.context = context
        self.max_generations = max_generations
        self.max_steps_per_episode = max_steps_per_episode
        self.render_mode = render_mode
        self.fast_evaluation = fast_evaluation

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
        self.population: List[Individual] = initial_population.copy()
        self.current_generation = 0
        self.current_step = 0
        self.generations_without_improvement = 0
        self.best_fitness_ever = float("inf")
        self.episode_heuristic_counts: Dict[int, int] = {}

        # Render buffer
        self.render_buffer: List[str] = []
        self._fitness_evaluator: Optional[Callable] = None

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
        super().reset(seed=seed)

        # Reset episode counters
        self.current_generation = 0
        self.current_step = 0
        self.generations_without_improvement = 0
        self.best_fitness_ever = float("inf")
        self.episode_heuristic_counts = {}
        self.render_buffer = []

        # Reset components
        self.state_encoder.reset()
        self.reward_calculator.reset()

        # Re-initialize population (if provided in options)
        if options and "initial_population" in options:
            self.population = options["initial_population"].copy()

        # Get initial observation
        observation = self.state_encoder.encode(
            self.population,
            self.current_generation,
            self.generations_without_improvement,
        )

        info = self._get_info()

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
        # Convert numpy array to int (SB3 returns actions as arrays)
        if isinstance(action, np.ndarray):
            action = int(action.item())

        # Validate action
        if not self.action_mapper.is_valid_action(action):
            # Invalid action - penalize and return
            obs = self.state_encoder.encode(
                self.population,
                self.current_generation,
                self.generations_without_improvement,
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

        logger.debug(
            f"Step {self.current_step}: Action={action_label} ({action}), "
            f"Gen={self.current_generation}, Prev_Fitness={prev_fitness}"
        )

        modified_individual, success = self.action_mapper.apply_action(
            action,
            working_individual,
            self.context,
            population=self.population,
            generation=self.current_generation,
        )

        logger.debug(f"Action {action_label} success={success}")

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

        logger.debug(
            f"Result: Reward={reward:.4f}, BestFit={current_best_fitness:.2f}, "
            f"Diversity={population_diversity:.4f}, Stagnation={self.generations_without_improvement} {improvement}"
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
    )
