"""
Specialist agents for different search scenarios.

ENHANCEMENT #4: Task-specific RL agents with coordinator.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.domain.types import Individual


class SpecialistAgent(ABC):
    """
    Base class for specialist RL agents.

    Each specialist focuses on a specific search scenario:
    - Repair: Infeasible → Feasible
    - Optimizer: Feasible → Better soft constraints
    - Explorer: Stagnated → Diverse
    - Intensifier: Elite → Perfect
    """

    def __init__(self, name: str, model_path: str | None = None):
        """
        Initialize specialist agent.

        Args:
            name: Agent name/identifier
            model_path: Path to trained model
        """
        self.name = name
        self.model_path = model_path
        # Import PPO type for annotation
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from stable_baselines3 import PPO
        self.model: PPO | None = None
        self.activation_count = 0
        self.success_count = 0

    @abstractmethod
    def should_activate(
        self, population: list[Individual], state: dict[str, Any]
    ) -> bool:
        """
        Determine if this specialist should be activated.

        Args:
            population: Current GA population
            state: Search state (generation, stagnation, etc.)

        Returns:
            True if this agent's expertise is needed
        """

    @abstractmethod
    def select_action(self, observation: NDArray[np.float32]) -> int:
        """
        Select heuristic action for current state.

        Args:
            observation: State observation vector

        Returns:
            Action ID (heuristic index)
        """

    def load_model(self) -> None:
        """Load trained model from disk."""
        if self.model is None and self.model_path:
            # Lazy loading
            try:
                from stable_baselines3 import PPO

                loaded_model: PPO = PPO.load(self.model_path)
                self.model = loaded_model
                print(f"Loaded {self.name} model from {self.model_path}")
            except Exception as e:
                print(f"Warning: Failed to load {self.name} model: {e}")
                self.model = None

    def record_activation(self, was_successful: bool) -> None:
        """
        Record agent activation and outcome.

        Args:
            was_successful: Whether the action improved the population
        """
        self.activation_count += 1
        if was_successful:
            self.success_count += 1

    def get_success_rate(self) -> float:
        """Get agent's success rate."""
        if self.activation_count == 0:
            return 0.0
        return self.success_count / self.activation_count


class RepairAgent(SpecialistAgent):
    """
    Specialist for repairing infeasible solutions.

    Expertise: Hard constraint violations → 0
    Heuristics: Kempe chains, ejection chains, targeted swaps
    """

    def __init__(self, model_path: str | None = None):
        super().__init__("RepairAgent", model_path)
        self.hard_violation_threshold = 1.0  # Activate if any hard violations

    def should_activate(
        self, population: list[Individual], state: dict[str, Any]
    ) -> bool:
        """Activate if population has infeasible solutions."""
        # Check if best solution has hard violations
        if not population:
            return False

        best_fitness = min(ind.fitness.values for ind in population)  # type: ignore[attr-defined]
        has_hard_violations = best_fitness[0] > 0

        return bool(has_hard_violations)

    def select_action(self, observation: NDArray[np.float32]) -> int:
        """Select repair-focused heuristic."""
        if self.model is None:
            self.load_model()

        if self.model:
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action)
        # Fallback: Prioritize improvement heuristics
        # Kempe chain (action 10), ejection chain (action 11)
        return int(np.random.choice([10, 11]))


class OptimizerAgent(SpecialistAgent):
    """
    Specialist for optimizing feasible solutions.

    Expertise: Soft constraint violations → 0
    Heuristics: Local search, variable depth search, guided local search
    """

    def __init__(self, model_path: str | None = None):
        super().__init__("OptimizerAgent", model_path)

    def should_activate(
        self, population: list[Individual], state: dict[str, Any]
    ) -> bool:
        """Activate if population is feasible but not optimal."""
        if not population:
            return False

        best_fitness = min(ind.fitness.values for ind in population)  # type: ignore[attr-defined]
        is_feasible = best_fitness[0] == 0
        has_soft_violations = best_fitness[1] > 0

        return bool(is_feasible and has_soft_violations)

    def select_action(self, observation: NDArray[np.float32]) -> int:
        """Select optimization-focused heuristic."""
        if self.model is None:
            self.load_model()

        if self.model:
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action)
        # Fallback: Meta-heuristics for fine-tuning
        # Variable depth search (12), guided local search (18)
        return int(np.random.choice([12, 18]))


class ExplorerAgent(SpecialistAgent):
    """
    Specialist for escaping stagnation.

    Expertise: Diversity maintenance, exploration
    Heuristics: Perturbations, diversity operators, large neighborhood search
    """

    def __init__(self, model_path: str | None = None):
        super().__init__("ExplorerAgent", model_path)
        self.stagnation_threshold = 10  # Activate after 10 gens without improvement

    def should_activate(
        self, population: list[Individual], state: dict[str, Any]
    ) -> bool:
        """Activate if search is stagnated."""
        stagnation = state.get("generations_without_improvement", 0)
        return bool(stagnation >= self.stagnation_threshold)

    def select_action(self, observation: NDArray[np.float32]) -> int:
        """Select diversity-focused heuristic."""
        if self.model is None:
            self.load_model()

        if self.model:
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action)
        # Fallback: Perturbation + diversity heuristics
        # Multi-perturbation (8), crowding mutation (14), adaptive diversity (16)
        return int(np.random.choice([8, 14, 16]))


class IntensifierAgent(SpecialistAgent):
    """
    Specialist for elite solution refinement.

    Expertise: Near-optimal → Perfect
    Heuristics: Intensive local search, careful refinement
    """

    def __init__(self, model_path: str | None = None):
        super().__init__("IntensifierAgent", model_path)
        self.elite_threshold = 10.0  # Activate if best soft < 10

    def should_activate(
        self, population: list[Individual], state: dict[str, Any]
    ) -> bool:
        """Activate if population has elite solutions needing refinement."""
        if not population:
            return False

        best_fitness = min(ind.fitness.values for ind in population)  # type: ignore[attr-defined]
        is_feasible = best_fitness[0] == 0
        is_near_optimal = best_fitness[1] < self.elite_threshold

        # Only activate late in search
        generation = state.get("current_generation", 0)
        max_generations = state.get("max_generations", 2000)
        is_late_search = generation > 0.7 * max_generations

        return bool(is_feasible and is_near_optimal and is_late_search)

    def select_action(self, observation: NDArray[np.float32]) -> int:
        """Select intensive refinement heuristic."""
        if self.model is None:
            self.load_model()

        if self.model:
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action)
        # Fallback: Intensive local search
        # Iterated local search (17), guided local search (18)
        return int(np.random.choice([17, 18]))
