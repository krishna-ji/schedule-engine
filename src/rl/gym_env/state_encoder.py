"""
State encoder for RL environment.

Converts GA population state into a normalized observation vector for the RL agent.
Encodes 15+ features capturing population quality, diversity, and progress.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from numpy.typing import NDArray

from src.core.types import Individual


@dataclass
class StateFeatures:
    """Container for extracted state features."""

    # Fitness metrics (5 features)
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    fitness_std: float
    fitness_range: float

    # Diversity metrics (3 features)
    population_diversity: float
    genotype_diversity: float
    fitness_diversity: float

    # Progress metrics (4 features)
    current_generation: int
    generations_without_improvement: int
    convergence_rate: float
    improvement_rate: float

    # Constraint violation metrics (3 features)
    avg_hard_violations: float
    avg_soft_violations: float
    violation_std: float

    # Heuristic history (dynamic)
    recent_heuristic_ids: List[int]  # Last N heuristic applications


class StateEncoder:
    """
    Encodes GA population state into normalized observation space.

    Observation space (Box):
    - Shape: (19,) base features + (history_size,) heuristic history
    - All values normalized to [0, 1] or [-1, 1]
    - Handles missing/invalid values gracefully
    """

    def __init__(
        self,
        max_generations: int = 2000,
        history_size: int = 10,
        normalize: bool = True,
    ):
        """
        Initialize state encoder.

        Args:
            max_generations: Maximum GA generations (for normalization)
            history_size: Number of recent heuristic applications to track
            normalize: Whether to normalize features to [0, 1]
        """
        self.max_generations = max_generations
        self.history_size = history_size
        self.normalize = normalize

        # Track previous state for delta features
        self.prev_best_fitness: float | None = None
        self.prev_avg_fitness: float | None = None

        # Heuristic application history
        self.heuristic_history: List[int] = []

    def encode(
        self,
        population: List[Individual],
        current_generation: int,
        generations_without_improvement: int,
    ) -> NDArray[np.float32]:
        """
        Encode population state into observation vector.

        Args:
            population: Current GA population
            current_generation: Current generation number
            generations_without_improvement: Generations since last improvement

        Returns:
            Normalized observation vector of shape (19 + history_size,)
        """
        features = self._extract_features(
            population, current_generation, generations_without_improvement
        )

        obs = self._features_to_vector(features)

        if self.normalize:
            obs = self._normalize_observation(obs)

        return obs.astype(np.float32)

    def _extract_features(
        self,
        population: List[Individual],
        current_generation: int,
        generations_without_improvement: int,
    ) -> StateFeatures:
        """Extract raw feature values from population."""
        if not population:
            return self._get_zero_features()

        # Extract fitness values (both objectives)
        hard_violations = np.array([ind.fitness.values[0] for ind in population])
        soft_violations = np.array([ind.fitness.values[1] for ind in population])

        # Combined fitness (weighted sum for single metric)
        fitness_values = hard_violations * 100 + soft_violations

        # Fitness statistics
        best_fitness = float(np.min(fitness_values))
        avg_fitness = float(np.mean(fitness_values))
        worst_fitness = float(np.max(fitness_values))
        fitness_std = float(np.std(fitness_values))
        fitness_range = worst_fitness - best_fitness

        # Diversity metrics
        population_diversity = self._calculate_diversity(population)
        genotype_diversity = self._calculate_genotype_diversity(population)
        fitness_diversity = fitness_std / (avg_fitness + 1e-6)

        # Progress metrics
        convergence_rate = self._calculate_convergence_rate(fitness_std, avg_fitness)
        improvement_rate = self._calculate_improvement_rate(best_fitness, avg_fitness)

        # Constraint violations
        avg_hard = float(np.mean(np.abs(hard_violations)))
        avg_soft = float(np.mean(np.abs(soft_violations)))
        violation_std = float(np.std(fitness_values))

        # Update previous state
        self.prev_best_fitness = best_fitness
        self.prev_avg_fitness = avg_fitness

        return StateFeatures(
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            worst_fitness=worst_fitness,
            fitness_std=fitness_std,
            fitness_range=fitness_range,
            population_diversity=population_diversity,
            genotype_diversity=genotype_diversity,
            fitness_diversity=fitness_diversity,
            current_generation=current_generation,
            generations_without_improvement=generations_without_improvement,
            convergence_rate=convergence_rate,
            improvement_rate=improvement_rate,
            avg_hard_violations=avg_hard,
            avg_soft_violations=avg_soft,
            violation_std=violation_std,
            recent_heuristic_ids=self.heuristic_history[-self.history_size :],
        )

    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity using fitness distance."""
        if len(population) < 2:
            return 0.0

        fitness_array = np.array([ind.fitness.values for ind in population])
        # Pairwise distances
        distances = []
        for i in range(len(fitness_array)):
            for j in range(i + 1, len(fitness_array)):
                dist = np.linalg.norm(fitness_array[i] - fitness_array[j])
                distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def _calculate_genotype_diversity(self, population: List[Individual]) -> float:
        """Calculate genotype diversity (unique chromosome structures)."""
        if not population:
            return 0.0

        # Count unique timeslot/room assignments
        unique_assignments = set()
        for ind in population:
            for gene in ind:
                unique_assignments.add((gene.timeslot_index, gene.room_id))

        # Normalize by population size * chromosome length
        max_diversity = len(population) * len(population[0])
        return len(unique_assignments) / max(max_diversity, 1)

    def _calculate_convergence_rate(
        self, fitness_std: float, avg_fitness: float
    ) -> float:
        """Calculate convergence rate (how uniform the population is)."""
        if avg_fitness == 0:
            return 0.0
        return 1.0 - min(fitness_std / (abs(avg_fitness) + 1e-6), 1.0)

    def _calculate_improvement_rate(
        self, current_best: float, current_avg: float
    ) -> float:
        """Calculate improvement rate since last state."""
        if self.prev_best_fitness is None:
            return 0.0

        improvement = self.prev_best_fitness - current_best
        return improvement / (abs(self.prev_best_fitness) + 1e-6)

    def _features_to_vector(self, features: StateFeatures) -> NDArray[np.float64]:
        """Convert features to numpy vector."""
        base_features = np.array(
            [
                features.best_fitness,
                features.avg_fitness,
                features.worst_fitness,
                features.fitness_std,
                features.fitness_range,
                features.population_diversity,
                features.genotype_diversity,
                features.fitness_diversity,
                float(features.current_generation),
                float(features.generations_without_improvement),
                features.convergence_rate,
                features.improvement_rate,
                features.avg_hard_violations,
                features.avg_soft_violations,
                features.violation_std,
            ],
            dtype=np.float64,
        )

        # Pad heuristic history to fixed size
        history = features.recent_heuristic_ids[-self.history_size :]
        history_padded = history + [0] * (self.history_size - len(history))
        history_array = np.array(history_padded, dtype=np.float64)

        return np.concatenate([base_features, history_array])

    def _normalize_observation(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize observation to [0, 1] range."""
        normalized = obs.copy()

        # Fitness metrics (indices 0-4): clip to reasonable range
        normalized[0:5] = np.clip(obs[0:5] / 1000.0, 0, 1)

        # Diversity metrics (indices 5-7): already in [0, 1]
        normalized[5:8] = np.clip(obs[5:8], 0, 1)

        # Current generation (index 8)
        normalized[8] = obs[8] / self.max_generations

        # Generations without improvement (index 9): clip to max_generations
        normalized[9] = min(obs[9] / self.max_generations, 1.0)

        # Convergence/improvement rates (indices 10-11): clip to [-1, 1]
        normalized[10:12] = np.clip(obs[10:12], -1, 1)

        # Violation metrics (indices 12-14): clip to reasonable range
        normalized[12:15] = np.clip(obs[12:15] / 100.0, 0, 1)

        # Heuristic history (indices 15+): normalize by max heuristic ID (20)
        normalized[15:] = obs[15:] / 20.0

        return normalized

    def _get_zero_features(self) -> StateFeatures:
        """Return zero features for empty population."""
        return StateFeatures(
            best_fitness=0.0,
            avg_fitness=0.0,
            worst_fitness=0.0,
            fitness_std=0.0,
            fitness_range=0.0,
            population_diversity=0.0,
            genotype_diversity=0.0,
            fitness_diversity=0.0,
            current_generation=0,
            generations_without_improvement=0,
            convergence_rate=0.0,
            improvement_rate=0.0,
            avg_hard_violations=0.0,
            avg_soft_violations=0.0,
            violation_std=0.0,
            recent_heuristic_ids=[],
        )

    def record_heuristic_application(self, heuristic_id: int) -> None:
        """Record a heuristic application for history tracking."""
        self.heuristic_history.append(heuristic_id)
        # Keep only recent history
        if len(self.heuristic_history) > self.history_size * 2:
            self.heuristic_history = self.heuristic_history[-self.history_size :]

    def reset(self) -> None:
        """Reset encoder state (call at episode start)."""
        self.prev_best_fitness = None
        self.prev_avg_fitness = None
        self.heuristic_history = []

    @property
    def observation_dim(self) -> int:
        """Get observation space dimension."""
        return 15 + self.history_size  # 15 base features + history
