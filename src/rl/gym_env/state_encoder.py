"""
State encoder for RL environment.

Converts GA population state into a normalized observation vector for the RL agent.
Encodes 15+ features capturing population quality, diversity, and progress.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from src.domain.types import Individual


@dataclass
class StateFeatures:
    """Container for extracted state features."""

    # Fitness metrics (5 features)
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    fitness_std: float
    fitness_range: float

    # Diversity metrics (5 features)
    population_diversity: float
    genotype_diversity: float
    phenotype_diversity: float
    fitness_diversity: float
    unique_fitness_ratio: float

    # Progress metrics (4 features)
    current_generation: int
    generations_without_improvement: int
    convergence_rate: float
    improvement_rate: float

    # Constraint violation metrics (3 features)
    avg_hard_violations: float
    avg_soft_violations: float
    violation_std: float

    # ENHANCEMENT #2: Per-constraint breakdown (8 hard + 4 soft = 12 features)
    constraint_breakdown: dict[str, float]  # Per-constraint violation counts

    # Heuristic history (dynamic)
    recent_heuristic_ids: list[int]  # Last N heuristic applications


class StateEncoder:
    """
    Encodes GA population state into normalized observation space.

    Observation space (Box):
    - Shape: (29,) base features + (history_size,) heuristic history
    - All values normalized to [0, 1] or [-1, 1]
    - Handles missing/invalid values gracefully

    Features include:
    - Fitness metrics (5): best, avg, worst, std, range
    - Diversity metrics (5): population, genotype, phenotype, fitness, unique_fitness_ratio
    - Progress metrics (4): generation, stagnation, convergence, improvement
    - Constraint metrics (3): hard, soft, violation_std
    - ENHANCEMENT #2: Per-constraint breakdown (13): 9 hard + 4 soft constraint violations
    - Heuristic history (dynamic): recent heuristic IDs
    """

    # Hard constraint names (9 total — Academic Nomenclature)
    HARD_CONSTRAINT_NAMES: ClassVar[list[str]] = [
        "CTE",  # Cohort Temporal Exclusivity
        "FTE",  # Faculty Temporal Exclusivity
        "SRE",  # Spatial Resource Exclusivity
        "FPC",  # Faculty Pedagogical Congruence
        "FFC",  # Facility Feature Congruence
        "FCA",  # Faculty Chronological Availability
        "CQF",  # Curriculum Quanta Fulfillment
        "ICTD",  # Intra-Course Temporal Dispersion
    ]

    # Soft constraint names (Academic Nomenclature)
    SOFT_CONSTRAINT_NAMES: ClassVar[list[str]] = [
        "CSC",  # Cohort Schedule Contiguity
        "FSC",  # Faculty Schedule Contiguity
        "MIP",  # Meridian Interval Preservation
        "SSCP",  # Symmetric Sub-Cohort Parallelism
    ]

    def __init__(
        self,
        max_generations: int = 2000,
        history_size: int = 10,
        normalize: bool = True,
        enable_constraint_breakdown: bool = True,
        diversity_update_interval: int = 1,
        diversity_sample_size: int | None = None,
    ):
        """
        Initialize state encoder.

        Args:
            max_generations: Maximum GA generations (for normalization)
            history_size: Number of recent heuristic applications to track
            normalize: Whether to normalize features to [0, 1]
            enable_constraint_breakdown: Enable per-constraint breakdown (Enhancement #2)
        """
        self.max_generations = max_generations
        self.history_size = history_size
        self.normalize = normalize
        self.enable_constraint_breakdown = enable_constraint_breakdown
        self.diversity_update_interval = max(1, diversity_update_interval)
        self.diversity_sample_size = diversity_sample_size

        # Track previous state for delta features
        self.prev_best_fitness: float | None = None
        self.prev_avg_fitness: float | None = None
        self._last_population_diversity: float | None = None
        self._last_phenotype_diversity: float | None = None
        self._last_diversity_generation: int | None = None

        # Heuristic application history
        self.heuristic_history: list[int] = []

    def encode(
        self,
        population: list[Individual],
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
            Normalized observation vector of shape (21 + history_size,)
        """
        features = self._extract_features(
            population, current_generation, generations_without_improvement
        )

        obs = self._features_to_vector(features)

        # FIX: Validate for NaN/Inf before normalization
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            import logging

            logging.getLogger(__name__).warning(
                f"Invalid observation detected (NaN/Inf). Clamping to valid range. "
                f"NaN count: {np.isnan(obs).sum()}, Inf count: {np.isinf(obs).sum()}"
            )
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

        if self.normalize:
            obs = self._normalize_observation(obs)

        return obs.astype(np.float32)

    def _extract_features(
        self,
        population: list[Individual],
        current_generation: int,
        generations_without_improvement: int,
    ) -> StateFeatures:
        """Extract raw feature values from population."""
        if not population:
            return self._get_zero_features()

        valid_population = [
            ind
            for ind in population
            if hasattr(ind, "fitness")
            and getattr(ind.fitness, "valid", False)
            and len(ind.fitness.values) >= 2
        ]

        if not valid_population:
            return self._get_zero_features()

        # Extract fitness values (both objectives) from valid individuals
        hard_violations = np.array([ind.fitness.values[0] for ind in valid_population])
        soft_violations = np.array([ind.fitness.values[1] for ind in valid_population])

        # Combined fitness (weighted sum for single metric)
        fitness_values = hard_violations * 100 + soft_violations

        # Fitness statistics
        best_fitness = float(np.min(fitness_values))
        avg_fitness = float(np.mean(fitness_values))
        worst_fitness = float(np.max(fitness_values))
        fitness_std = float(np.std(fitness_values))
        fitness_range = worst_fitness - best_fitness

        # Diversity metrics
        population_diversity = self._calculate_diversity(
            valid_population, current_generation
        )
        genotype_diversity = self._calculate_genotype_diversity(population)
        phenotype_diversity = self._calculate_phenotype_diversity(
            valid_population, current_generation
        )
        fitness_diversity = fitness_std / (avg_fitness + 1e-6)
        unique_fitness_ratio = self._calculate_unique_fitness_ratio(valid_population)

        # Progress metrics
        convergence_rate = self._calculate_convergence_rate(fitness_std, avg_fitness)
        improvement_rate = self._calculate_improvement_rate(best_fitness, avg_fitness)

        # Constraint violations
        avg_hard = float(np.mean(np.abs(hard_violations)))
        avg_soft = float(np.mean(np.abs(soft_violations)))
        violation_std = float(np.std(fitness_values))

        # ENHANCEMENT #2: Per-constraint breakdown
        constraint_breakdown = {}
        if self.enable_constraint_breakdown:
            constraint_breakdown = self._calculate_constraint_breakdown(population)

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
            phenotype_diversity=phenotype_diversity,
            fitness_diversity=fitness_diversity,
            unique_fitness_ratio=unique_fitness_ratio,
            current_generation=current_generation,
            generations_without_improvement=generations_without_improvement,
            convergence_rate=convergence_rate,
            improvement_rate=improvement_rate,
            avg_hard_violations=avg_hard,
            avg_soft_violations=avg_soft,
            violation_std=violation_std,
            constraint_breakdown=constraint_breakdown,
            recent_heuristic_ids=self.heuristic_history[-self.history_size :],
        )

    def _calculate_diversity(
        self, population: list[Individual], current_generation: int
    ) -> float:
        """Calculate population diversity using fitness distance (optimized with scipy)."""
        if (
            self.diversity_update_interval > 1
            and self._last_population_diversity is not None
            and self._last_diversity_generation is not None
            and current_generation % self.diversity_update_interval != 0
        ):
            return self._last_population_diversity

        if len(population) < 2:
            return 0.0

        sample = self._maybe_sample_population(population)
        fitness_array = np.array([ind.fitness.values for ind in sample])  # type: ignore[attr-defined]
        # Use scipy pdist for 10-30x faster pairwise distance calculation
        from scipy.spatial.distance import pdist

        distances = pdist(fitness_array, metric="euclidean")

        diversity = float(np.mean(distances)) if len(distances) > 0 else 0.0
        self._last_population_diversity = diversity
        self._last_diversity_generation = current_generation
        return diversity

    def _calculate_genotype_diversity(self, population: list[Individual]) -> float:
        """
        Calculate genotype diversity (unique chromosome structures).

        Measures diversity at the genetic level by counting unique
        timeslot/room assignment pairs across the population.
        """
        if not population:
            return 0.0

        # Count unique timeslot/room assignments
        unique_assignments = set()
        for ind in population:
            for gene in ind:
                quanta = tuple(sorted(getattr(gene, "quanta", [])))
                unique_assignments.add((quanta, gene.room_id))

        # Normalize by population size * chromosome length
        max_diversity = len(population) * len(population[0])
        return len(unique_assignments) / max(max_diversity, 1)

    def _calculate_phenotype_diversity(
        self, population: list[Individual], current_generation: int
    ) -> float:
        """
        Calculate phenotype diversity (unique fitness outcomes).

        Measures diversity at the solution level by analyzing how different
        individuals are in terms of their evaluated fitness. Uses normalized
        pairwise distances in fitness space.
        """
        if (
            self.diversity_update_interval > 1
            and self._last_phenotype_diversity is not None
            and self._last_diversity_generation is not None
            and current_generation % self.diversity_update_interval != 0
        ):
            return self._last_phenotype_diversity

        if len(population) < 2:
            return 0.0

        # Extract fitness vectors (hard, soft)
        sample = self._maybe_sample_population(population)
        fitness_array = np.array([ind.fitness.values for ind in sample])  # type: ignore[attr-defined]

        # Calculate pairwise Euclidean distances in fitness space (scipy optimized)
        from scipy.spatial.distance import pdist

        distances = pdist(fitness_array, metric="euclidean")

        # Average distance normalized by population size
        if len(distances) == 0:
            return 0.0

        avg_distance = np.mean(distances)
        # Normalize by typical fitness range to get [0, 1] scale
        # Using fitness range from current population
        fitness_range = np.max(fitness_array) - np.min(fitness_array)
        if fitness_range < 1e-6:
            return 0.0

        diversity = float(min(avg_distance / (fitness_range + 1e-6), 1.0))
        self._last_phenotype_diversity = diversity
        self._last_diversity_generation = current_generation
        return diversity

    def _calculate_unique_fitness_ratio(self, population: list[Individual]) -> float:
        """
        Calculate ratio of unique fitness values in population.

        Returns:
            Ratio in [0, 1] where 1.0 means all individuals have unique fitness,
            and 0.0 means all have identical fitness (full convergence).
        """
        if not population:
            return 0.0

        # Extract fitness tuples (hard, soft)
        fitness_tuples = [ind.fitness.values for ind in population]  # type: ignore[attr-defined]

        # Count unique fitness values (with small tolerance for floating point)
        # Round to 4 decimal places to avoid floating point precision issues
        rounded_fitness = [
            tuple(round(f, 4) for f in fitness) for fitness in fitness_tuples
        ]
        unique_count = len(set(rounded_fitness))

        return unique_count / len(population)

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

    def _calculate_constraint_breakdown(
        self, population: list[Individual]
    ) -> dict[str, float]:
        """
        Calculate average violation count for each constraint across population.

        ENHANCEMENT #2: Per-constraint breakdown for targeted repair.

        Extracts per-constraint violation counts from individual metadata
        if available (populated during evaluation). Averages across population.

        Returns:
            Dictionary mapping constraint names to average violation counts.
        """
        constraint_counts = dict.fromkeys(self.HARD_CONSTRAINT_NAMES, 0.0)
        constraint_counts.update(dict.fromkeys(self.SOFT_CONSTRAINT_NAMES, 0.0))

        # Extract from individual metadata if available
        # (This is populated during fitness evaluation in core/ga_scheduler.py)
        individuals_with_data = 0
        for ind in population:
            if hasattr(ind, "constraint_breakdown") and ind.constraint_breakdown:
                individuals_with_data += 1
                for constraint_name, value in ind.constraint_breakdown.items():
                    if constraint_name in constraint_counts:
                        constraint_counts[constraint_name] += value

        # Average across individuals
        if individuals_with_data > 0:
            for key in constraint_counts:
                constraint_counts[key] /= individuals_with_data

        return constraint_counts

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
                features.phenotype_diversity,
                features.fitness_diversity,
                features.unique_fitness_ratio,
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

        # ENHANCEMENT #2: Add per-constraint breakdown (12 features)
        if self.enable_constraint_breakdown:
            constraint_features = []
            # Add hard constraints (8 features)
            for constraint_name in self.HARD_CONSTRAINT_NAMES:
                value = features.constraint_breakdown.get(constraint_name, 0.0)
                constraint_features.append(value)
            # Add soft constraints (4 features)
            for constraint_name in self.SOFT_CONSTRAINT_NAMES:
                value = features.constraint_breakdown.get(constraint_name, 0.0)
                constraint_features.append(value)
            constraint_array = np.array(constraint_features, dtype=np.float64)
        else:
            constraint_array = np.array([], dtype=np.float64)

        # Pad heuristic history to fixed size
        history = features.recent_heuristic_ids[-self.history_size :]
        history_padded = history + [0] * (self.history_size - len(history))
        history_array = np.array(history_padded, dtype=np.float64)

        if self.enable_constraint_breakdown:
            return np.concatenate([base_features, constraint_array, history_array])
        return np.concatenate([base_features, history_array])

    def _normalize_observation(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize observation to [0, 1] range with robust handling.

        Uses clipping to prevent extreme values from destabilizing training.
        All features are normalized to [0, 1] or [-1, 1] as appropriate.
        """
        normalized = obs.copy()

        # Fitness metrics (indices 0-4): clip to reasonable range
        # Handle division by zero and extreme values
        normalized[0:5] = np.clip(obs[0:5] / (1000.0 + 1e-6), 0, 1)

        # Diversity metrics (indices 5-9): already in [0, 1], but clip for safety
        normalized[5:10] = np.clip(obs[5:10], 0, 1)

        # Current generation (index 10): normalize with bounds checking
        if self.max_generations > 0:
            normalized[10] = np.clip(obs[10] / self.max_generations, 0, 1)
        else:
            normalized[10] = 0.0

        # Generations without improvement (index 11): clip to [0, 1]
        if self.max_generations > 0:
            normalized[11] = np.clip(obs[11] / self.max_generations, 0, 1)
        else:
            normalized[11] = 0.0

        # Convergence/improvement rates (indices 12-13): clip to [-1, 1]
        normalized[12:14] = np.clip(obs[12:14], -1, 1)

        # Violation metrics (indices 14-16): clip to reasonable range
        # Handle potential explosion of violations for infeasible schedules
        normalized[14:17] = np.clip(obs[14:17] / (100.0 + 1e-6), 0, 1)

        if self.enable_constraint_breakdown:
            # ENHANCEMENT #2: Normalize per-constraint breakdown (indices 17-28)
            # 8 hard constraints + 4 soft constraints = 12 features
            normalized[17:29] = np.clip(obs[17:29] / (50.0 + 1e-6), 0, 1)

            # Heuristic history (indices 29+): normalize by max heuristic ID (20)
            normalized[29:] = np.clip(obs[29:] / 20.0, 0, 1)
        else:
            # Heuristic history (indices 17+): normalize by max heuristic ID (20)
            # Handle edge case where history might contain invalid IDs
            normalized[17:] = np.clip(obs[17:] / 20.0, 0, 1)

        # Final safety check: replace any NaN or Inf values with 0
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

        return normalized

    def _get_zero_features(self) -> StateFeatures:
        """Return zero features for empty population."""
        zero_constraint_breakdown = {}
        if self.enable_constraint_breakdown:
            zero_constraint_breakdown = dict.fromkeys(self.HARD_CONSTRAINT_NAMES, 0.0)
            zero_constraint_breakdown.update(
                dict.fromkeys(self.SOFT_CONSTRAINT_NAMES, 0.0)
            )

        return StateFeatures(
            best_fitness=0.0,
            avg_fitness=0.0,
            worst_fitness=0.0,
            fitness_std=0.0,
            fitness_range=0.0,
            population_diversity=0.0,
            genotype_diversity=0.0,
            phenotype_diversity=0.0,
            fitness_diversity=0.0,
            unique_fitness_ratio=0.0,
            current_generation=0,
            generations_without_improvement=0,
            convergence_rate=0.0,
            improvement_rate=0.0,
            avg_hard_violations=0.0,
            avg_soft_violations=0.0,
            violation_std=0.0,
            constraint_breakdown=zero_constraint_breakdown,
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
        self._last_population_diversity = None
        self._last_phenotype_diversity = None
        self._last_diversity_generation = None

    def _maybe_sample_population(
        self, population: list[Individual]
    ) -> list[Individual]:
        """Optionally subsample the population for diversity calculations."""
        if (
            self.diversity_sample_size is None
            or self.diversity_sample_size <= 0
            or len(population) <= self.diversity_sample_size
        ):
            return population

        import random

        return random.sample(population, self.diversity_sample_size)

    @property
    def observation_dim(self) -> int:
        """Get observation space dimension."""
        base_features = 17
        constraint_features = 12 if self.enable_constraint_breakdown else 0
        return base_features + constraint_features + self.history_size
