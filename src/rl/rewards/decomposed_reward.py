"""
MOEA/D-style decomposed reward calculator.

ENHANCEMENT #1: Decomposes multi-objective problem into multiple
single-objective subproblems using weight vectors.

Each RL agent in an ensemble optimizes a different weighted combination,
collectively exploring the entire Pareto front.
"""

import numpy as np
from numpy.typing import NDArray

from src.domain.types import Individual
from src.rl.rewards.base_reward import BaseRewardCalculator


class DecomposedReward(BaseRewardCalculator):
    """
    Decomposed reward using MOEA/D-style weight vectors.

    Reward = -g_te(x | λ, z*)

    Where:
    - g_te = Tchebycheff scalarization function
    - λ = weight vector for this agent
    - z* = ideal point (best values seen for each objective)

    Benefits:
    - Enables multi-agent ensemble (each agent targets different region)
    - More stable than pure multi-objective methods
    - Proven effective in MOEA/D algorithm

    Usage:
    - Train N agents with different weight vectors
    - Each agent becomes specialist for different Pareto trade-off
    - Ensemble collectively covers entire Pareto front
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize decomposed reward calculator.

        Args:
            config: Configuration with:
                - weight_vector: [w_hard, w_soft] for this agent
                - ideal_point: Best known values [z_hard*, z_soft*]
                - scalarization: 'tchebycheff' or 'weighted_sum'
        """
        super().__init__(config)

        # Weight vector for this agent (should sum to 1.0)
        self.weight_vector = np.array(self.config.get("weight_vector", [0.5, 0.5]))
        self.weight_vector = self.weight_vector / self.weight_vector.sum()

        # Ideal point (best values for each objective)
        # Updated dynamically during training
        self.ideal_point = np.array(self.config.get("ideal_point", [0.0, 0.0]))

        # Scalarization method
        self.scalarization = self.config.get("scalarization", "tchebycheff")

    def calculate(
        self,
        prev_population: list[Individual],
        current_population: list[Individual],
        action_cost: float = 0.0,
    ) -> float:
        """
        Calculate reward using decomposed scalarization.

        Args:
            prev_population: Population before action
            current_population: Population after action
            action_cost: Cost of the action taken

        Returns:
            Reward based on scalarized fitness improvement
        """
        # Get best fitness from each population
        prev_best = np.array(self.get_best_fitness(prev_population))
        curr_best = np.array(self.get_best_fitness(current_population))

        # Update ideal point
        self.ideal_point = np.minimum(self.ideal_point, curr_best)

        # Calculate scalarized values
        prev_scalar = self._scalarize(prev_best)
        curr_scalar = self._scalarize(curr_best)

        # Reward = improvement (negative because scalarization is minimized)
        reward = -(curr_scalar - prev_scalar)

        return float(reward)

    def _scalarize(self, fitness: NDArray[np.float64]) -> float:
        """
        Scalarize multi-objective fitness using weight vector.

        Args:
            fitness: [hard_violations, soft_violations]

        Returns:
            Scalar value to minimize
        """
        if self.scalarization == "tchebycheff":
            # Tchebycheff: max_i { w_i * |f_i - z_i*| }
            diff = np.abs(fitness - self.ideal_point)
            weighted_diff = self.weight_vector * diff
            return float(np.max(weighted_diff))

        elif self.scalarization == "weighted_sum":
            # Weighted sum: Σ w_i * f_i
            return float(np.dot(self.weight_vector, fitness))

        else:
            raise ValueError(f"Unknown scalarization: {self.scalarization}")

    def update_ideal_point(self, population: list[Individual]) -> None:
        """
        Update ideal point with best values from population.

        Should be called periodically during training.

        Args:
            population: Current GA population
        """
        if not population:
            return

        best = np.array(self.get_best_fitness(population))
        self.ideal_point = np.minimum(self.ideal_point, best)

    @staticmethod
    def generate_weight_vectors(n_agents: int) -> list[NDArray[np.float64]]:
        """
        Generate uniformly distributed weight vectors for ensemble.

        Args:
            n_agents: Number of agents in ensemble

        Returns:
            List of weight vectors, each of shape (2,)

        Example:
            n_agents=5 → weights:
            [1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]
        """
        weights = []
        for i in range(n_agents):
            w_hard = i / (n_agents - 1) if n_agents > 1 else 0.5
            w_soft = 1.0 - w_hard
            weights.append(np.array([w_hard, w_soft]))

        return weights
