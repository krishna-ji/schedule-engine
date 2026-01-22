"""
Rank-based multi-agent RL.

ENHANCEMENT #8: Specialist agents for different Pareto ranks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from stable_baselines3 import PPO

from schedule_engine.domain.types import Individual


class RankBasedAgent:
    """
    Agent specialized for solutions at a specific Pareto rank.

    Rank 1 (elite): Careful refinement, avoid disruption
    Rank 2-3 (good): Standard optimization
    Rank 4+ (poor): Aggressive repair and exploration
    """

    def __init__(self, rank: int, model_path: str | None = None):
        """
        Initialize rank-based agent.

        Args:
            rank: Target Pareto rank (1 = best)
            model_path: Path to trained model
        """
        self.rank = rank
        self.model_path = model_path
        self.model: PPO | None = None
        self.activation_count = 0

    def select_action(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> int:
        """
        Select heuristic action for solution at this rank.

        Args:
            observation: State observation
            deterministic: Use deterministic policy

        Returns:
            Heuristic ID
        """
        if self.model is None:
            self._load_model()

        if self.model:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            self.activation_count += 1
            return int(action)  # type: ignore[no-any-return]
        else:
            # Fallback: rank-dependent heuristic selection
            return self._fallback_action()

    def _fallback_action(self) -> int:
        """Fallback heuristic selection based on rank."""
        if self.rank == 1:
            # Elite: gentle refinement (VDS, GLS)
            return int(np.random.choice([12, 18]))  # type: ignore[no-any-return]
        elif self.rank <= 3:
            # Good: standard optimization (Kempe, ejection)
            return int(np.random.choice([10, 11, 12]))  # type: ignore[no-any-return]
        else:
            # Poor: aggressive repair (multi-perturbation, exploration)
            return int(np.random.choice([7, 8, 14]))  # type: ignore[no-any-return]

    def _load_model(self) -> None:
        """Load trained model."""
        if self.model_path:
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(self.model_path)
            except Exception as e:
                print(f"Warning: Failed to load rank-{self.rank} agent: {e}")


class RankBasedMultiAgent:
    """
    Multi-agent system with rank-specific specialists.

    Maintains 4 agents:
    - Agent 1: Rank 1 (elite solutions)
    - Agent 2: Rank 2 (good solutions)
    - Agent 3: Rank 3 (moderate solutions)
    - Agent 4: Rank 4+ (poor solutions)
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize rank-based multi-agent system.

        Args:
            config: Configuration with model paths
        """
        self.config = config or {}

        # Initialize 4 rank-specific agents
        self.agents = [
            RankBasedAgent(
                rank=i + 1, model_path=self.config.get(f"rank_{i + 1}_model_path")
            )
            for i in range(4)
        ]

    def select_action_for_individual(
        self,
        individual: Individual,
        population: list[Individual],
        observation: NDArray[np.float32],
    ) -> int:
        """
        Select action for individual based on its Pareto rank.

        Args:
            individual: Individual to act on
            population: Full population (for rank calculation)
            observation: State observation

        Returns:
            Heuristic ID
        """
        # Calculate Pareto rank
        pareto_rank = self._compute_pareto_rank(individual, population)

        # Select appropriate agent (clamp to [1, 4])
        agent_idx = min(pareto_rank, 4) - 1
        agent = self.agents[agent_idx]

        # Select action
        return agent.select_action(observation)

    def _compute_pareto_rank(
        self, individual: Individual, population: list[Individual]
    ) -> int:
        """
        Compute Pareto rank of individual in population.

        Rank 1 = non-dominated (Pareto front)
        Rank 2 = dominated only by rank 1, etc.

        Args:
            individual: Individual to rank
            population: Full population

        Returns:
            Pareto rank (1-based)
        """
        ind_fitness = np.array(individual.fitness.values)  # type: ignore[attr-defined]

        # Count how many individuals dominate this one
        domination_count = 0
        for other in population:
            if other is individual:
                continue

            other_fitness = np.array(other.fitness.values)  # type: ignore[attr-defined]

            # Check dominance (minimization)
            # other dominates ind if: other <= ind in all objectives AND other < ind in at least one
            if np.all(other_fitness <= ind_fitness) and np.any(
                other_fitness < ind_fitness
            ):
                domination_count += 1

        # Simple ranking: rank = 1 + number of dominators
        # (This is approximate; full NSGA-II ranking is more complex)
        if domination_count == 0:
            return 1  # Non-dominated
        elif domination_count <= 5:
            return 2
        elif domination_count <= 15:
            return 3
        else:
            return 4

    def get_statistics(self) -> dict[str, int]:
        """Get usage statistics for all agents."""
        return {
            f"rank_{i + 1}_activations": agent.activation_count
            for i, agent in enumerate(self.agents)
        }
