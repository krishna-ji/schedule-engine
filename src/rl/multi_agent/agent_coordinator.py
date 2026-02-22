"""
Agent coordinator for multi-agent RL system.

ENHANCEMENT #4: Coordinates specialist agent selection and execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from stable_baselines3 import PPO

    from src.domain.types import Individual

import numpy as np

from src.rl.multi_agent.specialist_agents import (
    ExplorerAgent,
    IntensifierAgent,
    OptimizerAgent,
    RepairAgent,
    SpecialistAgent,
)


class AgentCoordinator:
    """
    Coordinates multiple specialist RL agents.

    Strategies:
    1. State-based: Select agent based on search state
    2. UCB (Upper Confidence Bound): Balance exploration/exploitation of agents
    3. Meta-agent: RL agent learns which specialist to use
    """

    def __init__(self, strategy: str = "state_based", config: dict | None = None):
        """
        Initialize agent coordinator.

        Args:
            strategy: 'state_based', 'ucb', or 'meta_agent'
            config: Configuration with agent model paths
        """
        self.strategy = strategy
        self.config = config or {}

        # Initialize specialist agents
        self.agents: list[SpecialistAgent] = [
            RepairAgent(self.config.get("repair_model_path")),
            OptimizerAgent(self.config.get("optimizer_model_path")),
            ExplorerAgent(self.config.get("explorer_model_path")),
            IntensifierAgent(self.config.get("intensifier_model_path")),
        ]

        # UCB statistics
        self.ucb_counts = np.zeros(len(self.agents))
        self.ucb_rewards = np.zeros(len(self.agents))
        self.total_selections = 0

        # Meta-agent (if strategy=meta_agent)
        self.meta_agent: PPO | None = None

    def select_agent(
        self,
        population: list[Individual],
        state: dict[str, Any],
        observation: NDArray[np.float32] | None = None,
    ) -> SpecialistAgent:
        """
        Select which specialist agent to use.

        Args:
            population: Current GA population
            state: Search state dictionary
            observation: State observation (for meta-agent)

        Returns:
            Selected specialist agent
        """
        if self.strategy == "state_based":
            return self._select_state_based(population, state)

        if self.strategy == "ucb":
            return self._select_ucb()

        if self.strategy == "meta_agent":
            if observation is not None:
                return self._select_meta_agent(observation)
            # Fallback when no observation provided
            return self.agents[1]  # OptimizerAgent

        # Fallback: round-robin
        idx = self.total_selections % len(self.agents)
        self.total_selections += 1
        return self.agents[idx]

    def _select_state_based(
        self, population: list[Individual], state: dict[str, Any]
    ) -> SpecialistAgent:
        """
        Select agent based on search state (priority order).

        Priority:
        1. RepairAgent if infeasible
        2. ExplorerAgent if stagnated
        3. IntensifierAgent if near-optimal late in search
        4. OptimizerAgent otherwise
        """
        # Check each specialist's activation condition (priority order)
        for agent in self.agents:
            if agent.should_activate(population, state):
                return agent

        # Default: OptimizerAgent
        return self.agents[1]  # OptimizerAgent

    def _select_ucb(self, exploration_param: float = 2.0) -> SpecialistAgent:
        """
        Select agent using Upper Confidence Bound algorithm.

        UCB(i) = avg_reward(i) + c * sqrt(ln(N) / n(i))

        Args:
            exploration_param: Exploration parameter (c)

        Returns:
            Agent with highest UCB score
        """
        self.total_selections += 1

        # Calculate UCB scores
        ucb_scores = np.zeros(len(self.agents))
        for i in range(len(self.agents)):
            if self.ucb_counts[i] == 0:
                # Unvisited agent: infinite UCB (explore first)
                ucb_scores[i] = float("inf")
            else:
                avg_reward = self.ucb_rewards[i] / self.ucb_counts[i]
                exploration = exploration_param * np.sqrt(
                    np.log(self.total_selections) / self.ucb_counts[i]
                )
                ucb_scores[i] = avg_reward + exploration

        # Select agent with highest UCB
        selected_idx = int(np.argmax(ucb_scores))
        return self.agents[selected_idx]

    def _select_meta_agent(self, observation: NDArray[np.float32]) -> SpecialistAgent:
        """
        Use meta-agent to select specialist.

        Meta-agent action space: [0, 1, 2, 3] → [Repair, Optimizer, Explorer, Intensifier]

        Args:
            observation: State observation

        Returns:
            Selected specialist agent
        """
        if self.meta_agent is None:
            # Lazy load meta-agent
            meta_model_path = self.config.get("meta_agent_model_path")
            if meta_model_path:
                try:
                    from stable_baselines3 import PPO

                    loaded_model = PPO.load(meta_model_path)
                    self.meta_agent = loaded_model
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "Failed to load meta-agent: %s", e
                    )
                    self.meta_agent = None

        if self.meta_agent and observation is not None:
            action, _ = self.meta_agent.predict(observation, deterministic=True)
            agent_idx = int(action) % len(self.agents)
            return self.agents[agent_idx]
        # Fallback: state-based
        return self.agents[1]  # OptimizerAgent

    def update_ucb_reward(self, agent: SpecialistAgent, reward: float) -> None:
        """
        Update UCB statistics after agent execution.

        Args:
            agent: Agent that was executed
            reward: Reward received (fitness improvement)
        """
        # Find agent index
        agent_idx = self.agents.index(agent)

        # Update statistics
        self.ucb_counts[agent_idx] += 1
        self.ucb_rewards[agent_idx] += reward

    def get_agent_statistics(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for all agents.

        Returns:
            Dictionary mapping agent name to statistics
        """
        stats = {}
        for i, agent in enumerate(self.agents):
            stats[agent.name] = {
                "activation_count": agent.activation_count,
                "success_count": agent.success_count,
                "success_rate": agent.get_success_rate(),
            }

            if self.strategy == "ucb":
                stats[agent.name]["ucb_selections"] = int(self.ucb_counts[i])
                avg_reward = (
                    self.ucb_rewards[i] / self.ucb_counts[i]
                    if self.ucb_counts[i] > 0
                    else 0.0
                )
                stats[agent.name]["avg_ucb_reward"] = avg_reward

        return stats

    def reset_statistics(self) -> None:
        """Reset all agent statistics."""
        for agent in self.agents:
            agent.activation_count = 0
            agent.success_count = 0

        self.ucb_counts = np.zeros(len(self.agents))
        self.ucb_rewards = np.zeros(len(self.agents))
        self.total_selections = 0


class RankBasedCoordinator(AgentCoordinator):
    """
    Coordinator for rank-based multi-agent RL (ENHANCEMENT #8).

    Selects specialist based on solution's Pareto rank:
    - Rank 1: IntensifierAgent (careful refinement)
    - Rank 2: OptimizerAgent (standard optimization)
    - Rank 3: ExplorerAgent (moderate exploration)
    - Rank 4+: RepairAgent (aggressive repair)
    """

    def select_agent_by_rank(
        self, pareto_rank: int, max_rank: int = 4
    ) -> SpecialistAgent:
        """
        Select agent based on Pareto rank.

        Args:
            pareto_rank: Solution's Pareto rank (1 = best)
            max_rank: Maximum rank to consider

        Returns:
            Specialist agent for this rank
        """
        if pareto_rank == 1:
            # Elite solutions: careful refinement
            return self.agents[3]  # IntensifierAgent
        if pareto_rank == 2:
            # Good solutions: standard optimization
            return self.agents[1]  # OptimizerAgent
        if pareto_rank == 3:
            # Moderate solutions: exploration
            return self.agents[2]  # ExplorerAgent
        # Poor solutions: aggressive repair
        return self.agents[0]  # RepairAgent
