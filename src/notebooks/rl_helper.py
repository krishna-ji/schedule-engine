"""RL integration helpers for experiment notebooks.

Provides simplified RL-guided heuristic selection for Mode E.
For full RL training, use the scripts/training/ module.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.ga.sessiongene import SessionGene

if TYPE_CHECKING:
    from src.notebooks.data_loader import ScheduleData


class SimpleRLSelector:
    """Simple RL-like heuristic selector using Q-learning.

    This is a simplified version for notebooks. For production RL,
    use the full src.rl module with trained PPO/DQN agents.
    """

    def __init__(
        self,
        heuristic_names: list[str] | None = None,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
    ) -> None:
        """Initialize Q-learning based selector.

        Args:
            heuristic_names: List of heuristic names
            learning_rate: Q-value update rate
            discount: Future reward discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            min_epsilon: Minimum epsilon
        """
        from src.notebooks.heuristics import HEURISTICS

        self.heuristics = heuristic_names or list(HEURISTICS.keys())
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table: state -> action -> value
        # State = discretized constraint profile
        self.q_table: dict[str, dict[str, float]] = defaultdict(
            lambda: {h: 0.0 for h in self.heuristics}
        )

        # Statistics
        self.episode_rewards: list[float] = []
        self.action_counts: dict[str, int] = defaultdict(int)

    def _get_state(
        self,
        individual: list[SessionGene],
        data: ScheduleData,
    ) -> str:
        """Discretize constraint profile into state string."""
        from src.notebooks.evaluation import evaluate_constraints

        result = evaluate_constraints(individual, data)

        # Discretize each constraint into low/med/high
        def bucket(v: int) -> str:
            if v == 0:
                return "0"
            elif v < 100:
                return "L"
            elif v < 300:
                return "M"
            else:
                return "H"

        state_parts = [
            f"sg{bucket(result.breakdown.get('student_group_exclusivity', 0))}",
            f"in{bucket(result.breakdown.get('instructor_exclusivity', 0))}",
            f"rm{bucket(result.breakdown.get('room_exclusivity', 0))}",
        ]
        return "_".join(state_parts)

    def select_action(self, state: str) -> str:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            # Explore
            return random.choice(self.heuristics)
        else:
            # Exploit
            q_values = self.q_table[state]
            return max(q_values, key=lambda k: q_values[k])

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
    ) -> None:
        """Update Q-value using Q-learning."""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def apply(
        self,
        individual: list[SessionGene],
        data: ScheduleData,
        evaluate_fn: Any,
    ) -> tuple[str, int, float]:
        """Select and apply action using RL policy.

        Returns:
            Tuple of (action_name, fixes, reward)
        """
        from src.notebooks.heuristics import HEURISTICS

        # Get current state and fitness
        state = self._get_state(individual, data)
        old_fitness = evaluate_fn(individual)
        old_hard = old_fitness[0]

        # Select and apply action
        action = self.select_action(state)
        self.action_counts[action] += 1

        fn = HEURISTICS[action]
        fixes = fn(individual, data)

        # Get new state and fitness
        next_state = self._get_state(individual, data)
        new_fitness = evaluate_fn(individual)
        new_hard = new_fitness[0]

        # Calculate reward (improvement in hard constraints)
        improvement = old_hard - new_hard
        reward = float(improvement) if improvement > 0 else -1.0

        # Update Q-values
        self.update(state, action, reward, next_state)

        return action, fixes, reward

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        return {
            "epsilon": self.epsilon,
            "action_counts": dict(self.action_counts),
            "num_states": len(self.q_table),
        }


def load_trained_agent(model_dir: str | Path = "models/rl_agents") -> Any:
    """Load a trained RL agent from disk.

    Args:
        model_dir: Directory containing trained models

    Returns:
        Trained agent or None if not found
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return None

    # Look for latest model
    registry_file = model_path / "registry.json"
    if registry_file.exists():
        import json

        with open(registry_file) as f:
            registry = json.load(f)
            if registry.get("models"):
                latest = registry["models"][-1]
                print(f"Found trained model: {latest.get('name', 'unknown')}")
                # Return model info (actual loading requires stable-baselines3)
                return latest

    return None
