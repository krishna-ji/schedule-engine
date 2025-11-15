"""
Random agent baseline for RL evaluation.

Simple baseline that selects actions uniformly at random.
"""

from typing import Any, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym


class RandomAgent:
    """
    Random baseline agent.

    Selects actions uniformly at random from the action space.
    Used as a baseline for comparing RL agent performance.
    """

    def __init__(self, env: gym.Env, seed: int | None = None):
        """
        Initialize random agent.

        Args:
            env: Gymnasium environment
            seed: Random seed for reproducibility
        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.rng = np.random.default_rng(seed)

    def predict(
        self,
        observation: NDArray[np.float32],
        deterministic: bool = False,
    ) -> Tuple[int, None]:
        """
        Predict action (randomly).

        Args:
            observation: Current observation (unused)
            deterministic: Whether to be deterministic (no effect for random)

        Returns:
            (action, state) - state is None for random agent
        """
        action = self.rng.integers(0, self.action_space.n)
        return int(action), None

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 100,
        **kwargs,
    ) -> "RandomAgent":
        """
        Dummy learn method (does nothing - random agent doesn't learn).

        Args:
            total_timesteps: Number of timesteps (ignored)
            callback: Callback function (ignored)
            log_interval: Logging interval (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Self (unchanged)
        """
        # Random agent doesn't learn - just return self
        return self

    def save(self, path: str) -> None:
        """
        Dummy save method (nothing to save for random agent).

        Args:
            path: Save path (ignored)
        """
        pass

    @staticmethod
    def load(path: str, env: gym.Env, **kwargs) -> "RandomAgent":
        """
        Dummy load method (creates new random agent).

        Args:
            path: Load path (ignored)
            env: Environment
            **kwargs: Additional arguments

        Returns:
            New RandomAgent instance
        """
        return RandomAgent(env)

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "type": "random",
            "action_space": str(self.action_space),
            "observation_space": str(self.observation_space),
        }


def create_random_agent(env: gym.Env, seed: int | None = None) -> RandomAgent:
    """
    Factory function to create random agent.

    Args:
        env: Gymnasium environment
        seed: Random seed

    Returns:
        RandomAgent instance
    """
    return RandomAgent(env, seed=seed)
