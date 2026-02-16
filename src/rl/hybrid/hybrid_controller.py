"""
Hybrid controller for RL-based heuristic selection.

Combines RL agent with fallback strategies for robust production deployment.

Supports 3 modes:
- RL-Primary: Always use RL, fallback only on failure
- RL-Fallback: Try RL with timeout, fallback on timeout/error
- RL-Assisted: Mix RL (80%) with exploration (20%)
"""

import random
from enum import Enum
from typing import Any

import numpy as np

from src.rl.deployment.inference import RLInference
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class HybridMode(Enum):
    """Hybrid controller operating modes."""

    RL_PRIMARY = "rl_primary"  # Trust RL completely
    RL_FALLBACK = "rl_fallback"  # Try RL, fallback on timeout
    RL_ASSISTED = "rl_assisted"  # Mix RL with exploration


class FallbackStrategy(Enum):
    """Fallback strategies when RL unavailable."""

    RANDOM = "random"  # Random selection
    GREEDY = "greedy"  # Highest priority action
    ROUND_ROBIN = "round_robin"  # Cycle through actions
    RECENT_BEST = "recent_best"  # Action with best recent reward


class HybridController:
    """
    Manages hybrid RL + heuristic selection.

    Provides robust action selection by combining RL agent
    with fallback strategies. Tracks usage statistics and
    adapts to RL performance.
    """

    def __init__(
        self,
        rl_inference: RLInference,
        mode: HybridMode = HybridMode.RL_PRIMARY,
        fallback_strategy: FallbackStrategy = FallbackStrategy.RANDOM,
        rl_probability: float = 0.8,
        enable_action_masking: bool = True,
    ):
        """
        Initialize hybrid controller.

        Args:
            rl_inference: RL inference engine
            mode: Operating mode
            fallback_strategy: Fallback strategy
            rl_probability: Probability of using RL in RL_ASSISTED mode
            enable_action_masking: Mask invalid actions
        """
        self.rl_inference = rl_inference
        self.mode = mode
        self.fallback_strategy = fallback_strategy
        self.rl_probability = rl_probability
        self.enable_action_masking = enable_action_masking

        # Usage statistics
        self.rl_calls = 0
        self.fallback_calls = 0
        self.total_calls = 0

        # Round-robin state
        self.round_robin_idx = 0

        # Recent rewards for RECENT_BEST strategy
        self.recent_rewards: dict[int, list[float]] = {}
        self.recent_window = 10

        logger.info(
            f"Initialized HybridController "
            f"(mode={mode.value}, fallback={fallback_strategy.value})"
        )

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: list[int] | None = None,
        deterministic: bool = True,
    ) -> int:
        """
        Select action using hybrid strategy.

        Args:
            state: Current environment state
            valid_actions: List of valid actions (for masking)
            deterministic: Use deterministic RL policy

        Returns:
            Selected action
        """
        self.total_calls += 1

        # Select based on mode
        if self.mode == HybridMode.RL_PRIMARY:
            action = self._rl_primary(state, valid_actions, deterministic)
        elif self.mode == HybridMode.RL_FALLBACK:
            action = self._rl_fallback(state, valid_actions, deterministic)
        elif self.mode == HybridMode.RL_ASSISTED:
            action = self._rl_assisted(state, valid_actions, deterministic)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Validate action
        if valid_actions and action not in valid_actions:
            logger.warning(
                f"Invalid action {action}, falling back to {self.fallback_strategy.value}"
            )
            action = self._fallback_action(valid_actions)
            self.fallback_calls += 1

        return action

    def _rl_primary(
        self,
        state: np.ndarray,
        valid_actions: list[int] | None,
        deterministic: bool,
    ) -> int:
        """RL-Primary mode: Always use RL, fallback only on failure."""
        action = self.rl_inference.predict_action(state, deterministic=deterministic)

        if action is not None:
            self.rl_calls += 1
            return action
        logger.warning("RL prediction failed, using fallback")
        self.fallback_calls += 1
        return self._fallback_action(valid_actions)

    def _rl_fallback(
        self,
        state: np.ndarray,
        valid_actions: list[int] | None,
        deterministic: bool,
    ) -> int:
        """RL-Fallback mode: Try RL with timeout, fallback on timeout/error."""
        try:
            action = self.rl_inference.predict_action(
                state, deterministic=deterministic
            )

            if action is not None:
                self.rl_calls += 1
                return action
        except Exception as e:
            logger.warning(f"RL inference exception: {e}")

        # Fallback
        self.fallback_calls += 1
        return self._fallback_action(valid_actions)

    def _rl_assisted(
        self,
        state: np.ndarray,
        valid_actions: list[int] | None,
        deterministic: bool,
    ) -> int:
        """RL-Assisted mode: Mix RL with exploration."""
        if random.random() < self.rl_probability:
            # Use RL
            action = self.rl_inference.predict_action(
                state, deterministic=deterministic
            )

            if action is not None:
                self.rl_calls += 1
                return action

        # Use fallback (exploration)
        self.fallback_calls += 1
        return self._fallback_action(valid_actions)

    def _fallback_action(self, valid_actions: list[int] | None) -> int:
        """Execute fallback strategy."""
        if not valid_actions:
            valid_actions = list(range(20))  # Default action space

        if self.fallback_strategy == FallbackStrategy.RANDOM:
            return random.choice(valid_actions)

        if self.fallback_strategy == FallbackStrategy.GREEDY:
            # Return first action (assume sorted by priority)
            return valid_actions[0]

        if self.fallback_strategy == FallbackStrategy.ROUND_ROBIN:
            # Cycle through actions
            action = valid_actions[self.round_robin_idx % len(valid_actions)]
            self.round_robin_idx += 1
            return action

        if self.fallback_strategy == FallbackStrategy.RECENT_BEST:
            # Select action with best recent reward
            if not self.recent_rewards:
                return random.choice(valid_actions)

            # Filter to valid actions
            valid_recent = {
                a: rewards
                for a, rewards in self.recent_rewards.items()
                if a in valid_actions
            }

            if not valid_recent:
                return random.choice(valid_actions)

            # Select best
            best_action: int = max(valid_recent, key=lambda a: np.mean(valid_recent[a]))
            return best_action

        # Default to random
        return random.choice(valid_actions)

    def update_reward(self, action: int, reward: float) -> None:
        """
        Update recent reward for action (used by RECENT_BEST strategy).

        Args:
            action: Action taken
            reward: Reward received
        """
        if action not in self.recent_rewards:
            self.recent_rewards[action] = []

        self.recent_rewards[action].append(reward)

        # Keep only recent window
        if len(self.recent_rewards[action]) > self.recent_window:
            self.recent_rewards[action].pop(0)

    def get_statistics(self) -> dict[str, float | int | str]:
        """Get usage statistics."""
        rl_pct = 100 * self.rl_calls / self.total_calls if self.total_calls > 0 else 0
        fallback_pct = (
            100 * self.fallback_calls / self.total_calls if self.total_calls > 0 else 0
        )

        return {
            "total_calls": self.total_calls,
            "rl_calls": self.rl_calls,
            "fallback_calls": self.fallback_calls,
            "rl_percentage": rl_pct,
            "fallback_percentage": fallback_pct,
            "mode": self.mode.value,
            "fallback_strategy": self.fallback_strategy.value,
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.rl_calls = 0
        self.fallback_calls = 0
        self.total_calls = 0
        self.round_robin_idx = 0
        logger.debug("Reset hybrid controller statistics")

    def set_mode(self, mode: HybridMode) -> None:
        """Change operating mode."""
        old_mode = self.mode
        self.mode = mode
        logger.info(f"Changed mode: {old_mode.value} -> {mode.value}")

    def set_fallback_strategy(self, strategy: FallbackStrategy) -> None:
        """Change fallback strategy."""
        old_strategy = self.fallback_strategy
        self.fallback_strategy = strategy
        logger.info(
            f"Changed fallback strategy: {old_strategy.value} -> {strategy.value}"
        )


def create_hybrid_controller(
    rl_inference: RLInference,
    mode: str = "rl_primary",
    fallback_strategy: str = "random",
    **kwargs: Any,
) -> HybridController:
    """
    Convenience function to create hybrid controller.

    Args:
        rl_inference: RL inference engine
        mode: Operating mode (string)
        fallback_strategy: Fallback strategy (string)
        **kwargs: Additional arguments

    Returns:
        Configured HybridController
    """
    # Parse enums
    mode_enum = HybridMode(mode)
    fallback_enum = FallbackStrategy(fallback_strategy)

    return HybridController(
        rl_inference=rl_inference,
        mode=mode_enum,
        fallback_strategy=fallback_enum,
        **kwargs,
    )
