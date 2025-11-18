"""
Probability policy for adaptive operator probabilities.

ENHANCEMENT #3: RL learns to tune crossover/mutation probabilities dynamically.
"""

from typing import Tuple, Dict
import numpy as np
from numpy.typing import NDArray


class ProbabilityPolicy:
    """
    RL policy for adaptive operator probabilities.

    Instead of fixed cxpb=0.7, mutpb=0.2, the RL agent learns to adjust
    these probabilities based on search state (convergence, diversity, stagnation).

    Action space:
    - Option 1: Discrete (9 actions): 3 levels for cxpb × 3 levels for mutpb
    - Option 2: Continuous (2 actions): [cxpb, mutpb] in [0, 1]²

    Benefits:
    - Early search: High mutation for exploration
    - Late search: High crossover for exploitation
    - Stagnation: Boost mutation to escape
    - Converged: Reduce both to avoid disruption
    """

    def __init__(self, mode: str = "discrete", config: Dict = None):
        """
        Initialize probability policy.

        Args:
            mode: 'discrete' or 'continuous'
            config: Configuration with probability levels
        """
        self.mode = mode
        self.config = config or {}

        # Default probability levels for discrete mode
        self.cxpb_levels = self.config.get("cxpb_levels", [0.5, 0.75, 0.9])
        self.mutpb_levels = self.config.get("mutpb_levels", [0.1, 0.25, 0.4])

        # Current probabilities
        self.current_cxpb = 0.75
        self.current_mutpb = 0.25

    def get_action_space_size(self) -> int:
        """
        Get size of action space.

        Returns:
            Number of discrete actions (discrete mode) or dimension (continuous)
        """
        if self.mode == "discrete":
            return len(self.cxpb_levels) * len(self.mutpb_levels)  # 9 actions
        else:
            return 2  # [cxpb, mutpb]

    def action_to_probabilities(
        self, action: int | NDArray[np.float32]
    ) -> Tuple[float, float]:
        """
        Convert RL action to (cxpb, mutpb) probabilities.

        Args:
            action: Discrete action index (0-8) or continuous values [cxpb, mutpb]

        Returns:
            (crossover_prob, mutation_prob)
        """
        if self.mode == "discrete":
            # Decode discrete action to probability grid
            action_int = (
                int(action) if isinstance(action, (int, np.integer)) else int(action[0])
            )
            cx_idx = action_int // len(self.mutpb_levels)
            mut_idx = action_int % len(self.mutpb_levels)

            cxpb = self.cxpb_levels[cx_idx]
            mutpb = self.mutpb_levels[mut_idx]
        else:
            # Continuous action [cxpb, mutpb]
            action_array = np.asarray(action)
            cxpb = float(np.clip(action_array[0], 0.0, 1.0))
            mutpb = float(np.clip(action_array[1], 0.0, 1.0))

        # Update current state
        self.current_cxpb = cxpb
        self.current_mutpb = mutpb

        return (cxpb, mutpb)

    def get_current_probabilities(self) -> Tuple[float, float]:
        """
        Get current operator probabilities.

        Returns:
            (crossover_prob, mutation_prob)
        """
        return (self.current_cxpb, self.current_mutpb)

    def get_exploration_bonus(
        self, diversity: float, stagnation: int
    ) -> Tuple[float, float]:
        """
        Calculate exploration bonus for probabilities.

        When diversity is low or stagnation is high, boost mutation.

        Args:
            diversity: Population diversity in [0, 1]
            stagnation: Generations without improvement

        Returns:
            (cxpb_bonus, mutpb_bonus) to add to base probabilities
        """
        # Low diversity → increase mutation
        diversity_bonus = 0.0
        if diversity < 0.2:
            diversity_bonus = 0.1 * (0.2 - diversity)

        # Stagnation → increase mutation
        stagnation_bonus = 0.0
        if stagnation > 10:
            stagnation_bonus = min(0.2, 0.01 * (stagnation - 10))

        # Apply bonuses (more mutation, less crossover)
        mutpb_bonus = diversity_bonus + stagnation_bonus
        cxpb_bonus = -0.5 * mutpb_bonus  # Reduce crossover when boosting mutation

        return (cxpb_bonus, mutpb_bonus)

    def apply_constraints(self, cxpb: float, mutpb: float) -> Tuple[float, float]:
        """
        Apply constraints to probabilities.

        Ensures:
        - Both probabilities in [0, 1]
        - Sum doesn't exceed reasonable limit (e.g., 1.2)
        - Minimum values for both (e.g., 0.05)

        Args:
            cxpb: Crossover probability
            mutpb: Mutation probability

        Returns:
            Constrained (cxpb, mutpb)
        """
        # Clip to [0.05, 1.0]
        cxpb = np.clip(cxpb, 0.05, 1.0)
        mutpb = np.clip(mutpb, 0.05, 1.0)

        # Ensure sum doesn't exceed 1.2 (avoid too much disruption)
        if cxpb + mutpb > 1.2:
            scale = 1.2 / (cxpb + mutpb)
            cxpb *= scale
            mutpb *= scale

        return (float(cxpb), float(mutpb))

    def reset(self) -> None:
        """Reset policy to default probabilities."""
        self.current_cxpb = 0.75
        self.current_mutpb = 0.25


class ProbabilitySchedule:
    """
    Fixed schedule for probability adaptation (non-RL baseline).

    Useful for comparison with RL-based adaptation.
    """

    def __init__(self, strategy: str = "linear"):
        """
        Initialize probability schedule.

        Args:
            strategy: 'linear', 'exponential', or 'step'
        """
        self.strategy = strategy

    def get_probabilities(
        self, generation: int, max_generations: int
    ) -> Tuple[float, float]:
        """
        Get probabilities based on generation progress.

        Args:
            generation: Current generation
            max_generations: Total generations

        Returns:
            (cxpb, mutpb) for this generation
        """
        progress = generation / max(max_generations, 1)  # [0, 1]

        if self.strategy == "linear":
            # Linear transition: high mutation → high crossover
            mutpb = 0.4 * (1 - progress) + 0.1 * progress
            cxpb = 0.5 * (1 - progress) + 0.9 * progress

        elif self.strategy == "exponential":
            # Exponential decay of mutation
            mutpb = 0.4 * np.exp(-3 * progress)
            cxpb = 0.9 * (1 - np.exp(-3 * progress))

        elif self.strategy == "step":
            # Step function: switch at 50% progress
            if progress < 0.5:
                cxpb, mutpb = 0.6, 0.3  # Exploration
            else:
                cxpb, mutpb = 0.9, 0.1  # Exploitation

        else:
            # Default: fixed probabilities
            cxpb, mutpb = 0.75, 0.25

        return (float(cxpb), float(mutpb))
