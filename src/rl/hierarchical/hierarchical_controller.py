"""
Hierarchical RL controller for two-level heuristic selection.

ENHANCEMENT #7: High-level selects category, low-level selects specific heuristic.
"""

import numpy as np
from numpy.typing import NDArray


class HighLevelPolicy:
    """
    High-level policy: Selects heuristic category.

    Action space (5 categories):
    0. Construction
    1. Perturbation
    2. Improvement
    3. Diversity
    4. Meta

    Reduces action space from 19 → 5 for faster learning.
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize high-level policy.

        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model: PPO | None = None  # type: ignore[name-defined]
        self.categories = [
            "construction",
            "perturbation",
            "improvement",
            "diversity",
            "meta",
        ]

    def select_category(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> str:
        """
        Select heuristic category.

        Args:
            observation: State observation
            deterministic: Use deterministic policy

        Returns:
            Category name
        """
        if self.model is None:
            self._load_model()

        if self.model:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            category_idx = int(action) % len(self.categories)
            return self.categories[category_idx]  # type: ignore[no-any-return]
        else:
            # Fallback: round-robin or random
            return np.random.choice(self.categories)  # type: ignore[no-any-return]

    def _load_model(self) -> None:
        """Load trained model."""
        if self.model_path:
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(self.model_path)
            except Exception as e:
                print(f"Warning: Failed to load high-level policy: {e}")


class LowLevelPolicy:
    """
    Low-level policy: Selects specific heuristic within category.

    Each category has its own low-level policy:
    - Construction: 3 heuristics
    - Perturbation: 5 heuristics
    - Improvement: 3 heuristics
    - Diversity: 4 heuristics
    - Meta: 4 heuristics
    """

    def __init__(self, category: str, model_path: str | None = None):
        """
        Initialize low-level policy for category.

        Args:
            category: Heuristic category
            model_path: Path to trained model
        """
        self.category = category
        self.model_path = model_path
        self.model: PPO | None = None  # type: ignore[name-defined]

        # Heuristic mappings per category
        self.heuristics = {
            "construction": [
                0,
                1,
                2,
            ],  # largest_degree_first, most_constrained_first, earliest_deadline_first
            "perturbation": [
                3,
                4,
                5,
                6,
                7,
            ],  # random_swap, temporal_shift, room_shuffle, instructor_reassign, multi_perturbation
            "improvement": [
                10,
                11,
                12,
            ],  # kempe_chain, ejection_chain, variable_depth_search
            "diversity": [
                13,
                14,
                15,
                16,
            ],  # distance_preserving_crossover, crowding_mutation, niching_selection, adaptive_diversity_maintenance
            "meta": [
                17,
                18,
                19,
                20,
            ],  # variable_neighborhood_descent, iterated_local_search, adaptive_large_neighborhood, guided_local_search
        }

    def select_heuristic(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> int:
        """
        Select specific heuristic within category.

        Args:
            observation: State observation
            deterministic: Use deterministic policy

        Returns:
            Heuristic ID (global index)
        """
        if self.model is None:
            self._load_model()

        category_heuristics = self.heuristics[self.category]

        if self.model:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            local_idx = int(action) % len(category_heuristics)
            return category_heuristics[local_idx]  # type: ignore[no-any-return]
        else:
            # Fallback: random from category
            return np.random.choice(category_heuristics)  # type: ignore[no-any-return]

    def _load_model(self) -> None:
        """Load trained model."""
        if self.model_path:
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(self.model_path)
            except Exception as e:
                print(
                    f"Warning: Failed to load low-level policy for {self.category}: {e}"
                )


class HierarchicalController:
    """
    Hierarchical controller combining high-level and low-level policies.

    Two-level decision making:
    1. High-level: Which category? (5 actions)
    2. Low-level: Which heuristic in that category? (3-5 actions)

    Benefits:
    - Smaller action spaces (faster learning)
    - Shared knowledge within categories
    - More interpretable decisions
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize hierarchical controller.

        Args:
            config: Configuration with model paths
        """
        self.config = config or {}

        # High-level policy
        self.high_level = HighLevelPolicy(self.config.get("high_level_model_path"))

        # Low-level policies (one per category)
        self.low_level_policies = {
            category: LowLevelPolicy(
                category, self.config.get(f"{category}_model_path")
            )
            for category in self.high_level.categories
        }

        # Statistics
        self.category_counts = dict.fromkeys(self.high_level.categories, 0)
        self.heuristic_counts: dict[int, int] = {}

    def select_heuristic(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> int:
        """
        Select heuristic using hierarchical policy.

        Args:
            observation: State observation
            deterministic: Use deterministic policies

        Returns:
            Heuristic ID
        """
        # Step 1: High-level selects category
        category = self.high_level.select_category(observation, deterministic)
        self.category_counts[category] += 1

        # Step 2: Low-level selects heuristic within category
        heuristic_id = self.low_level_policies[category].select_heuristic(
            observation, deterministic
        )
        self.heuristic_counts[heuristic_id] = (
            self.heuristic_counts.get(heuristic_id, 0) + 1
        )

        return heuristic_id

    def get_statistics(self) -> dict:
        """Get usage statistics."""
        return {
            "category_counts": self.category_counts,
            "heuristic_counts": self.heuristic_counts,
            "total_selections": sum(self.category_counts.values()),
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.category_counts = dict.fromkeys(self.high_level.categories, 0)
        self.heuristic_counts = {}
