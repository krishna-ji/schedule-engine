"""
AdaptiveExperiment - NSGA-II + Adaptive Heuristics (Mode D).

Uses epsilon-greedy or UCB to adaptively select heuristics.

Usage:
    from schedule_engine.experiments import AdaptiveExperiment

    exp = AdaptiveExperiment(
        seed=42,
        pop_size=50,
        ngen=100,
        repair_policy="epsilon_greedy",
        repair_epsilon=0.1,
    )
    exp.run()
"""

from __future__ import annotations

from typing import Any

from schedule_engine.experiments.modes.memetic import MemeticExperiment


class AdaptiveExperiment(MemeticExperiment):
    """
    Adaptive heuristic selection experiment.

    Same as MemeticExperiment but with adaptive policies (epsilon_greedy, ucb).
    Learns which heuristics work best and adapts selection probabilities.

    Additional Parameters:
    ---------------------
    repair_policy : str
        Policy for heuristic selection. Options: "epsilon_greedy", "ucb", "softmax"
    repair_epsilon : float
        Exploration rate for epsilon-greedy (probability of random selection)
    """

    def __init__(
        self,
        *,
        repair_prob: float = 0.45,
        repair_max_steps: int = 3,
        repair_policy: str = "epsilon_greedy",
        repair_budget_ms: float = 120.0,
        repair_max_candidates: int = 30,
        repair_epsilon: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize adaptive experiment."""
        super().__init__(
            local_search_prob=repair_prob,
            local_search_iterations=repair_max_steps,
            repair_policy=repair_policy,
            repair_budget_ms=repair_budget_ms,
            repair_max_candidates=repair_max_candidates,
            repair_epsilon=repair_epsilon,
            **kwargs,
        )

    def _get_experiment_name(self) -> str:
        """Return experiment name."""
        return "ga_04_repair_bandit"
