"""
RoundRobinExperiment - NSGA-II + Round-Robin Heuristics (Mode C).

Applies heuristics in a fixed round-robin order.

Usage:
    from schedule_engine.experiments import RoundRobinExperiment

    exp = RoundRobinExperiment(
        seed=42,
        pop_size=50,
        ngen=100,
        repair_prob=0.3,
    )
    exp.run()
"""

from __future__ import annotations

from typing import Any

from schedule_engine.experiments.modes.memetic import MemeticExperiment


class RoundRobinExperiment(MemeticExperiment):
    """
    Round-Robin heuristic selection experiment.

    Same as MemeticExperiment but with round_robin policy hardcoded.
    This ensures a fair distribution of heuristic applications.

    Inherits all parameters from MemeticExperiment.
    """

    def __init__(
        self,
        *,
        repair_prob: float = 0.3,
        repair_max_steps: int = 3,
        repair_budget_ms: float = 120.0,
        repair_max_candidates: int = 30,
        repair_epsilon: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize round-robin experiment."""
        super().__init__(
            local_search_prob=repair_prob,
            local_search_iterations=repair_max_steps,
            repair_policy="round_robin",  # Fixed policy
            repair_budget_ms=repair_budget_ms,
            repair_max_candidates=repair_max_candidates,
            repair_epsilon=repair_epsilon,
            **kwargs,
        )

    def _get_experiment_name(self) -> str:
        """Return experiment name."""
        return "mode_c_roundrobin"
