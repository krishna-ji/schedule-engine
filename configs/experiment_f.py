"""Experiment F config instance (test profile)."""

from configs.experiments.heuristic_testing import (
    EXPERIMENT_DESCRIPTION,
    EXPERIMENT_NAME,
    HeuristicTestingTestConfig,
)

config = HeuristicTestingTestConfig()

__all__ = ["EXPERIMENT_NAME", "EXPERIMENT_DESCRIPTION", "config"]
