"""
Modes Package - Experiment mode implementations.

Provides:
- BaselineExperiment: Pure NSGA-II (Mode A)
- MemeticExperiment: NSGA-II + Local Search (Mode B)
- RoundRobinExperiment: Round-robin heuristic selection (Mode C)
- AdaptiveExperiment: UCB adaptive selection (Mode D)
- RLGuidedExperiment: Q-learning guided selection (Mode E)
"""

from schedule_engine.experiments.modes.adaptive import AdaptiveExperiment
from schedule_engine.experiments.modes.baseline import BaselineExperiment
from schedule_engine.experiments.modes.memetic import MemeticExperiment
from schedule_engine.experiments.modes.rl_guided import RLGuidedExperiment
from schedule_engine.experiments.modes.roundrobin import RoundRobinExperiment
from schedule_engine.experiments.modes.ultimate import UltimateExperiment

__all__ = [
    "BaselineExperiment",
    "MemeticExperiment",
    "RoundRobinExperiment",
    "AdaptiveExperiment",
    "RLGuidedExperiment",
    "UltimateExperiment",
]
