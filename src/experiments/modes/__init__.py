"""
Modes Package - Experiment mode implementations.

Provides:
- BaselineExperiment: Pure NSGA-II (Mode A)
- MemeticExperiment: NSGA-II + Local Search (Mode B)
- RoundRobinExperiment: Round-robin heuristic selection (Mode C)
- AdaptiveExperiment: UCB adaptive selection (Mode D)
"""

from src.experiments.modes.adaptive import AdaptiveExperiment
from src.experiments.modes.baseline import BaselineExperiment
from src.experiments.modes.memetic import MemeticExperiment
from src.experiments.modes.roundrobin import RoundRobinExperiment
from src.experiments.modes.ultimate import UltimateExperiment

__all__ = [
    "AdaptiveExperiment",
    "BaselineExperiment",
    "MemeticExperiment",
    "RoundRobinExperiment",
    "UltimateExperiment",
]
