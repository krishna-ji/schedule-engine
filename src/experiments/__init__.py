"""
Experiments Package

OOP Framework for running scheduling experiments.

Provides:
- BaseExperiment: Abstract base class for all experiments
- BaselineExperiment: Pure NSGA-II (Mode A)
- MemeticExperiment: NSGA-II + Local Search (Mode B)
- RoundRobinExperiment: Round-robin heuristic selection (Mode C)
- AdaptiveExperiment: UCB-based adaptive selection (Mode D)
- RLGuidedExperiment: Q-learning guided selection (Mode E)

Usage:
    from src.experiments import BaselineExperiment

    exp = BaselineExperiment(
        seed=42,
        pop_size=50,
        ngen=100,
        output_dir="output/baseline",
    )
    exp.run()
"""

from src.experiments.base import BaseExperiment
from src.experiments.modes.adaptive import AdaptiveExperiment
from src.experiments.modes.baseline import BaselineExperiment
from src.experiments.modes.cp_hybrid import CPHybridExperiment
from src.experiments.modes.memetic import MemeticExperiment
from src.experiments.modes.roundrobin import RoundRobinExperiment
from src.experiments.modes.ultimate import UltimateExperiment
from src.experiments.output.base import BaseExporter
from src.experiments.output.repair_exporter import RepairExporter

__all__ = [
    "AdaptiveExperiment",
    "BaseExperiment",
    "BaseExporter",
    "BaselineExperiment",
    "CPHybridExperiment",
    "MemeticExperiment",
    "RepairExporter",
    "RoundRobinExperiment",
    "UltimateExperiment",
]
