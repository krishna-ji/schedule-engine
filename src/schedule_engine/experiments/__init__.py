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
    from schedule_engine.experiments import BaselineExperiment

    exp = BaselineExperiment(
        seed=42,
        pop_size=50,
        ngen=100,
        output_dir="output/baseline",
    )
    exp.run()
"""

from schedule_engine.experiments.base import BaseExperiment
from schedule_engine.experiments.modes.adaptive import AdaptiveExperiment
from schedule_engine.experiments.modes.baseline import BaselineExperiment
from schedule_engine.experiments.modes.memetic import MemeticExperiment
from schedule_engine.experiments.modes.rl_guided import RLGuidedExperiment
from schedule_engine.experiments.modes.roundrobin import RoundRobinExperiment
from schedule_engine.experiments.modes.ultimate import UltimateExperiment
from schedule_engine.experiments.output.base import BaseExporter
from schedule_engine.experiments.output.repair_exporter import RepairExporter
from schedule_engine.experiments.output.rl_exporter import RLExporter

__all__ = [
    "BaseExperiment",
    "BaseExporter",
    "RepairExporter",
    "RLExporter",
    "BaselineExperiment",
    "MemeticExperiment",
    "RoundRobinExperiment",
    "AdaptiveExperiment",
    "RLGuidedExperiment",
    "UltimateExperiment",
]
