"""GA Core: Fundamental data types and evaluation for the genetic algorithm.

Provides:
    - PopulationFactory: Unified population creation API
    - evaluate / evaluate_detailed: Fitness evaluation functions
    - quanta_list_to_contiguous: Legacy gene format converter
"""

from __future__ import annotations

from src.ga.core.evaluator import evaluate, evaluate_detailed
from src.ga.core.population_factory import PopulationFactory
from src.ga.core.quanta_converter import quanta_list_to_contiguous

__all__ = [
    "PopulationFactory",
    "evaluate",
    "evaluate_detailed",
    "quanta_list_to_contiguous",
]
