"""GA Core: Fundamental data types and evaluation for the genetic algorithm.

Provides:
    - get_creator: Centralized DEAP creator registry
    - create_individual: Factory for DEAP Individual wrapping SessionGenes
    - PopulationFactory: Unified population creation API
    - evaluate / evaluate_detailed: Fitness evaluation functions
    - quanta_list_to_contiguous: Legacy gene format converter
"""

from __future__ import annotations

from schedule_engine.ga.core.creator_registry import get_creator
from schedule_engine.ga.core.evaluator import evaluate, evaluate_detailed
from schedule_engine.ga.core.individual import create_individual
from schedule_engine.ga.core.population_factory import PopulationFactory
from schedule_engine.ga.core.quanta_converter import quanta_list_to_contiguous

__all__ = [
    "get_creator",
    "create_individual",
    "evaluate",
    "evaluate_detailed",
    "PopulationFactory",
    "quanta_list_to_contiguous",
]
