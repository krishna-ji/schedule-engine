"""GA module: Genetic Algorithm components for schedule optimization.

Exposes:
    - GAScheduler, GAConfig, GAMetrics: Main scheduler classes
    - get_creator: Centralized DEAP creator registry
    - create_individual: Factory function for creating individuals
    - SessionGene: Gene representation for course sessions
    - RepairPipeline: Unified repair operations interface
    - PopulationFactory: Single entry point for population creation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from schedule_engine.domain.gene import SessionGene
from schedule_engine.ga.creator_registry import get_creator
from schedule_engine.ga.population_factory import PopulationFactory
from schedule_engine.ga.repair_pipeline import RepairPipeline
from schedule_engine.ga.scheduler import GAConfig, GAMetrics, GAScheduler

if TYPE_CHECKING:
    from schedule_engine.domain.types import Individual

__all__ = [
    "GAScheduler",
    "GAConfig",
    "GAMetrics",
    "get_creator",
    "create_individual",
    "SessionGene",
    "RepairPipeline",
    "PopulationFactory",
]


def create_individual(gene_list: list[SessionGene]) -> Individual:
    """Lazy import wrapper to avoid circular dependency during package init."""

    from schedule_engine.ga.individual import create_individual as _create_individual

    return _create_individual(gene_list)
