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

from src.domain.gene import SessionGene
from src.ga.core.creator_registry import get_creator
from src.ga.core.population_factory import PopulationFactory
from src.ga.repair.pipeline import RepairPipeline
from src.ga.scheduler import GAConfig, GAMetrics, GAScheduler

if TYPE_CHECKING:
    from src.domain.types import Individual

__all__ = [
    "GAConfig",
    "GAMetrics",
    "GAScheduler",
    "PopulationFactory",
    "RepairPipeline",
    "SessionGene",
    "create_individual",
    "get_creator",
]


def create_individual(gene_list: list[SessionGene]) -> Individual:
    """Lazy import wrapper to avoid circular dependency during package init."""

    from src.ga.core.individual import (
        create_individual as _create_individual,
    )

    return _create_individual(gene_list)
