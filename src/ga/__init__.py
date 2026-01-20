"""
GA module: Genetic Algorithm components for schedule optimization.

Exposes:
    - GAScheduler, GAConfig, GAMetrics: Main scheduler classes
    - get_creator: Centralized DEAP creator registry
    - create_individual: Factory function for creating individuals
    - SessionGene: Gene representation for course sessions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.gene import SessionGene
from src.ga.creator_registry import get_creator
from src.ga.scheduler import GAConfig, GAMetrics, GAScheduler

if TYPE_CHECKING:
    from src.domain.types import Individual

__all__ = [
    "GAScheduler",
    "GAConfig",
    "GAMetrics",
    "get_creator",
    "create_individual",
    "SessionGene",
]


def create_individual(gene_list: list[SessionGene]) -> Individual:
    """Lazy import wrapper to avoid circular dependency during package init."""

    from src.ga.individual import create_individual as _create_individual

    return _create_individual(gene_list)
