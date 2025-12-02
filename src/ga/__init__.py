"""
GA module: Genetic Algorithm components for schedule optimization.

Exposes:
    - get_creator: Centralized DEAP creator registry
    - create_individual: Factory function for creating individuals
    - SessionGene: Gene representation for course sessions
"""

from __future__ import annotations

from src.core.types import Individual
from src.ga.creator_registry import get_creator
from src.ga.sessiongene import SessionGene

__all__ = ["get_creator", "create_individual", "SessionGene"]


def create_individual(gene_list: list[SessionGene]) -> Individual:
    """Lazy import wrapper to avoid circular dependency during package init."""

    from src.ga.individual import create_individual as _create_individual

    return _create_individual(gene_list)
