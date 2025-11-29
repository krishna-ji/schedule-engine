"""
GA module: Genetic Algorithm components for schedule optimization.

Exposes:
    - get_creator: Centralized DEAP creator registry
    - create_individual: Factory function for creating individuals
    - SessionGene: Gene representation for course sessions
"""

from __future__ import annotations

from src.ga.creator_registry import get_creator
from src.ga.individual import create_individual
from src.ga.sessiongene import SessionGene

__all__ = ["get_creator", "create_individual", "SessionGene"]
