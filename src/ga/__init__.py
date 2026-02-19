"""GA module: Genetic Algorithm components for schedule optimization.

Exposes:
    - SessionGene: Gene representation for course sessions
    - RepairPipeline: Unified repair operations interface
    - PopulationFactory: Single entry point for population creation
"""

from __future__ import annotations

from src.domain.gene import SessionGene
from src.ga.core.population_factory import PopulationFactory
from src.ga.repair.pipeline import RepairPipeline

__all__ = [
    "PopulationFactory",
    "RepairPipeline",
    "SessionGene",
]
