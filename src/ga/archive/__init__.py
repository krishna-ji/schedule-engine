"""
Archive-based diversity maintenance.

ENHANCEMENT #5: Novelty search and MAP-Elites for behavioral diversity.
"""

from src.ga.archive.behavioral_descriptors import BehavioralDescriptors
from src.ga.archive.novelty_archive import NoveltyArchive
from src.ga.archive.map_elites import MAPElites

__all__ = [
    "BehavioralDescriptors",
    "NoveltyArchive",
    "MAPElites",
]
