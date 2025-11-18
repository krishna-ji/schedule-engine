"""
Diversity package for novelty search and behavioral characterization.

ENHANCEMENT #6: Archive-based diversity for quality-diversity optimization.
"""

from src.diversity.archive import BehavioralArchive
from src.diversity.behavioral_features import extract_behavioral_features
from src.diversity.novelty_metric import compute_novelty, k_nearest_neighbors

__all__ = [
    "BehavioralArchive",
    "extract_behavioral_features",
    "compute_novelty",
    "k_nearest_neighbors",
]
