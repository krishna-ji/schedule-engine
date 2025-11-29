"""
Centralized DEAP creator registry for GA types.

This module initializes all DEAP creator types used throughout the GA pipeline.
Centralizing creator logic prevents duplicate registrations and ensures consistent
fitness function definitions across the codebase.

Usage:
    Import this module before using any creator types:
    >>> from src.ga.creator_registry import get_creator
    >>> creator = get_creator()
    >>> individual = creator.Individual([gene1, gene2, ...])
"""

from deap import base, creator


def _initialize_creator():
    """
    Initialize DEAP creator types if not already registered.

    Creates:
        - FitnessMulti: Multi-objective fitness with weights (-1.0, -1.0)
                        for hard and soft constraint minimization (both minimized)
        - Individual: List-based individual with FitnessMulti fitness

    Weights rationale:
        - Both objectives are minimized. Relative priority should be set via
          constraint weights in YAML and the soft_weight_factor in config,
          not by changing FitnessMulti magnitudes under NSGA-II.
    """
    # Only create if not already registered (prevents DEAP re-registration errors)
    if not hasattr(creator, "FitnessMulti"):
        # Minimize both objectives (hard, soft). Magnitude shouldn't be used to
        # prioritize objectives under NSGA-II; use config weights instead.
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)


def get_creator():
    """
    Get the initialized DEAP creator instance.

    Returns:
        creator: DEAP creator with registered types (FitnessMulti, Individual)

    Example:
        >>> from src.ga.creator_registry import get_creator
        >>> creator = get_creator()
        >>> ind = creator.Individual()
    """
    _initialize_creator()
    return creator


# Auto-initialize on module import for convenience
_initialize_creator()
