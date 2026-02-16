from typing import cast

from src.domain.gene import SessionGene
from src.domain.types import Individual
from src.ga.core.creator_registry import get_creator

# Get centralized creator instance
creator = get_creator()


def create_individual(gene_list: list[SessionGene]) -> Individual:
    """
    Wraps a list of SessionGene Objects into a DEAP Individual.

    Args:
        gene_list (List[SessionGene]): List of SessionGene objects representing the individual's genes.

    Returns:
        creator.Individual: A new individual initialized with the provided genes.
    """
    return cast(Individual, creator.Individual(gene_list))
