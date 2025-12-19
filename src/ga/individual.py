from typing import TYPE_CHECKING, cast

from src.core.types import Individual
from src.ga.creator_registry import get_creator
from src.ga.sessiongene import SessionGene

# Get centralized creator instance
creator = get_creator()

if TYPE_CHECKING:
    pass


def create_individual(gene_list: list[SessionGene]) -> Individual:
    """
    Wraps a list of SessionGene Objects into a DEAP Individual.

    Args:
        gene_list (List[SessionGene]): List of SessionGene objects representing the individual's genes.

    Returns:
        creator.Individual: A new individual initialized with the provided genes.
    """
    return cast(Individual, creator.Individual(gene_list))
