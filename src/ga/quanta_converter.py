"""
Helper function for SessionGene migration.

Converts quanta list to start_quanta + num_quanta format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.gene import SessionGene


def quanta_list_to_contiguous(quanta_list: list[int]) -> tuple[int, int]:
    """
    Convert a quanta list to start_quanta and num_quanta.

    For continuity enforcement, we assume the list represents a contiguous block.
    If not contiguous, we take the min as start and length as duration.

    Args:
        quanta_list: List of quantum indices (e.g., [10, 11, 12])

    Returns:
        (start_quanta, num_quanta) tuple

    Examples:
        [10, 11, 12] → (10, 3)
        [5] → (5, 1)
        [] → (0, 1)  # Fallback
        [10, 12, 14] → (10, 3)  # Non-contiguous: uses min and length
    """
    if not quanta_list:
        return (0, 1)  # Fallback: first quantum, 1 hour

    sorted_quanta = sorted(quanta_list)
    start_quanta = sorted_quanta[0]
    num_quanta = len(sorted_quanta)

    return (start_quanta, num_quanta)


def assign_quanta_to_gene(gene: SessionGene, quanta_list: list[int]) -> None:
    """
    Assign quanta list to SessionGene using new API.

    Converts list to contiguous representation and sets start_quanta + num_quanta.

    Args:
        gene: SessionGene instance to update
        quanta_list: List of quantum indices
    """
    start_q, num_q = quanta_list_to_contiguous(quanta_list)
    gene.start_quanta = start_q
    gene.num_quanta = num_q
