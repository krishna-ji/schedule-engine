"""Memetic/local search helpers for experiment notebooks.

Provides local search operators for Mode B and beyond.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

from src.ga.sessiongene import SessionGene

if TYPE_CHECKING:
    from src.notebooks.data_loader import ScheduleData


def local_search_individual(
    individual: list[SessionGene],
    data: ScheduleData,
    evaluate_fn: Callable[[list[SessionGene]], tuple[int, int]],
    max_iterations: int = 10,
    improvement_threshold: int = 0,
) -> tuple[list[SessionGene], int]:
    """Apply local search to improve an individual.

    Simple greedy local search that tries to improve gene assignments
    by random reassignment and accepting improvements.

    Args:
        individual: Individual to improve
        data: Schedule data
        evaluate_fn: Fitness evaluation function
        max_iterations: Maximum iterations
        improvement_threshold: Minimum improvement to accept

    Returns:
        Tuple of (improved_individual, total_improvement)
    """
    from src.notebooks.population import _room_suitable

    current_fitness = evaluate_fn(individual)
    current_hard = current_fitness[0]
    total_improvement = 0

    for _ in range(max_iterations):
        # Pick random gene to optimize
        idx = random.randint(0, len(individual) - 1)
        gene = individual[idx]

        # Try alternative assignments
        best_new_gene = None
        best_hard = current_hard

        course = data.courses.get((gene.course_id, gene.course_type))
        if not course:
            continue

        # Try 5 random alternatives
        for _ in range(5):
            # Random time
            new_start = random.randint(
                0, max(0, data.qts.total_quanta - gene.num_quanta)
            )

            # Random room (suitable)
            suitable_rooms = [
                r
                for r in data.rooms
                if _room_suitable(data.rooms[r], course.required_room_features)
            ]
            if not suitable_rooms:
                suitable_rooms = list(data.rooms.keys())
            new_room = random.choice(suitable_rooms)

            # Create modified gene
            modified = SessionGene(
                course_id=gene.course_id,
                course_type=gene.course_type,
                instructor_id=gene.instructor_id,
                group_ids=gene.group_ids,
                room_id=new_room,
                start_quanta=new_start,
                num_quanta=gene.num_quanta,
            )

            # Test improvement
            individual[idx] = modified
            new_fitness = evaluate_fn(individual)
            new_hard = new_fitness[0]

            if new_hard < best_hard:
                best_hard = new_hard
                best_new_gene = modified

            # Restore original for next iteration
            individual[idx] = gene

        # Accept best improvement found
        if best_new_gene and best_hard < current_hard - improvement_threshold:
            individual[idx] = best_new_gene
            improvement = current_hard - best_hard
            total_improvement += improvement
            current_hard = best_hard

    return individual, total_improvement


def memetic_generation_callback(
    local_search_prob: float = 0.1,
    local_search_iterations: int = 10,
) -> Callable[..., None]:
    """Create generation callback for memetic search.

    Returns callback function that applies local search to some individuals
    after each generation.

    Args:
        local_search_prob: Probability of applying local search to each individual
        local_search_iterations: Iterations per local search

    Returns:
        Callback function for run_nsga2
    """

    def callback(
        gen: int,
        population: list,
        stats: object,
        data: "ScheduleData",
        evaluate_fn: Callable[[list[SessionGene]], tuple[int, int]],
    ) -> None:
        """Apply local search to population members."""
        for ind in population:
            if random.random() < local_search_prob:
                genes = list(ind)  # Get genes from DEAP Individual
                improved_genes, _ = local_search_individual(
                    genes, data, evaluate_fn, local_search_iterations
                )
                ind[:] = improved_genes
                # Invalidate fitness after modification
                del ind.fitness.values

    return callback
