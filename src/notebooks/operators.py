"""GA operators for experiment notebooks.

Provides crossover and mutation operators.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import TYPE_CHECKING

from src.ga.sessiongene import SessionGene

if TYPE_CHECKING:
    from src.notebooks.data_loader import ScheduleData


def course_aware_crossover(
    ind1: list[SessionGene],
    ind2: list[SessionGene],
    cx_prob: float = 0.5,
) -> tuple[list[SessionGene], list[SessionGene]]:
    """Course-aware crossover preserving (course, group) structure.

    Swaps entire courses between parents rather than individual genes.
    This maintains course integrity while exploring the solution space.

    Args:
        ind1, ind2: Two individuals to crossover
        cx_prob: Probability of swapping each course (default: 0.5)

    Returns:
        Tuple of two modified individuals
    """
    # Group genes by course
    genes1: dict[tuple[str, str], list[SessionGene]] = defaultdict(list)
    genes2: dict[tuple[str, str], list[SessionGene]] = defaultdict(list)

    for g in ind1:
        genes1[(g.course_id, g.course_type)].append(g)
    for g in ind2:
        genes2[(g.course_id, g.course_type)].append(g)

    new1: list[SessionGene] = []
    new2: list[SessionGene] = []

    # For each course, swap with probability
    all_courses = set(genes1.keys()) | set(genes2.keys())
    for course_key in all_courses:
        if random.random() < cx_prob:
            # Swap: ind1 gets ind2's genes, ind2 gets ind1's
            new1.extend(copy.deepcopy(genes2.get(course_key, [])))
            new2.extend(copy.deepcopy(genes1.get(course_key, [])))
        else:
            # Keep: each keeps their own
            new1.extend(copy.deepcopy(genes1.get(course_key, [])))
            new2.extend(copy.deepcopy(genes2.get(course_key, [])))

    ind1[:] = new1
    ind2[:] = new2
    return ind1, ind2


def smart_mutation(
    individual: list[SessionGene],
    data: ScheduleData,
    time_prob: float = 0.2,
    room_prob: float = 0.1,
    instructor_prob: float = 0.1,
) -> tuple[list[SessionGene]]:
    """Smart mutation respecting course constraints.

    Mutates time slots, rooms, and instructors while respecting:
    - Room suitability (labs for practicals)
    - Instructor qualifications

    Args:
        individual: Individual to mutate (modified in place)
        data: Schedule data for constraints
        time_prob: Probability of mutating time slot
        room_prob: Probability of mutating room
        instructor_prob: Probability of mutating instructor

    Returns:
        Tuple containing the mutated individual
    """
    from src.notebooks.population import _room_suitable

    for gene in individual:
        # Time mutation
        if random.random() < time_prob:
            gene.start_quanta = random.randint(
                0, max(0, data.qts.total_quanta - gene.num_quanta)
            )

        # Room mutation
        if random.random() < room_prob:
            course = data.courses.get((gene.course_id, gene.course_type))
            if course:
                suitable = [
                    r
                    for r in data.rooms
                    if _room_suitable(data.rooms[r], course.required_room_features)
                ]
                if suitable:
                    gene.room_id = random.choice(suitable)

        # Instructor mutation
        if random.random() < instructor_prob:
            course = data.courses.get((gene.course_id, gene.course_type))
            if course and course.qualified_instructor_ids:
                gene.instructor_id = random.choice(course.qualified_instructor_ids)

    return (individual,)


def uniform_mutation(
    individual: list[SessionGene],
    data: ScheduleData,
    gene_prob: float = 0.1,
) -> tuple[list[SessionGene]]:
    """Uniform mutation - each gene has probability of full random reassignment.

    Args:
        individual: Individual to mutate
        data: Schedule data
        gene_prob: Probability of mutating each gene

    Returns:
        Tuple containing the mutated individual
    """
    from src.notebooks.population import _room_suitable

    for gene in individual:
        if random.random() < gene_prob:
            course = data.courses.get((gene.course_id, gene.course_type))
            if course:
                # Random instructor
                qualified = course.qualified_instructor_ids or list(
                    data.instructors.keys()
                )
                gene.instructor_id = random.choice(qualified)

                # Random room
                suitable = [
                    r
                    for r in data.rooms
                    if _room_suitable(data.rooms[r], course.required_room_features)
                ]
                if suitable:
                    gene.room_id = random.choice(suitable)
                else:
                    gene.room_id = random.choice(list(data.rooms.keys()))

                # Random time
                gene.start_quanta = random.randint(
                    0, max(0, data.qts.total_quanta - gene.num_quanta)
                )

    return (individual,)
