"""
Constraint-Aware Mutation and Crossover Operators

These operators actively REFUSE to create group overlap violations, unlike
the standard operators that create conflicts and rely on repair.

Key Feature:
- Before applying mutation/crossover, simulate the change
- If the change would create a group family conflict, reject it
- Try alternative mutations/swaps until a valid one is found

This is a "prevention over cure" approach for the most expensive constraint.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.domain.gene import SessionGene
from src.ga.core.population import build_group_family_map, get_family_map_from_json

if TYPE_CHECKING:
    from src.domain.types import Individual, SchedulingContext


def _get_or_build_family_map(context: SchedulingContext) -> dict[str, set[str]]:
    """Get cached family map or build one using JSON-based hierarchy."""
    if hasattr(context, "_group_family_map") and context._group_family_map:
        return context._group_family_map  # type: ignore

    # Use JSON-based hierarchy (correct behavior)
    try:
        family_map = get_family_map_from_json("data/Groups.json")
    except FileNotFoundError:
        # Fallback: build from hierarchy if JSON not found
        from src.ga.core.population import analyze_group_hierarchy

        hierarchy = analyze_group_hierarchy(context.groups)
        family_map = build_group_family_map(hierarchy)

    context._group_family_map = family_map  # type: ignore

    return family_map


def _build_time_occupation_map(
    individual: list[SessionGene],
    family_map: dict[str, set[str]],
    exclude_gene: SessionGene | None = None,
) -> dict[int, set[str]]:
    """
    Build a map of quantum -> set of occupied group families.

    Used for fast conflict checking during mutation.
    """
    occupation: dict[int, set[str]] = {}

    for gene in individual:
        if exclude_gene and gene is exclude_gene:
            continue

        for q in range(gene.start_quanta, gene.end_quanta):
            if q not in occupation:
                occupation[q] = set()

            # Add all groups and their families
            for gid in gene.group_ids:
                occupation[q].update(family_map.get(gid, {gid}))

    return occupation


def _would_create_group_conflict(
    gene: SessionGene,
    new_start: int,
    individual: list[SessionGene],
    family_map: dict[str, set[str]],
) -> bool:
    """
    Check if moving gene to new_start would create a group family conflict.
    """
    occupation = _build_time_occupation_map(individual, family_map, exclude_gene=gene)

    # Get family of groups in this gene
    gene_family: set[str] = set()
    for gid in gene.group_ids:
        gene_family.update(family_map.get(gid, {gid}))

    # Check each quantum in proposed new position
    for q in range(new_start, new_start + gene.num_quanta):
        occupied_at_q = occupation.get(q, set())
        if gene_family & occupied_at_q:  # Intersection = conflict
            return True

    return False


def _find_conflict_free_time(
    gene: SessionGene,
    individual: list[SessionGene],
    family_map: dict[str, set[str]],
    available_quanta: list[int],
    max_attempts: int = 20,
) -> int | None:
    """
    Find a time slot that doesn't create group conflicts.

    Uses random sampling rather than exhaustive search for efficiency.
    """
    occupation = _build_time_occupation_map(individual, family_map, exclude_gene=gene)

    gene_family: set[str] = set()
    for gid in gene.group_ids:
        gene_family.update(family_map.get(gid, {gid}))

    duration = gene.num_quanta
    max_start = len(available_quanta) - duration

    if max_start < 0:
        return None

    # Try random positions
    for _ in range(max_attempts):
        start_idx = random.randint(0, max_start)
        start_q = available_quanta[start_idx]

        # Check no conflicts in proposed range
        conflict = False
        for q in range(start_q, start_q + duration):
            occupied_at_q = occupation.get(q, set())
            if gene_family & occupied_at_q:
                conflict = True
                break

        if not conflict:
            return start_q

    return None


def mutate_gene_constraint_aware(
    gene: SessionGene,
    individual: list[SessionGene],
    context: SchedulingContext,
    max_attempts: int = 5,
) -> SessionGene:
    """
    Mutate a gene while actively avoiding group overlap creation.

    Only changes time slot if it doesn't create new conflicts.
    Falls back to original time if no valid mutation found.
    """
    family_map = _get_or_build_family_map(context)

    # Get course info
    course_key = (gene.course_id, gene.course_type)
    context.courses.get(course_key)
    # INSTRUCTOR: Mutate with qualification check
    qualified_instructors = [
        inst_id
        for inst_id, inst in context.instructors.items()
        if course_key in getattr(inst, "qualified_courses", [])
    ]

    if gene.instructor_id in qualified_instructors and random.random() < 0.7:
        new_instructor = gene.instructor_id
    elif qualified_instructors:
        new_instructor = random.choice(qualified_instructors)
    else:
        # STRICT: never assign unqualified — keep current
        new_instructor = gene.instructor_id
    # ROOM: Mutate with suitability check
    from src.ga.operators.mutation import find_suitable_rooms_for_course

    primary_group = gene.group_ids[0] if gene.group_ids else ""
    suitable_rooms = find_suitable_rooms_for_course(
        gene.course_id, gene.course_type, primary_group, context
    )

    if gene.room_id in suitable_rooms and random.random() < 0.5:
        new_room = gene.room_id
    elif suitable_rooms:
        new_room = random.choice(suitable_rooms)
    else:
        # STRICT: never assign unsuitable room — keep current
        new_room = gene.room_id
    # TIME: Mutate ONLY if it doesn't create conflicts
    new_start = gene.start_quanta  # Default: keep current time

    if random.random() < 0.7:  # 70% chance to attempt time mutation
        conflict_free_start = _find_conflict_free_time(
            gene, individual, family_map, list(context.available_quanta)
        )
        if conflict_free_start is not None:
            new_start = conflict_free_start
        # If no conflict-free slot found, keep original time

    return SessionGene(
        course_id=gene.course_id,
        course_type=gene.course_type,
        instructor_id=new_instructor,
        group_ids=gene.group_ids,  # NEVER mutated
        room_id=new_room,
        start_quanta=new_start,
        num_quanta=gene.num_quanta,  # NEVER mutated
        co_instructor_ids=_update_co_instructors(gene, new_instructor, context),
    )


def mutate_individual_constraint_aware(
    individual: Individual,
    context: SchedulingContext,
    mut_prob: float = 0.2,
) -> tuple[Individual]:
    """
    Apply constraint-aware mutation to individual.

    Unlike standard mutation, this actively refuses to create group overlaps.
    """
    for i in range(len(individual)):
        if random.random() < mut_prob:
            individual[i] = mutate_gene_constraint_aware(
                individual[i], list(individual), context
            )

    return (individual,)


def _update_co_instructors(
    gene: SessionGene,
    new_main_instructor: str,
    context: SchedulingContext,
) -> list[str]:
    """Return valid co-instructor list when main instructor may change.

    Structural invariant: practical genes always have exactly 1 co-instructor.
    """
    if gene.course_type != "practical":
        return []
    current_co = getattr(gene, "co_instructor_ids", [])
    if current_co and current_co[0] != new_main_instructor:
        return list(current_co)
    course_key = (gene.course_id, gene.course_type)
    candidates = [
        iid
        for iid, inst in context.instructors.items()
        if course_key in getattr(inst, "qualified_courses", [])
        and iid != new_main_instructor
    ]
    if candidates:
        return [random.choice(candidates)]
    others = [iid for iid in context.instructors if iid != new_main_instructor]
    return [random.choice(others)] if others else list(current_co)

    return (individual,)


def crossover_constraint_aware(
    ind1: list[SessionGene],
    ind2: list[SessionGene],
    context: SchedulingContext,
    cx_prob: float = 0.5,
    validate: bool = True,
) -> tuple[list[SessionGene], list[SessionGene]]:
    """
    Constraint-aware crossover that refuses to create group overlaps.

    Before swapping time attributes, checks if the swap would create
    conflicts. Only proceeds if the swap is "safe".
    """
    family_map = _get_or_build_family_map(context)

    # Build lookup tables
    gene_map1 = {(gene.course_id, tuple(sorted(gene.group_ids))): gene for gene in ind1}
    gene_map2 = {(gene.course_id, tuple(sorted(gene.group_ids))): gene for gene in ind2}

    # Verify structure
    if validate:
        keys1 = set(gene_map1.keys())
        keys2 = set(gene_map2.keys())
        if keys1 != keys2:
            raise ValueError(
                "Crossover: Individuals have mismatched (course, group) pairs!"
            )

    keys_to_process = (
        gene_map1.keys()
        if validate
        else (set(gene_map1.keys()) & set(gene_map2.keys()))
    )

    swaps_accepted = 0
    swaps_rejected = 0

    for key in keys_to_process:
        if random.random() >= cx_prob:
            continue

        gene1 = gene_map1[key]
        gene2 = gene_map2[key]

        # Simulate swap for ind1
        # Would putting gene2's time in gene1's individual create conflicts?
        would_conflict_in_1 = _would_create_group_conflict(
            gene1, gene2.start_quanta, ind1, family_map
        )

        # Simulate swap for ind2
        would_conflict_in_2 = _would_create_group_conflict(
            gene2, gene1.start_quanta, ind2, family_map
        )

        if would_conflict_in_1 or would_conflict_in_2:
            # Reject this swap - would create conflicts
            swaps_rejected += 1
            continue

        # STRICT: reject swap if it would violate qualification or suitability
        from src.ga.operators.crossover import _is_swap_valid

        if not _is_swap_valid(gene1, gene2, context):
            swaps_rejected += 1
            continue

        # Safe to swap!
        swaps_accepted += 1

        # Swap instructor
        gene1.instructor_id, gene2.instructor_id = (
            gene2.instructor_id,
            gene1.instructor_id,
        )

        # Swap room
        gene1.room_id, gene2.room_id = gene2.room_id, gene1.room_id

        # Swap co-instructors
        gene1.co_instructor_ids, gene2.co_instructor_ids = (
            gene2.co_instructor_ids,
            gene1.co_instructor_ids,
        )

        # Swap time (start only - duration is fixed)
        gene1.start_quanta, gene2.start_quanta = gene2.start_quanta, gene1.start_quanta

    # Validate time bounds
    from src.io.time_system import QuantumTimeSystem

    time_system = QuantumTimeSystem()
    max_valid_quantum = time_system.total_quanta

    for gene in ind1 + ind2:
        if (
            gene.num_quanta > 0
            and (gene.start_quanta + gene.num_quanta - 1) >= max_valid_quantum
        ):
            max_allowed_start = max(0, max_valid_quantum - gene.num_quanta)
            gene.start_quanta = min(gene.start_quanta, max_allowed_start)

    return ind1, ind2


# INTEGRATION HELPERS
def get_constraint_aware_mutation(context: SchedulingContext):
    """
    Factory function to create a constraint-aware mutation operator.
    """

    def mutation_wrapper(
        individual: Individual, mut_prob: float = 0.2
    ) -> tuple[Individual]:
        return mutate_individual_constraint_aware(individual, context, mut_prob)

    return mutation_wrapper


def get_constraint_aware_crossover(context: SchedulingContext):
    """
    Factory function to create a constraint-aware crossover operator.
    """

    def crossover_wrapper(
        ind1: list[SessionGene], ind2: list[SessionGene], cx_prob: float = 0.5
    ) -> tuple[list[SessionGene], list[SessionGene]]:
        return crossover_constraint_aware(ind1, ind2, context, cx_prob)

    return crossover_wrapper
