import random
from typing import TYPE_CHECKING

from src.domain.gene import SessionGene

if TYPE_CHECKING:
    from src.domain.types import SchedulingContext


def _is_swap_valid(
    gene1: SessionGene,
    gene2: SessionGene,
    context: "SchedulingContext",
) -> bool:
    """Check if swapping instructor/room between gene1 and gene2 would
    violate instructor qualification or room suitability constraints.

    Both genes must be for the same course offering (same course_id, group_ids).
    """
    from src.utils.room_compatibility import is_room_suitable_for_course

    course_key1 = (gene1.course_id, gene1.course_type)
    course_key2 = (gene2.course_id, gene2.course_type)

    # --- Instructor qualification check ---
    # After swap: gene1 gets gene2's instructor, gene2 gets gene1's instructor
    inst1 = context.instructors.get(gene2.instructor_id)
    inst2 = context.instructors.get(gene1.instructor_id)

    if inst1 and course_key1 not in getattr(inst1, "qualified_courses", []):
        return False
    if inst2 and course_key2 not in getattr(inst2, "qualified_courses", []):
        return False

    # --- Room suitability check ---
    for gene, incoming_room_id in [(gene1, gene2.room_id), (gene2, gene1.room_id)]:
        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        room = context.rooms.get(incoming_room_id)
        if not course or not room:
            continue

        required = getattr(course, "required_room_features", "lecture")
        req_str = (
            (required if isinstance(required, str) else str(required)).lower().strip()
        )
        room_type = getattr(room, "room_features", "lecture")
        room_str = (
            (room_type if isinstance(room_type, str) else str(room_type))
            .lower()
            .strip()
        )
        lab_feats = getattr(course, "specific_lab_features", None)
        room_spec = getattr(room, "specific_features", None)

        if not is_room_suitable_for_course(req_str, room_str, lab_feats, room_spec):
            return False

    return True


def crossover_course_group_aware(
    ind1: list[SessionGene],
    ind2: list[SessionGene],
    cx_prob: float = 0.5,
    validate: bool = True,
    context: "SchedulingContext | None" = None,
) -> tuple[list[SessionGene], list[SessionGene]]:
    """
    Position-Independent Crossover that preserves (course, group) structure.

    Instead of swapping entire genes by index, this operator matches genes by their
    (course_id, group_ids) identity and swaps ONLY mutable attributes (instructor,
    room, time slots). This ensures the fundamental (course, group) enrollment
    structure is never corrupted, even if gene positions differ between individuals.

    CRITICAL: This is the recommended crossover for timetabling problems where
    chromosome structure represents fixed course-group enrollments. It enables
    future features like gene sorting, compaction, and clustering without risk
    of creating duplicate or missing (course, group) pairs.

    Args:
        ind1, ind2 (List[SessionGene]): Two individuals to perform crossover on.
        cx_prob (float): Probability of swapping attributes for each gene pair.
                        Default 0.5 means each gene has 50% chance of exchange.

    Returns:
        tuple: (ind1, ind2) with swapped attributes (not swapped genes)

    Raises:
        ValueError: If validation is enabled and individuals have mismatched
                   (course, group) pairs, indicating structural corruption.

    Example:
        Parent 1: Gene(MATH101, GroupA, Instructor=I1, Room=R1, Time=[10,11,12])
        Parent 2: Gene(MATH101, GroupA, Instructor=I2, Room=R2, Time=[20,21,22])

        After crossover (50% prob):
        Child 1:  Gene(MATH101, GroupA, Instructor=I2, Room=R2, Time=[20,21,22])
        Child 2:  Gene(MATH101, GroupA, Instructor=I1, Room=R1, Time=[10,11,12])

        Note: MATH101-GroupA still exists in both (no duplication/loss)
    """
    # Build lookup tables: (course_id, tuple(sorted(group_ids))) -> gene
    # We sort group_ids to ensure consistent key regardless of list order
    gene_map1 = {(gene.course_id, tuple(sorted(gene.group_ids))): gene for gene in ind1}
    gene_map2 = {(gene.course_id, tuple(sorted(gene.group_ids))): gene for gene in ind2}

    # Verify both individuals have same (course, group) pairs
    # This catches any corruption early with a clear error message
    if validate:
        keys1 = set(gene_map1.keys())
        keys2 = set(gene_map2.keys())

        if keys1 != keys2:
            missing_in_ind1 = keys2 - keys1
            missing_in_ind2 = keys1 - keys2
            raise ValueError(
                f"[X] CROSSOVER ERROR: Individuals have mismatched (course, group) pairs!\n"
                f"   Individual 1 has {len(keys1)} pairs, Individual 2 has {len(keys2)} pairs.\n"
                f"   Missing in Individual 1: {missing_in_ind1}\n"
                f"   Missing in Individual 2: {missing_in_ind2}\n"
                f"   This indicates population corruption or invalid mutation."
            )

    # For each (course, group) pair, probabilistically swap ATTRIBUTES
    # If validation is disabled, only swap for common keys (intersection)
    keys_to_process = (
        gene_map1.keys()
        if validate
        else (set(gene_map1.keys()) & set(gene_map2.keys()))
    )

    for key in keys_to_process:
        if random.random() < cx_prob:
            gene1 = gene_map1[key]
            gene2 = gene_map2[key]

            # STRICT: reject swap if it would violate qualification or suitability
            if context is not None and not _is_swap_valid(gene1, gene2, context):
                continue

            # Swap ONLY mutable attributes (NOT course_id or group_ids)
            # This preserves the fundamental chromosome structure
            gene1.instructor_id, gene2.instructor_id = (
                gene2.instructor_id,
                gene1.instructor_id,
            )
            gene1.room_id, gene2.room_id = gene2.room_id, gene1.room_id
            # Swap time allocation (start ONLY - duration is fixed by course requirements)
            gene1.start_quanta, gene2.start_quanta = (
                gene2.start_quanta,
                gene1.start_quanta,
            )
            # Swap co-instructors (practical sessions)
            gene1.co_instructor_ids, gene2.co_instructor_ids = (
                gene2.co_instructor_ids,
                gene1.co_instructor_ids,
            )
            # DO NOT swap num_quanta - it's fixed by course.quanta_per_week (L+T or P)

    # Validate start_quanta don't exceed valid range after swap
    # If invalid, clip start_quanta only (num_quanta is FIXED by course requirements)
    from src.io.time_system import QuantumTimeSystem

    time_system = QuantumTimeSystem()
    max_valid_quantum = time_system.total_quanta

    for gene in ind1:
        if (
            gene.num_quanta > 0
            and (gene.start_quanta + gene.num_quanta - 1) >= max_valid_quantum
        ):
            # Start quantum would make session extend beyond valid range
            # Adjust start_quanta to fit (num_quanta stays FIXED)
            max_allowed_start = max(0, max_valid_quantum - gene.num_quanta)
            gene.start_quanta = min(gene.start_quanta, max_allowed_start)

    for gene in ind2:
        if (
            gene.num_quanta > 0
            and (gene.start_quanta + gene.num_quanta - 1) >= max_valid_quantum
        ):
            # Start quantum would make session extend beyond valid range
            # Adjust start_quanta to fit (num_quanta stays FIXED)
            max_allowed_start = max(0, max_valid_quantum - gene.num_quanta)
            gene.start_quanta = min(gene.start_quanta, max_allowed_start)

    return ind1, ind2
