import random

from src.config import get_config
from src.ga.sessiongene import SessionGene


def crossover_course_group_aware(
    ind1: list[SessionGene], ind2: list[SessionGene], cx_prob: float = 0.5
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
    # Can be disabled via config for performance or experimental operators
    if get_config().ga.validate_population_integrity:
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
                f"   This indicates population corruption or invalid mutation.\n"
                f"   To disable this check, set get_config().ga.validate_population_integrity=False in config/ga_params.py"
            )

    # For each (course, group) pair, probabilistically swap ATTRIBUTES
    # If validation is disabled, only swap for common keys (intersection)
    keys_to_process = (
        gene_map1.keys()
        if get_config().ga.validate_population_integrity
        else (set(gene_map1.keys()) & set(gene_map2.keys()))
    )

    for key in keys_to_process:
        if random.random() < cx_prob:
            gene1 = gene_map1[key]
            gene2 = gene_map2[key]

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
            # DO NOT swap num_quanta - it's fixed by course.quanta_per_week (L+T or P)

    # Validate start_quanta don't exceed valid range after swap
    # If invalid, clip start_quanta only (num_quanta is FIXED by course requirements)
    from src.encoder.quantum_time_system import QuantumTimeSystem

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
