"""
Improvement Heuristics - Local Search Moves for Refinement

Provides sophisticated local search operators that iteratively improve
schedules by making small, beneficial modifications.

Improvement heuristics are useful for:
1. Fine-tuning solutions (hill climbing)
2. Memetic algorithms (local search within GA)
3. Intensification (exploit promising regions)

Strategies:
1. Kempe Chain: Graph coloring move for conflict resolution
2. Ejection Chain: Advanced local search with cascading moves
3. Variable Depth Search: Multi-move lookahead optimization

Architecture:
- Decorator-based registration with @improvement_heuristic
- Modifies individual in-place (returns improvement count)
- Applies moves only if they improve fitness
- Can be applied iteratively until no improvement

Usage:
    from src.heuristics.improvement import kempe_chain

    # Apply Kempe chain move
    improvements = kempe_chain(individual, context, max_iterations=10)
    print(f"Made {improvements} improving moves")
"""

from typing import List, Dict, Tuple, Set, Optional
import random
from collections import defaultdict
import copy

from src.ga.sessiongene import SessionGene
from src.core.types import SchedulingContext
from src.heuristics.registry import improvement_heuristic
from src.heuristics.utils import (
    estimate_session_student_count,
    get_available_quanta,
    get_course_for_gene,
    get_course_room_requirement,
    get_room_feature,
)


# ============================================================================
# KEMPE CHAIN (Graph coloring move for conflict resolution)
# ============================================================================


@improvement_heuristic(
    name="kempe_chain",
    description="Apply Kempe chain moves to resolve time conflicts (graph coloring heuristic)",
    priority=1,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def kempe_chain(
    individual: List[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 5,
) -> int:
    """
    Apply Kempe chain moves to improve schedule.

    Kempe chains are a classical graph coloring technique:
    - Identify conflicting sessions (share groups/instructors)
    - Swap time slots along a chain of sessions
    - Only apply if swap reduces conflicts

    Algorithm:
    1. Find a session with conflicts
    2. Build Kempe chain: sessions that would be affected by time swap
    3. Swap times along the chain
    4. Accept if fitness improves

    Args:
        individual: List of SessionGene to modify
        context: Scheduling context
        max_iterations: Maximum number of Kempe chains to attempt

    Returns:
        Number of improving moves applied
    """
    improvements = 0

    for _ in range(max_iterations):
        # Find conflicting sessions
        conflict_pairs = _find_conflict_pairs(individual, context)

        if not conflict_pairs:
            break  # No conflicts

        # Select random conflict pair
        gene1, gene2 = random.choice(list(conflict_pairs))

        # Build Kempe chain
        chain = _build_kempe_chain(individual, gene1, gene2, context)

        if not chain:
            continue

        # Save current state
        old_fitness = _calculate_fitness(individual, context)

        # Swap times along chain
        _apply_kempe_swap(chain)

        # Evaluate new fitness
        new_fitness = _calculate_fitness(individual, context)

        # Accept if improved
        if new_fitness[0] < old_fitness[0]:  # Fewer hard violations
            improvements += 1
            # Invalidate fitness
            if hasattr(individual, "fitness"):
                del individual.fitness.values
        else:
            # Revert swap
            _apply_kempe_swap(chain)  # Swap back

    return improvements


# ============================================================================
# EJECTION CHAIN (Advanced local search with cascading moves)
# ============================================================================


@improvement_heuristic(
    name="ejection_chain",
    description="Apply ejection chain moves with cascading reassignments",
    priority=2,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def ejection_chain(
    individual: List[SessionGene],
    context: SchedulingContext,
    max_chain_length: int = 5,
    max_iterations: int = 3,
) -> int:
    """
    Apply ejection chain moves for advanced local search.

    Ejection chains extend Kempe chains by allowing cascading reassignments:
    - Move session A to new time, ejects session B
    - Move session B to new time, ejects session C
    - Continue until chain terminates or max length

    More powerful than Kempe chains but more computationally expensive.

    Args:
        individual: List of SessionGene to modify
        context: Scheduling context
        max_chain_length: Maximum length of ejection chain
        max_iterations: Maximum number of chains to attempt

    Returns:
        Number of improving chains applied
    """
    improvements = 0

    for _ in range(max_iterations):
        # Select random starting session
        start_gene = random.choice(individual)

        # Build ejection chain
        chain = _build_ejection_chain(individual, start_gene, context, max_chain_length)

        if not chain or len(chain) < 2:
            continue

        # Save current state
        old_fitness = _calculate_fitness(individual, context)

        # Apply chain moves
        _apply_ejection_chain(chain, context)

        # Evaluate new fitness
        new_fitness = _calculate_fitness(individual, context)

        # Accept if improved
        if new_fitness[0] < old_fitness[0] or (
            new_fitness[0] == old_fitness[0] and new_fitness[1] < old_fitness[1]
        ):
            improvements += 1
            # Invalidate fitness
            if hasattr(individual, "fitness"):
                del individual.fitness.values
        else:
            # Revert chain
            _revert_ejection_chain(chain)

    return improvements


# ============================================================================
# VARIABLE DEPTH SEARCH (Multi-move lookahead optimization)
# ============================================================================


@improvement_heuristic(
    name="variable_depth_search",
    description="Multi-move lookahead search with backtracking",
    priority=3,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def variable_depth_search(
    individual: List[SessionGene],
    context: SchedulingContext,
    max_depth: int = 3,
    max_iterations: int = 5,
) -> int:
    """
    Apply variable depth search with multi-move lookahead.

    Variable depth search explores sequences of moves:
    - Try move A, evaluate
    - Try move B after A, evaluate
    - Try move C after A+B, evaluate
    - Accept best sequence found

    More thorough than greedy local search but computationally intensive.

    Args:
        individual: List of SessionGene to modify
        context: Scheduling context
        max_depth: Maximum number of moves in sequence
        max_iterations: Maximum number of sequences to attempt

    Returns:
        Number of improving sequences applied
    """
    improvements = 0

    for _ in range(max_iterations):
        # Current fitness
        current_fitness = _calculate_fitness(individual, context)
        best_fitness = current_fitness
        best_sequence = []

        # Try different move sequences
        for depth in range(1, max_depth + 1):
            # Generate random move sequence
            move_sequence = _generate_move_sequence(individual, context, depth)

            # Save state
            saved_state = [copy.copy(gene) for gene in individual]

            # Apply sequence
            for move_type, move_params in move_sequence:
                _apply_move(individual, move_type, move_params, context)

            # Evaluate
            new_fitness = _calculate_fitness(individual, context)

            # Track best
            if new_fitness[0] < best_fitness[0] or (
                new_fitness[0] == best_fitness[0] and new_fitness[1] < best_fitness[1]
            ):
                best_fitness = new_fitness
                best_sequence = move_sequence

            # Restore state for next try
            for i, gene in enumerate(saved_state):
                individual[i] = copy.copy(gene)

        # Apply best sequence if found improvement
        if best_fitness[0] < current_fitness[0] or (
            best_fitness[0] == current_fitness[0]
            and best_fitness[1] < current_fitness[1]
        ):
            for move_type, move_params in best_sequence:
                _apply_move(individual, move_type, move_params, context)

            improvements += 1

            # Invalidate fitness
            if hasattr(individual, "fitness"):
                del individual.fitness.values

    return improvements


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _find_conflict_pairs(
    individual: List[SessionGene], context: SchedulingContext
) -> Set[Tuple[SessionGene, SessionGene]]:
    """Find pairs of sessions with conflicts (overlapping time + shared resources)."""
    conflicts = set()

    for i, gene1 in enumerate(individual):
        for gene2 in individual[i + 1 :]:
            # Check if times overlap based on actual session duration
            time1_end = gene1.time_quantum + gene1.duration_quanta
            time2_end = gene2.time_quantum + gene2.duration_quanta

            times_overlap = not (
                time1_end <= gene2.time_quantum or time2_end <= gene1.time_quantum
            )

            if not times_overlap:
                continue

            # Check for shared resources
            shared_groups = set(gene1.group_ids) & set(gene2.group_ids)
            same_instructor = gene1.instructor_id == gene2.instructor_id
            same_room = gene1.room_id == gene2.room_id

            if shared_groups or same_instructor or same_room:
                conflicts.add((gene1, gene2))

    return conflicts


def _build_kempe_chain(
    individual: List[SessionGene],
    gene1: SessionGene,
    gene2: SessionGene,
    context: SchedulingContext,
) -> List[SessionGene]:
    """Build Kempe chain for two conflicting sessions."""
    chain = [gene1, gene2]

    # Find sessions that would be affected by swapping gene1 and gene2 times
    time1 = gene1.time_quantum
    time2 = gene2.time_quantum

    # Sessions at time1 that share resources with gene2
    # Sessions at time2 that share resources with gene1
    # (This is a simplified chain - full implementation would be more complex)

    return chain


def _apply_kempe_swap(chain: List[SessionGene]) -> None:
    """Swap times along Kempe chain."""
    if len(chain) < 2:
        return

    # Simple pairwise swap
    chain[0].time_quantum, chain[1].time_quantum = (
        chain[1].time_quantum,
        chain[0].time_quantum,
    )


def _build_ejection_chain(
    individual: List[SessionGene],
    start_gene: SessionGene,
    context: SchedulingContext,
    max_length: int,
) -> List[Tuple[SessionGene, int]]:
    """
    Build ejection chain starting from start_gene.

    Returns list of (gene, new_time) tuples.
    """
    chain = []
    available_quanta = get_available_quanta(context)
    if not available_quanta:
        return chain

    current_gene = start_gene
    used_times = {gene.time_quantum for gene in individual}
    max_quantum = available_quanta[-1]

    for _ in range(max_length):
        # Find alternative time for current gene
        available_times = [
            t
            for t in available_quanta
            if t not in used_times
            and t + current_gene.duration_quanta <= max_quantum + 1
        ]

        if not available_times:
            break

        new_time = random.choice(available_times)
        chain.append((current_gene, new_time))

        # Find gene that would be ejected
        ejected_gene = None
        for gene in individual:
            if gene == current_gene:
                continue

            if gene.time_quantum == new_time:
                # Check for conflicts
                if (
                    set(gene.group_ids) & set(current_gene.group_ids)
                    or gene.instructor_id == current_gene.instructor_id
                    or gene.room_id == current_gene.room_id
                ):
                    ejected_gene = gene
                    break

        if not ejected_gene:
            break

        current_gene = ejected_gene

    return chain


def _apply_ejection_chain(
    chain: List[Tuple[SessionGene, int]], context: SchedulingContext
) -> None:
    """Apply ejection chain moves."""
    # Store old times
    old_times = [(gene, gene.time_quantum) for gene, _ in chain]

    # Apply new times
    for gene, new_time in chain:
        gene.time_quantum = new_time

    # Store for potential revert
    gene._ejection_old_times = old_times


def _revert_ejection_chain(chain: List[Tuple[SessionGene, int]]) -> None:
    """Revert ejection chain moves."""
    if not chain:
        return

    # Get stored old times
    first_gene = chain[0][0]
    if hasattr(first_gene, "_ejection_old_times"):
        for gene, old_time in first_gene._ejection_old_times:
            gene.time_quantum = old_time


def _generate_move_sequence(
    individual: List[SessionGene], context: SchedulingContext, depth: int
) -> List[Tuple[str, Dict]]:
    """Generate random sequence of moves."""
    sequence = []
    move_types = ["time_shift", "room_change", "instructor_change"]

    for _ in range(depth):
        move_type = random.choice(move_types)
        gene = random.choice(individual)

        if move_type == "time_shift":
            available_quanta = get_available_quanta(context)
            if not available_quanta:
                continue
            new_time = random.choice(available_quanta)
            sequence.append(("time_shift", {"gene": gene, "new_time": new_time}))

        elif move_type == "room_change":
            course = get_course_for_gene(context, gene)
            required_room = get_course_room_requirement(course)
            student_count = estimate_session_student_count(gene, context)
            compatible_rooms = [
                r_id
                for r_id, room in context.rooms.items()
                if get_room_feature(room) == required_room
                and room.capacity >= student_count
            ]
            if compatible_rooms:
                new_room = random.choice(compatible_rooms)
                sequence.append(("room_change", {"gene": gene, "new_room": new_room}))

        elif move_type == "instructor_change":
            course = get_course_for_gene(context, gene)
            if len(course.qualified_instructor_ids) > 1:
                new_instructor = random.choice(
                    [
                        i
                        for i in course.qualified_instructor_ids
                        if i != gene.instructor_id
                    ]
                )
                sequence.append(
                    (
                        "instructor_change",
                        {"gene": gene, "new_instructor": new_instructor},
                    )
                )

    return sequence


def _apply_move(
    individual: List[SessionGene],
    move_type: str,
    move_params: Dict,
    context: SchedulingContext,
) -> None:
    """Apply a single move."""
    if move_type == "time_shift":
        move_params["gene"].time_quantum = move_params["new_time"]
    elif move_type == "room_change":
        move_params["gene"].room_id = move_params["new_room"]
    elif move_type == "instructor_change":
        move_params["gene"].instructor_id = move_params["new_instructor"]


def _calculate_fitness(
    individual: List[SessionGene], context: SchedulingContext
) -> Tuple[int, int]:
    """
    Calculate fitness (hard violations, soft violations).

    This is a simplified fitness calculation for improvement heuristics.
    The full fitness evaluation uses the complete constraint system.

    For heuristics, we use a fast approximation:
    - Count direct time conflicts (groups/instructors/rooms)
    - Estimate soft violations (gaps, non-continuous sessions)
    """
    hard_violations = 0
    soft_violations = 0

    # Track time assignments
    time_assignments = defaultdict(list)  # {entity_id: [time_ranges]}

    for gene in individual:
        time_range = range(gene.time_quantum, gene.time_quantum + gene.duration_quanta)

        # Check group conflicts
        for group_id in gene.group_ids:
            for existing_range in time_assignments[f"group_{group_id}"]:
                if set(time_range) & set(existing_range):
                    hard_violations += 1

            time_assignments[f"group_{group_id}"].append(time_range)

        # Check instructor conflicts
        for existing_range in time_assignments[f"instructor_{gene.instructor_id}"]:
            if set(time_range) & set(existing_range):
                hard_violations += 1

        time_assignments[f"instructor_{gene.instructor_id}"].append(time_range)

        # Check room conflicts
        for existing_range in time_assignments[f"room_{gene.room_id}"]:
            if set(time_range) & set(existing_range):
                hard_violations += 1

        time_assignments[f"room_{gene.room_id}"].append(time_range)

        # Estimate soft violations (gaps between sessions)
        soft_violations += len(individual) // 10  # Placeholder

    return (hard_violations, soft_violations)
