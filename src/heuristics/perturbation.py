"""
Perturbation Heuristics - Shake Solutions to Escape Local Optima

Provides perturbation operators that strategically modify schedules
to escape local optima and explore new regions of the search space.

Perturbation heuristics are useful for:
1. Diversification (escape plateaus in GA)
2. Iterated local search (shake after convergence)
3. Large neighborhood search (destroy-repair patterns)

Strategies:
1. Random Swap: Exchange time/room between two sessions
2. Temporal Shift: Move session to different time slot
3. Room Shuffle: Reassign rooms to compatible sessions
4. Instructor Reassign: Change instructor to another qualified option

Architecture:
- Decorator-based registration with @perturbation_heuristic
- Modifies individual in-place (returns modification count)
- Invalidates fitness after modification
- Can be chained for multi-move perturbations

Usage:
    from src.heuristics.perturbation import temporal_shift

    # Perturb a schedule
    modifications = temporal_shift(individual, context, delta=3)
    print(f"Shifted {modifications} sessions")
"""

import random
from collections import defaultdict

from src.domain.types import SchedulingContext
from src.domain.gene import SessionGene
from src.heuristics.registry import perturbation_heuristic
from src.heuristics.utils import (
    estimate_session_student_count,
    get_available_quanta,
    get_course_for_gene,
    get_course_room_requirement,
    get_room_feature,
    is_instructor_available,
    move_gene_to_time_if_valid,
)

# ================
# RANDOM SWAP (Exchange time/room between two sessions)
# ================


@perturbation_heuristic(
    name="random_swap",
    description="Randomly swap time slots or rooms between two compatible sessions",
    priority=1,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def random_swap(
    individual: list[SessionGene],
    context: SchedulingContext,
    swap_type: str = "time",
    num_swaps: int = 1,
) -> int:
    """
    Randomly swap time slots or rooms between two sessions.

    Swap types:
    - "time": Exchange time_quantum between two sessions
    - "room": Exchange room_id between two compatible sessions
    - "both": Exchange both time and room

    Args:
        individual: List of SessionGene to modify
        context: Scheduling context
        swap_type: Type of swap ("time", "room", or "both")
        num_swaps: Number of swaps to perform

    Returns:
        Number of successful swaps performed
    """
    if len(individual) < 2:
        return 0

    swaps_performed = 0

    for _ in range(num_swaps):
        # Select two random genes
        gene1, gene2 = random.sample(individual, 2)

        if swap_type == "time" or swap_type == "both":
            # Swap entire time blocks (quanta) between sessions
            # Use new contiguous block representation (start_quanta, num_quanta)
            original_start_gene1 = gene1.start_quanta
            original_num_gene1 = gene1.num_quanta

            gene1.start_quanta = gene2.start_quanta
            gene1.num_quanta = gene2.num_quanta
            gene1.__post_init__()  # Re-validate after swap

            gene2.start_quanta = original_start_gene1
            gene2.num_quanta = original_num_gene1
            gene2.__post_init__()  # Re-validate after swap

            swaps_performed += 1

        if swap_type == "room" or swap_type == "both":
            # Only swap rooms if both courses compatible with both rooms
            course1 = get_course_for_gene(context, gene1)
            course2 = get_course_for_gene(context, gene2)
            room1 = context.rooms[gene1.room_id]
            room2 = context.rooms[gene2.room_id]

            # Check compatibility
            required_type_1 = get_course_room_requirement(course1)
            required_type_2 = get_course_room_requirement(course2)
            room1_type = get_room_feature(room1)
            room2_type = get_room_feature(room2)
            students_course1 = estimate_session_student_count(gene1, context)
            students_course2 = estimate_session_student_count(gene2, context)
            room1_ok_for_course2 = (
                room1_type == required_type_2 and room1.capacity >= students_course2
            )
            room2_ok_for_course1 = (
                room2_type == required_type_1 and room2.capacity >= students_course1
            )

            if room1_ok_for_course2 and room2_ok_for_course1:
                gene1.room_id, gene2.room_id = gene2.room_id, gene1.room_id
                swaps_performed += 1

    # Invalidate fitness
    if hasattr(individual, "fitness") and swaps_performed > 0:
        del individual.fitness.values

    return swaps_performed


# ================
# TEMPORAL SHIFT (Move session to different time slot)
# ================


@perturbation_heuristic(
    name="temporal_shift",
    description="Shift sessions forward or backward in time by delta quanta",
    priority=2,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def temporal_shift(
    individual: list[SessionGene],
    context: SchedulingContext,
    delta: int | None = None,
    probability: float = 0.3,
) -> int:
    """
    Shift sessions forward or backward in time.

    Each session has 'probability' chance of being shifted by 'delta' quanta.
    Delta can be positive (shift forward) or negative (shift backward).

    Args:
        individual: List of SessionGene to modify
        context: Scheduling context
        delta: Time shift in quanta (random if None)
        probability: Probability of shifting each session

    Returns:
        Number of sessions shifted
    """
    available_quanta = get_available_quanta(context)
    if not available_quanta:
        return 0

    valid_quanta = set(available_quanta)
    shifts_performed = 0

    for gene in individual:
        if random.random() > probability:
            continue

        # Determine shift delta
        shift_delta = random.randint(-5, 5) if delta is None else delta

        # Calculate new time quantum based on start of session
        candidate_start = gene.time_quantum + shift_delta

        if candidate_start == gene.time_quantum:
            continue

        if move_gene_to_time_if_valid(gene, candidate_start, valid_quanta):
            shifts_performed += 1

    # Invalidate fitness
    if hasattr(individual, "fitness") and shifts_performed > 0:
        del individual.fitness.values

    return shifts_performed


# ================
# ROOM SHUFFLE (Reassign rooms to compatible sessions)
# ================


@perturbation_heuristic(
    name="room_shuffle",
    description="Randomly reassign rooms to sessions while maintaining compatibility",
    priority=3,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def room_shuffle(
    individual: list[SessionGene],
    context: SchedulingContext,
    probability: float = 0.2,
) -> int:
    """
    Randomly reassign rooms to sessions.

    Only reassigns to compatible rooms (correct type and sufficient capacity).
    Each session has 'probability' chance of being reassigned.

    Args:
        individual: List of SessionGene to modify
        context: Scheduling context
        probability: Probability of reassigning each session

    Returns:
        Number of rooms reassigned
    """
    reassignments = 0

    # Group rooms by type for efficient lookup
    rooms_by_type = defaultdict(list)
    for room_id, room in context.rooms.items():
        rooms_by_type[get_room_feature(room)].append((room_id, room))

    for gene in individual:
        if random.random() > probability:
            continue

        course = get_course_for_gene(context, gene)

        # Get compatible rooms
        required_room = get_course_room_requirement(course)
        student_count = estimate_session_student_count(gene, context)
        compatible_rooms = [
            room_id
            for room_id, room in rooms_by_type.get(required_room, [])
            if room.capacity >= student_count
        ]

        if not compatible_rooms:
            continue

        # Exclude current room if there are alternatives
        if len(compatible_rooms) > 1 and gene.room_id in compatible_rooms:
            compatible_rooms.remove(gene.room_id)

        # Assign new room
        gene.room_id = random.choice(compatible_rooms)
        reassignments += 1

    # Invalidate fitness
    if hasattr(individual, "fitness") and reassignments > 0:
        del individual.fitness.values

    return reassignments


# ================
# INSTRUCTOR REASSIGN (Change instructor to qualified alternative)
# ================


@perturbation_heuristic(
    name="instructor_reassign",
    description="Reassign instructors to other qualified instructors for courses",
    priority=4,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def instructor_reassign(
    individual: list[SessionGene],
    context: SchedulingContext,
    probability: float = 0.15,
    prefer_available: bool = True,
) -> int:
    """
    Reassign instructors to qualified alternatives.

    Optionally prefers instructors who are available at the session time.
    Each session has 'probability' chance of being reassigned.

    Args:
        individual: List of SessionGene to modify
        context: Scheduling context
        probability: Probability of reassigning each session
        prefer_available: Prefer instructors available at session time

    Returns:
        Number of instructors reassigned
    """
    reassignments = 0

    for gene in individual:
        if random.random() > probability:
            continue

        course = get_course_for_gene(context, gene)

        # Get qualified instructors
        qualified_instructors = list(course.qualified_instructor_ids)

        if len(qualified_instructors) <= 1:
            continue  # No alternatives

        # Filter by availability if requested
        if prefer_available:
            available_instructors = []
            time_range = range(
                gene.time_quantum, gene.time_quantum + gene.duration_quanta
            )

            for instructor_id in qualified_instructors:
                instructor = context.instructors.get(instructor_id)
                if instructor and is_instructor_available(instructor, time_range):
                    available_instructors.append(instructor_id)

            if available_instructors:
                qualified_instructors = available_instructors

        # Exclude current instructor if there are alternatives
        if (
            len(qualified_instructors) > 1
            and gene.instructor_id in qualified_instructors
        ):
            qualified_instructors.remove(gene.instructor_id)

        if not qualified_instructors:
            continue

        # Assign new instructor
        gene.instructor_id = random.choice(qualified_instructors)
        reassignments += 1

    # Invalidate fitness
    if hasattr(individual, "fitness") and reassignments > 0:
        del individual.fitness.values

    return reassignments


# ================
# MULTI-PERTURBATION (Chain multiple perturbation operators)
# ================


@perturbation_heuristic(
    name="multi_perturbation",
    description="Apply multiple perturbation operators in sequence for stronger diversification",
    priority=5,
    enabled_by_default=False,  # Disabled by default (aggressive)
    requires_population=False,
    modifies_individual=True,
)
def multi_perturbation(
    individual: list[SessionGene],
    context: SchedulingContext,
    operators: list[str] | None = None,
) -> int:
    """
    Apply multiple perturbation operators in sequence.

    Useful for strong diversification or escaping deep local optima.

    Args:
        individual: List of SessionGene to modify
        context: Scheduling context
        operators: List of operator names to apply (default: all)

    Returns:
        Total number of modifications across all operators
    """
    if operators is None:
        operators = ["temporal_shift", "room_shuffle", "instructor_reassign"]

    total_modifications = 0

    for operator_name in operators:
        if operator_name == "random_swap":
            total_modifications += random_swap(
                individual, context, swap_type="both", num_swaps=2
            )
        elif operator_name == "temporal_shift":
            total_modifications += temporal_shift(individual, context, probability=0.2)
        elif operator_name == "room_shuffle":
            total_modifications += room_shuffle(individual, context, probability=0.15)
        elif operator_name == "instructor_reassign":
            total_modifications += instructor_reassign(
                individual, context, probability=0.1
            )

    return total_modifications
