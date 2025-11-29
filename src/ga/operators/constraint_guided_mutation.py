"""
Constraint-Guided Mutation Operator

PHASE 2: Priority 2 Enhancement

Targets sessions with constraint violations for focused repair.
Instead of random mutation, identifies problematic sessions and mutates those.

Strategy:
1. Decode individual to CourseSession objects
2. Identify sessions causing hard violations
3. Mutate violating sessions preferentially (80% probability)
4. Fallback to random mutation (20% for diversity)

Expected Impact: 20-30% faster convergence to zero violations.
"""

import random

from src.core.types import SchedulingContext
from src.decoder.individual_decoder import decode_individual
from src.entities.decoded_session import CourseSession


def constraint_guided_mutation(
    individual, context: SchedulingContext
) -> tuple[list, dict]:
    """
    Mutate genes corresponding to sessions with violations.

    Args:
        individual: List of SessionGene
        context: SchedulingContext with courses, groups, instructors, rooms

    Returns:
        Tuple of (modified individual, mutation stats dict)
    """

    # Decode to identify violations
    decoded = decode_individual(
        individual,
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    # Find sessions with violations
    violating_indices = _find_violating_sessions(decoded, context)

    # Decide whether to target violation or mutate randomly
    if violating_indices and random.random() < 0.8:
        # Target violation (80% of the time)
        target_idx = random.choice(violating_indices)
        _mutate_session(individual[target_idx], context)
        return individual, {"targeted_mutations": 1, "random_mutations": 0}
    else:
        # Random mutation (20% for diversity)
        if len(individual) > 0:
            target_idx = random.randint(0, len(individual) - 1)
            _mutate_session(individual[target_idx], context)
        return individual, {"targeted_mutations": 0, "random_mutations": 1}


def _find_violating_sessions(
    decoded_sessions: list[CourseSession], context: SchedulingContext
) -> list[int]:
    """
    Identify indices of sessions causing hard constraint violations.

    Checks:
    - Group overlaps (double-booking)
    - Room conflicts (double-booking)
    - Instructor conflicts (double-booking)
    - Instructor qualification mismatches

    NOTE: Availability checks removed (see COMPLETE_AVAILABILITY_REMOVAL.md)

    Returns:
        List of integer indices into decoded_sessions
    """
    violating = []

    for idx, session in enumerate(decoded_sessions):
        # Check group overlaps
        if _has_group_overlap(session, decoded_sessions, idx):
            violating.append(idx)
            continue

        # Check room conflicts
        if _has_room_conflict(session, decoded_sessions, idx):
            violating.append(idx)
            continue

        # Check instructor conflicts
        if _has_instructor_conflict(session, decoded_sessions, idx):
            violating.append(idx)
            continue

        # Check instructor qualification
        if not _is_instructor_qualified(session, context):
            violating.append(idx)
            continue

    return violating


def _is_instructor_qualified(
    session: CourseSession, context: SchedulingContext
) -> bool:
    """Check if instructor is qualified to teach the course."""
    course = context.courses.get(session.course_id)  # type: ignore[call-overload]
    if not course:
        return True  # Unknown course, assume OK

    # If course has no qualification requirements, anyone can teach
    if not course.qualified_instructor_ids:
        return True

    # Check if instructor is in the qualified list
    return session.instructor_id in course.qualified_instructor_ids


def _has_group_overlap(
    session: CourseSession, all_sessions: list[CourseSession], current_idx: int
) -> bool:
    """Check if group has overlapping sessions."""
    for idx, other in enumerate(all_sessions):
        if idx == current_idx:
            continue

        # Check if any group in session overlaps with any group in other
        session_groups = (
            session.group_ids
            if isinstance(session.group_ids, list)
            else [session.group_ids]
        )
        other_groups = (
            other.group_ids if isinstance(other.group_ids, list) else [other.group_ids]
        )

        # Same group and overlapping time?
        if (set(session_groups) & set(other_groups)) and (
            set(session.session_quanta) & set(other.session_quanta)
        ):
            return True
    return False


def _has_room_conflict(
    session: CourseSession, all_sessions: list[CourseSession], current_idx: int
) -> bool:
    """Check if room is double-booked."""
    for idx, other in enumerate(all_sessions):
        if idx == current_idx:
            continue

        # Same room and overlapping time?
        if session.room_id == other.room_id and (
            set(session.session_quanta) & set(other.session_quanta)
        ):
            return True
    return False


def _has_instructor_conflict(
    session: CourseSession, all_sessions: list[CourseSession], current_idx: int
) -> bool:
    """Check if instructor is double-booked."""
    for idx, other in enumerate(all_sessions):
        if idx == current_idx:
            continue

        # Same instructor and overlapping time?
        if session.instructor_id == other.instructor_id and (
            set(session.session_quanta) & set(other.session_quanta)
        ):
            return True
    return False


def _mutate_session(gene, context: SchedulingContext):
    """
    Mutate a single SessionGene.

    Strategy (weighted random):
    - 40% chance: change time slots
    - 30% chance: change room
    - 20% chance: change instructor
    - 10% chance: change multiple attributes (aggressive)
    """
    mutation_type = random.random()

    # Convert available_quanta to list for sampling
    available_quanta_list = list(context.available_quanta)

    if mutation_type < 0.4:
        # Change time slots - find contiguous block
        num_quanta = gene.num_quanta
        if num_quanta > 0 and len(available_quanta_list) >= num_quanta:
            # Find a random valid start time that allows contiguous block

            valid_starts = [
                q
                for q in available_quanta_list
                if all((q + i) in available_quanta_list for i in range(num_quanta))
            ]
            if valid_starts:
                gene.start_quanta = random.choice(valid_starts)
                # num_quanta stays the same

    elif mutation_type < 0.7:
        # Change room
        if context.rooms:
            gene.room_id = random.choice(list(context.rooms.keys()))

    elif mutation_type < 0.9:
        # Change instructor (must be qualified)
        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        if course and course.qualified_instructor_ids:
            gene.instructor_id = random.choice(course.qualified_instructor_ids)
        elif context.instructors:
            # Fallback to any instructor if no qualified ones
            gene.instructor_id = random.choice(list(context.instructors.keys()))

    else:
        # Change multiple attributes (aggressive mutation)
        num_quanta = gene.num_quanta
        if num_quanta > 0 and len(available_quanta_list) >= num_quanta:
            # Find a random valid start time that allows contiguous block
            valid_starts = [
                q
                for q in available_quanta_list
                if all((q + i) in available_quanta_list for i in range(num_quanta))
            ]
            if valid_starts:
                gene.start_quanta = random.choice(valid_starts)

        if context.rooms:
            gene.room_id = random.choice(list(context.rooms.keys()))
