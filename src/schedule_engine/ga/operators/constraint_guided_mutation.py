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

from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.session import CourseSession
from schedule_engine.domain.types import Individual, SchedulingContext
from schedule_engine.io.decoder import decode_individual


def constraint_guided_mutation(
    individual: Individual, context: SchedulingContext
) -> tuple[Individual, dict[str, int]]:
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

    # Repair multiple violating genes per call (not just 1)
    # This makes mutation strong enough to overcome crossover disruption
    max_repairs = min(len(violating_indices), max(3, len(violating_indices) // 5))
    targeted = 0
    rand_mut = 0

    if violating_indices:
        # Shuffle to avoid always fixing the same genes first
        repair_targets = random.sample(violating_indices, max_repairs)
        for target_idx in repair_targets:
            if random.random() < 0.8:
                _mutate_session(individual[target_idx], context, individual=individual)
                targeted += 1
            else:
                # Random mutation for diversity
                rand_idx = random.randint(0, len(individual) - 1)
                _mutate_session(individual[rand_idx], context, individual=individual)
                rand_mut += 1
    else:
        # No violations found — random mutation for diversity
        if len(individual) > 0:
            target_idx = random.randint(0, len(individual) - 1)
            _mutate_session(individual[target_idx], context, individual=individual)
            rand_mut = 1

    return individual, {"targeted_mutations": targeted, "random_mutations": rand_mut}


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
        # group_ids is always a list[str] per SessionGene definition
        session_groups = session.group_ids
        other_groups = other.group_ids

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


def _mutate_session(
    gene: SessionGene,
    context: SchedulingContext,
    individual: list[SessionGene] | None = None,
) -> None:
    """
    Mutate a single SessionGene — conflict-aware when individual is provided.

    Strategy (weighted random):
    - 40% chance: change time slots (conflict-aware)
    - 30% chance: change room (type-aware)
    - 20% chance: change instructor
    - 10% chance: change multiple attributes (aggressive)
    """
    mutation_type = random.random()

    # Build conflict-aware available quanta if individual is provided
    available_quanta_list = list(context.available_quanta)

    # Build blocked set: quanta used by same group/instructor in other genes
    blocked: set[int] = set()
    if individual is not None:
        gene_groups = set(gene.group_ids)
        for other in individual:
            if other is gene:
                continue
            if gene_groups & set(other.group_ids):
                for q in range(
                    other.start_quanta, other.start_quanta + other.num_quanta
                ):
                    blocked.add(q)
            if other.instructor_id == gene.instructor_id:
                for q in range(
                    other.start_quanta, other.start_quanta + other.num_quanta
                ):
                    blocked.add(q)

    # Prefer conflict-free quanta for time mutations
    free_quanta = [q for q in available_quanta_list if q not in blocked]
    time_pool = (
        free_quanta if len(free_quanta) >= gene.num_quanta else available_quanta_list
    )

    if mutation_type < 0.4:
        # Change time slots - find contiguous block avoiding conflicts
        num_quanta = gene.num_quanta
        if num_quanta > 0 and len(time_pool) >= num_quanta:
            valid_starts = [
                q
                for q in time_pool
                if all((q + i) in time_pool for i in range(num_quanta))
            ]
            if valid_starts:
                gene.start_quanta = random.choice(valid_starts)

    elif mutation_type < 0.7:
        # Change room — type-aware selection
        if context.rooms:
            from schedule_engine.ga.operators.mutation import (
                find_suitable_rooms_for_course,
            )

            primary_group = gene.group_ids[0] if gene.group_ids else ""
            suitable = find_suitable_rooms_for_course(
                gene.course_id, gene.course_type, primary_group, context
            )
            if suitable:
                gene.room_id = random.choice(suitable)
            else:
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
        # Change multiple attributes (aggressive mutation) — room is type-aware
        num_quanta = gene.num_quanta
        if num_quanta > 0 and len(time_pool) >= num_quanta:
            valid_starts = [
                q
                for q in time_pool
                if all((q + i) in time_pool for i in range(num_quanta))
            ]
            if valid_starts:
                gene.start_quanta = random.choice(valid_starts)

        if context.rooms:
            from schedule_engine.ga.operators.mutation import (
                find_suitable_rooms_for_course,
            )

            primary_group = gene.group_ids[0] if gene.group_ids else ""
            suitable = find_suitable_rooms_for_course(
                gene.course_id, gene.course_type, primary_group, context
            )
            if suitable:
                gene.room_id = random.choice(suitable)
            else:
                gene.room_id = random.choice(list(context.rooms.keys()))
