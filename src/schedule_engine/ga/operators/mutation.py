import random

from schedule_engine.domain.course import Course
from schedule_engine.domain.gene import SessionGene
from schedule_engine.domain.types import Individual, SchedulingContext


def mutate_gene(gene: SessionGene, context: SchedulingContext) -> SessionGene:
    """
    Performs constraint-aware mutation on a single gene.

    CRITICAL: Course, Group, and Duration are NEVER mutated!
    Only mutates: Instructor, Room, Time slots (when, not how long)

    This preserves the fundamental (course, group) enrollment structure
    and maintains course duration requirements (num_quanta = course.quanta_per_week).

    Architecture: Uses SessionGene's contiguous representation (start_quanta + num_quanta)
    introduced in Nov 2025 to structurally enforce session continuity.
    """
    # Get course info for constraint-aware mutation
    # Look up using tuple key (course_id, course_type)
    course_key = (gene.course_id, gene.course_type)
    course = context.courses.get(course_key)
    # COURSE & GROUP: NEVER MUTATED
    # Keep course_id and group_ids exactly as they are
    new_course_id = gene.course_id
    new_group_ids = gene.group_ids

    # Find qualified instructors for this course
    # instructor.qualified_courses now contains tuples (course_code, course_type)
    qualified_instructors = [
        inst_id
        for inst_id, inst in context.instructors.items()
        if course_key in getattr(inst, "qualified_courses", [])
    ]

    # If current instructor is qualified, keep with high probability (70%)
    if gene.instructor_id in qualified_instructors and random.random() < 0.7:
        new_instructor = gene.instructor_id
    else:
        new_instructor = random.choice(
            qualified_instructors if qualified_instructors else [gene.instructor_id]
        )
    # ROOM: Mutate intelligently
    # Smart room selection with capacity and feature constraints
    # Use first group for room suitability check
    primary_group = gene.group_ids[0] if gene.group_ids else None
    suitable_rooms = find_suitable_rooms_for_course(
        gene.course_id,
        gene.course_type,
        primary_group if primary_group else "",
        context,
    )
    if gene.room_id in suitable_rooms and random.random() < 0.5:
        new_room = gene.room_id  # Keep current room if suitable
    else:
        new_room = random.choice(
            suitable_rooms if suitable_rooms else list(context.rooms.keys())
        )
    # TIME: Mutate intelligently (preserve quanta count!)
    # CRITICAL: Keep the SAME number of quanta to preserve course requirements
    new_quanta = mutate_time_quanta(gene, course, context, individual=None)

    # Convert quanta list to contiguous representation
    from schedule_engine.ga.core.quanta_converter import quanta_list_to_contiguous

    start_q, num_q = quanta_list_to_contiguous(new_quanta)

    return SessionGene(
        course_id=new_course_id,  # NEVER MUTATED
        course_type=gene.course_type,  # NEVER MUTATED
        instructor_id=new_instructor,  # Mutated
        group_ids=new_group_ids,  # NEVER MUTATED
        room_id=new_room,  # Mutated
        start_quanta=start_q,
        num_quanta=num_q,
    )


def mutate_time_quanta(
    gene: SessionGene,
    course: Course | None,
    context: SchedulingContext,
    individual: list[SessionGene] | None = None,
) -> list[int]:
    """
    Conflict-aware time mutation that PRESERVES quanta count.

    CRITICAL: Number of quanta MUST stay the same to maintain course requirements!
    Duration (num_quanta) is fixed by course.quanta_per_week and should never change.

    Only changes WHEN the session happens, not HOW LONG it is.

    When `individual` is provided, avoids quanta already occupied by the gene's
    groups and instructor (conflict-aware).  Otherwise falls back to random.

    Returns:
        List[int]: New quanta list with EXACT same length as gene.num_quanta
    """
    num_quanta = gene.num_quanta

    # 30% chance to keep current time slots completely unchanged
    if random.random() < 0.3:
        return gene.get_quanta_list()

    available_quanta = list(context.available_quanta)

    if len(available_quanta) < num_quanta:
        return gene.get_quanta_list()

    # --- Conflict-aware: build set of quanta blocked for THIS gene ---
    blocked: set[int] = set()
    if individual is not None:
        gene_groups = set(gene.group_ids)
        for other in individual:
            if other is gene:
                continue
            # Same group? Block those quanta.
            if gene_groups & set(other.group_ids):
                for q in range(
                    other.start_quanta, other.start_quanta + other.num_quanta
                ):
                    blocked.add(q)
            # Same instructor? Block those quanta.
            if other.instructor_id == gene.instructor_id:
                for q in range(
                    other.start_quanta, other.start_quanta + other.num_quanta
                ):
                    blocked.add(q)

    # Prefer conflict-free quanta
    free_quanta = [q for q in available_quanta if q not in blocked]
    pool = free_quanta if len(free_quanta) >= num_quanta else available_quanta

    # Attempt to find consecutive slots in the preferred pool
    for _attempt in range(10):
        start_idx = random.randint(0, len(pool) - num_quanta)
        consecutive_quanta = pool[start_idx : start_idx + num_quanta]

        if len(consecutive_quanta) == num_quanta and (
            num_quanta == 1
            or (max(consecutive_quanta) - min(consecutive_quanta)) < num_quanta * 2
        ):
            return consecutive_quanta

    # Fallback to random selection from the pool
    return random.sample(pool, num_quanta)


def find_suitable_rooms_for_course(
    course_id: str, course_type: str, group_id: str, context: SchedulingContext
) -> list[str]:
    """
    Find rooms suitable for a specific course and group combination.
    Takes into account group size, course requirements, and room features.
    """
    course = context.courses.get((course_id, course_type))
    group = context.groups.get(group_id)

    if not course:
        return list(context.rooms.keys())

    # Get course requirements (always a string like "lecture" or "practical")
    required_room_features = getattr(course, "required_room_features", "lecture")

    # Get group size for capacity matching
    group_size = getattr(group, "student_count", 30) if group else 30

    suitable_room_ids = []

    for room_id, room in context.rooms.items():
        # room.room_features is a string like "lecture" or "practical"
        room_features = getattr(room, "room_features", "lecture")
        room_capacity = getattr(room, "capacity", 50)

        # Check capacity requirement first
        if room_capacity < group_size:
            continue

        # Check room type compatibility using centralized logic
        from schedule_engine.utils.room_compatibility import is_room_type_compatible

        if is_room_type_compatible(
            required_room_features.lower().strip(), room_features.lower().strip()
        ):
            suitable_room_ids.append(room_id)

    return suitable_room_ids if suitable_room_ids else list(context.rooms.keys())


def mutate_individual(
    individual: Individual,
    context: SchedulingContext,
    mut_prob: float = 0.2,
    guided: bool = True,
) -> tuple[Individual]:
    """
    Applies mutation to an individual with optional constraint guidance.

    PHASE 2: Constraint-Guided Mutation Integration

    Args:
        individual: List of SessionGene
        context: SchedulingContext with courses, groups, instructors, rooms
        mut_prob: Probability of mutation for each gene (ignored in guided mode)
        guided: If True, use constraint-guided mutation (targets violations)
                If False, use traditional random mutation

    Returns:
        Tuple containing modified individual (DEAP compatibility)
    """
    import logging

    logger = logging.getLogger(__name__)

    if guided:
        # PHASE 2: Constraint-guided mutation (smarter, targets violations)
        logger.debug(" Using constraint-guided mutation (targets violations)")
        from schedule_engine.ga.operators.constraint_guided_mutation import (
            constraint_guided_mutation,
        )

        modified_individual, stats = constraint_guided_mutation(individual, context)
        return (modified_individual,)
    else:
        # Traditional random mutation — now conflict-aware
        logger.debug(" Using random mutation (conflict-aware)")
        for i in range(len(individual)):
            if random.random() < mut_prob:
                gene = individual[i]
                course_key = (gene.course_id, gene.course_type)
                course = context.courses.get(course_key)
                # Conflict-aware time mutation using full individual context
                new_quanta = mutate_time_quanta(
                    gene, course, context, individual=individual
                )
                from schedule_engine.ga.core.quanta_converter import (
                    quanta_list_to_contiguous,
                )

                start_q, num_q = quanta_list_to_contiguous(new_quanta)

                # Instructor: qualified-aware
                qualified_instructors = [
                    inst_id
                    for inst_id, inst in context.instructors.items()
                    if course_key in getattr(inst, "qualified_courses", [])
                ]
                if (
                    gene.instructor_id in qualified_instructors
                    and random.random() < 0.5
                ):
                    new_instructor = gene.instructor_id
                else:
                    new_instructor = random.choice(
                        qualified_instructors
                        if qualified_instructors
                        else [gene.instructor_id]
                    )

                # Room: type-aware
                primary_group = gene.group_ids[0] if gene.group_ids else None
                suitable_rooms = find_suitable_rooms_for_course(
                    gene.course_id,
                    gene.course_type,
                    primary_group if primary_group else "",
                    context,
                )
                if gene.room_id in suitable_rooms and random.random() < 0.3:
                    new_room = gene.room_id
                else:
                    new_room = random.choice(
                        suitable_rooms if suitable_rooms else list(context.rooms.keys())
                    )

                individual[i] = SessionGene(
                    course_id=gene.course_id,
                    course_type=gene.course_type,
                    instructor_id=new_instructor,
                    group_ids=gene.group_ids,
                    room_id=new_room,
                    start_quanta=start_q,
                    num_quanta=num_q,
                )
        return (individual,)
