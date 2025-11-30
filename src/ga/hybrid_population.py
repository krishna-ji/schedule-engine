"""
Hybrid Population Initialization

PHASE 3: Priority 3 Enhancement (ENHANCED)

Generates population with diverse initialization strategies:
- 40% Greedy construction (ENHANCED: was 25%, high quality, better feasibility)
- 40% Smart constraint-aware (was 50%, existing method, balanced)
- 20% Random (was 25%, high diversity, lower quality)

This mix provides:
- Quality: More greedy individuals → better initial feasibility
- Diversity: Random individuals explore search space
- Balance: Smart individuals maintain proven approach

Expected Impact: 20-35% better initial population → faster convergence

Configuration: Greedy percentage controlled by config.enhancements.greedy_initialization_percent
"""

import os
import random
from concurrent.futures import ProcessPoolExecutor

from src.core.types import SchedulingContext
from src.ga.course_group_pairs import generate_course_group_pairs
from src.ga.group_hierarchy import analyze_group_hierarchy
from src.ga.individual import create_individual
from src.ga.population import generate_course_group_aware_population
from src.ga.sessiongene import SessionGene
from src.utils.system_info import get_cpu_count


def generate_hybrid_population(n: int, context: SchedulingContext) -> list:
    """
    Generate population with hybrid initialization strategy.

    ENHANCED: Configurable greedy percentage via config.enhancements

    Composition (default):
    - 40% greedy (ENHANCED: was 25%, constructive heuristic, feasible solutions)
    - 40% constraint-aware (was 50%, existing smart seeding)
    - 20% random (was 25%, pure random for diversity)

    Args:
        n: Population size
        context: SchedulingContext with courses, groups, instructors, rooms, quanta

    Returns:
        List of n individuals
    """
    population = []

    # ENHANCEMENT: Get greedy percentage from config
    from src.config import get_config

    enhancement_cfg = get_config().enhancements

    if enhancement_cfg.master_enabled:
        greedy_percent = enhancement_cfg.greedy_initialization_percent
    else:
        greedy_percent = 0.25  # Fallback to original 25%

    # Calculate counts for each strategy
    greedy_count = int(n * greedy_percent)  # Allow 0 if percentage is 0
    random_count = max(1, int(n * 0.2))  # Fixed 20% random (minimum 1)
    smart_count = n - greedy_count - random_count  # Rest are smart

    # Detect if we're in a worker process (suppress info messages)
    silent = os.environ.get("_GA_WORKER_PROCESS") == "1"

    if not silent:
        print(
            f"Hybrid initialization: {greedy_count} greedy, {smart_count} smart, {random_count} random"
        )

    # Pre-generate course-group pairs ONCE (avoid duplicate warnings)
    # Set silent=True since warnings already shown in input encoder table
    hierarchy = analyze_group_hierarchy(context.groups)
    pair_tuples = generate_course_group_pairs(
        context.courses, context.groups, hierarchy, silent=True
    )

    # Determine parallelization strategy
    num_workers = get_cpu_count()
    use_parallel = num_workers > 1 and n >= 10

    # Generate greedy individuals using registered construction heuristics
    # Cycle through: largest_degree_first, most_constrained_first, earliest_deadline_first
    from src.heuristics.construction import (
        earliest_deadline_first,
        largest_degree_first,
        most_constrained_first,
    )

    construction_heuristics = [
        largest_degree_first,
        most_constrained_first,
        earliest_deadline_first,
    ]

    for i in range(greedy_count):
        # Round-robin through construction heuristics for diversity
        heuristic = construction_heuristics[i % len(construction_heuristics)]
        try:
            genes = heuristic(context)
            if genes:
                population.append(create_individual(genes))
        except Exception:
            # Fallback to smart if construction heuristic fails
            fallback = generate_course_group_aware_population(1, context)
            if fallback:
                population.append(fallback[0])

    # Generate smart constraint-aware individuals (50%)
    # Note: generate_course_group_aware_population handles its own parallelization
    smart_population = generate_course_group_aware_population(smart_count, context)
    population.extend(smart_population)

    # Generate random individuals (25%)
    if use_parallel:
        random_tasks = [(context, pair_tuples) for _ in range(random_count)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_random_construction_wrapper, random_tasks))
        population.extend([ind for ind in results if ind is not None])
    else:
        for _i in range(random_count):
            individual = _random_construction(context, pair_tuples)
            if individual:
                population.append(create_individual(individual))

    # Ensure we have exactly n individuals
    while len(population) < n:
        # Fill with smart individuals if needed
        extra = generate_course_group_aware_population(1, context)
        population.extend(extra)

    return population[:n]  # Trim to exactly n if we have more


def _greedy_construction(
    context: SchedulingContext, pair_tuples: list[tuple]
) -> list[SessionGene]:
    """
    Greedy constructive heuristic for creating feasible schedule.

    Strategy:
    1. Sort course-group pairs by difficulty (most constrained first)
    2. For each pair, assign first feasible time/room/instructor
    3. Track resource usage to avoid conflicts

    Args:
        context: SchedulingContext
        pair_tuples: Pre-generated course-group pairs (avoids duplicate warnings)

    Returns:
        Individual with SessionGenes (feasible if possible)
    """
    if not pair_tuples:
        return []

    # Sort pairs by difficulty (most constrained first)
    sorted_pairs = sorted(
        pair_tuples,
        key=lambda p: _calculate_constraint_difficulty(p, context),
        reverse=True,  # Most difficult first
    )

    individual = []
    group_schedule: dict[tuple[str, int], bool] = {}  # {(group_id, quantum): True}
    room_usage: dict[tuple[str, int], bool] = {}  # {(room_id, quantum): True}
    instructor_usage: dict[
        tuple[str, int], bool
    ] = {}  # {(instructor_id, quantum): True}

    # Import subsession breaker (canonical L/T/P logic)
    from src.ga.population import get_subsession_durations

    # Schedule each pair greedily
    for course_key, group_ids, _session_type, num_quanta in sorted_pairs:
        if num_quanta == 0:
            continue

        course = context.courses.get(course_key)
        if not course:
            continue

        # FIXED: Break into subsessions using canonical logic
        # Theory → [2, 2, ...] with [1] if odd
        # Practical → [full_duration]
        subsession_durations = get_subsession_durations(
            course.quanta_per_week, course.course_type
        )

        # Schedule each subsession separately
        for _subsession_idx, subsession_duration in enumerate(subsession_durations):
            # Find first feasible assignment for THIS subsession
            gene = _find_feasible_assignment(
                course_key,
                group_ids,
                subsession_duration,  # Use subsession duration, not full course duration
                context,
                group_schedule,
                room_usage,
                instructor_usage,
            )

            # FIXED: Always create gene, even if greedy fails (use random fallback)
            if not gene:
                gene = _random_gene(course_key, group_ids, subsession_duration, context)

            if gene:
                individual.append(gene)

                # Mark resources as used
                for quantum in range(gene.start_quanta, gene.end_quanta):
                    for gid in gene.group_ids:
                        group_schedule[(gid, quantum)] = True
                    room_usage[(gene.room_id, quantum)] = True
                    instructor_usage[(gene.instructor_id, quantum)] = True

    return individual


def _calculate_constraint_difficulty(
    pair_tuple: tuple, context: SchedulingContext
) -> float:
    """
    Estimate scheduling difficulty for a course-group pair.

    Factors:
    - Few qualified instructors → harder
    - Long duration → harder
    - Limited suitable rooms → harder

    Returns:
        Float score (higher = more difficult)
    """
    course_key, group_ids, session_type, num_quanta = pair_tuple
    course = context.courses.get(course_key)

    if not course:
        return 0.0

    score = 0.0

    # Instructor availability
    qualified_count = len(course.qualified_instructor_ids)
    if qualified_count == 0:
        score += 100.0  # Very hard
    else:
        score += 10.0 / qualified_count  # Fewer = harder

    # Duration constraint
    score += num_quanta * 2.0  # Longer sessions = harder

    # Room type constraint (labs are often scarcer)
    if session_type == "practical":
        score += 15.0  # Labs are typically more constrained

    # Multiple groups (theory) vs single group (practical)
    if len(group_ids) > 1:
        score += len(group_ids) * 3.0  # Coordinating multiple groups harder

    return score


def _find_feasible_assignment(
    course_key: tuple[str, str],
    group_ids: list[str],
    num_quanta: int,
    context: SchedulingContext,
    group_schedule: dict,
    room_usage: dict,
    instructor_usage: dict,
) -> SessionGene | None:
    """
    Find first feasible assignment for a course-group pair.

    Tries to find time/room/instructor combination with no conflicts.
    Returns None if no feasible assignment found after reasonable attempts.
    """
    course = context.courses.get(course_key)
    if not course:
        return None

    # Try multiple starting positions
    available_quanta = list(context.available_quanta)
    max_attempts = min(50, len(available_quanta))

    for _attempt in range(max_attempts):
        # FIXED: Handle cases where num_quanta > len(available_quanta)
        if num_quanta <= len(available_quanta):
            # Try contiguous block of quanta
            start_idx = random.randint(0, len(available_quanta) - num_quanta)
            candidate_quanta = available_quanta[start_idx : start_idx + num_quanta]
        else:
            # Wrap around to get exactly num_quanta
            candidate_quanta = []
            while len(candidate_quanta) < num_quanta:
                candidate_quanta.extend(available_quanta)
            candidate_quanta = candidate_quanta[:num_quanta]

        if len(candidate_quanta) != num_quanta:
            continue

        # Check group availability
        group_free = all(
            (gid, q) not in group_schedule
            for gid in group_ids
            for q in candidate_quanta
        )
        if not group_free:
            continue

        # Find available room
        room_id = _find_available_room(
            candidate_quanta,
            context.rooms,
            room_usage,
            course_key[1],  # course_type
        )
        if not room_id:
            continue

        # Find available qualified instructor
        instructor_id = _find_available_instructor(
            candidate_quanta, course, context.instructors, instructor_usage
        )
        if not instructor_id:
            continue

        # Success! Create gene with contiguous quanta
        from src.ga.quanta_converter import quanta_list_to_contiguous

        start_q, num_q = quanta_list_to_contiguous(sorted(candidate_quanta))

        return SessionGene(
            course_id=course_key[0],
            course_type=course_key[1],
            instructor_id=instructor_id,
            group_ids=sorted(group_ids),
            room_id=room_id,
            start_quanta=start_q,
            num_quanta=num_q,
        )

    # Fallback to random if no feasible found
    return _random_gene(course_key, group_ids, num_quanta, context)


def _find_available_room(
    quanta: list[int], rooms: dict, room_usage: dict, course_type: str
) -> str | None:
    """Find first room available during quanta and matching course type."""
    for room_id, room in rooms.items():
        # Check room type matches
        room_type = getattr(room, "room_type", "lecture").lower()

        if course_type == "practical" and room_type != "lab":
            continue
        # Allow any room for theory (including labs)

        # Check room is free during all quanta
        free = all((room_id, q) not in room_usage for q in quanta)
        if free:
            return str(room_id)

    return None


def _find_available_instructor(
    quanta: list[int], course, instructors: dict, instructor_usage: dict
) -> str | None:
    """Find first qualified instructor available during quanta."""
    qualified_ids = getattr(course, "qualified_instructor_ids", [])

    for instructor_id in qualified_ids:
        instructor = instructors.get(instructor_id)
        if not instructor:
            continue

        # Check availability constraints if present
        availability = getattr(instructor, "availability", None)
        if availability:
            available = all(q in availability for q in quanta)
            if not available:
                continue

        # Check not double-booked
        free = all((instructor_id, q) not in instructor_usage for q in quanta)
        if free:
            return str(instructor_id)

    return None


def _random_construction(
    context: SchedulingContext, pair_tuples: list[tuple]
) -> list[SessionGene]:
    """
    Generate completely random individual for diversity.

    Uses existing constraint-aware generation but with maximum randomness.

    Args:
        context: SchedulingContext
        pair_tuples: Pre-generated pairs (unused, kept for API consistency)
    """
    # Generate single individual with smart approach
    # (Already has randomness built-in)
    pop = generate_course_group_aware_population(1, context)
    return pop[0] if pop else []


def _random_gene(
    course_key: tuple[str, str],
    group_ids: list[str],
    num_quanta: int,
    context: SchedulingContext,
) -> SessionGene:
    """Generate random SessionGene (fallback when greedy fails)."""
    course = context.courses.get(course_key)

    # Random time slots
    available_quanta_list = list(
        context.available_quanta
    )  # Convert to list for sampling
    if len(available_quanta_list) >= num_quanta:
        quanta = sorted(random.sample(available_quanta_list, num_quanta))
    else:
        # FIXED: Wrap around to get exactly num_quanta (never reduce!)
        quanta = []
        while len(quanta) < num_quanta:
            quanta.extend(available_quanta_list)
        quanta = sorted(quanta[:num_quanta])

    # Random room
    room_id = random.choice(list(context.rooms.keys())) if context.rooms else "ROOM1"

    # Random qualified instructor (or any instructor)
    if course and course.qualified_instructor_ids:
        instructor_id = random.choice(course.qualified_instructor_ids)
    elif context.instructors:
        instructor_id = random.choice(list(context.instructors.keys()))
    else:
        instructor_id = "INST1"

    # Convert quanta list to contiguous representation
    from src.ga.quanta_converter import quanta_list_to_contiguous

    start_q, num_q = quanta_list_to_contiguous(quanta)

    return SessionGene(
        course_id=course_key[0],
        course_type=course_key[1],
        instructor_id=instructor_id,
        group_ids=sorted(group_ids),
        room_id=room_id,
        start_quanta=start_q,
        num_quanta=num_q,
    )


def _greedy_construction_wrapper(args):
    """Wrapper for parallel greedy construction."""
    context, pair_tuples = args
    individual = _greedy_construction(context, pair_tuples)
    if individual:
        return create_individual(individual)
    return None


def _random_construction_wrapper(args):
    """Wrapper for parallel random construction."""
    context, pair_tuples = args
    individual = _random_construction(context, pair_tuples)
    if individual:
        return create_individual(individual)
    return None
