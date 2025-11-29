from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor

from src.core.types import SchedulingContext
from src.ga.course_group_pairs import generate_course_group_pairs
from src.ga.group_hierarchy import analyze_group_hierarchy
from src.ga.individual import create_individual
from src.ga.sessiongene import SessionGene
from src.utils.console_service import get_console
from src.utils.system_info import get_cpu_count


def get_subsession_durations(quanta_per_week: int, course_type: str) -> list[int]:
    """
    Break course duration into subsessions based on pedagogical requirements.

    Theory courses: Break into 2-quanta blocks (with 1-quanta remainder if odd)
    Practical courses: Single continuous session (can span multiple days)

    Args:
        quanta_per_week: Total quanta required per week
        course_type: "theory" or "practical"

    Returns:
        List of subsession durations in quanta

    Examples:
        Theory 6 quanta → [2, 2, 2] (three 2-hour sessions)
        Theory 5 quanta → [2, 2, 1] (two 2-hour + one 1-hour)
        Practical 30 quanta → [30] (single 30-hour studio)
    """
    if course_type == "practical":
        # Practicals: Single continuous session (can span days if > quanta_per_day)
        return [quanta_per_week]
    else:
        # Theory: Break into 2-quanta blocks with remainder
        if quanta_per_week % 2 == 0:
            # Even: All 2-quanta blocks (e.g., 6 → [2,2,2])
            return [2] * (quanta_per_week // 2)
        else:
            # Odd: 2-quanta blocks + 1-quanta remainder (e.g., 5 → [2,2,1])
            blocks = [2] * (quanta_per_week // 2)
            blocks.append(1)
            return blocks


from src.utils.parallel_worker import get_worker_context, init_worker

console = get_console()


def generate_pure_random_population(
    n: int,
    context: SchedulingContext,
    parallel: bool = True,
) -> list:
    """
    Generate population with PURE RANDOM initialization (no heuristics).

    For baseline experiments - no conflict avoidance, no greedy logic.
    Each gene gets completely random instructor, room, and time slot.

    Args:
        n: Population size
        context: SchedulingContext
        parallel: Enable/disable parallelization

    Returns:
        List of n individuals (each with random genes)
    """
    hierarchy = analyze_group_hierarchy(context.groups)
    course_group_pairs = generate_course_group_pairs(
        context.courses, context.groups, hierarchy, silent=True
    )

    # Generate population sequentially (parallel requires data_dir approach like other functions)
    # TODO: Implement parallel version using data_dir + worker loading pattern
    population = []

    for _i in range(n):
        genes = []
        for course_id, group_ids, _session_type, _num_quanta in course_group_pairs:
            course = context.courses.get(course_id)
            if not course:
                continue

            # Break into subsessions
            subsession_durations = get_subsession_durations(
                course.quanta_per_week, course.course_type
            )

            for subsession_duration in subsession_durations:
                # Pure random assignment - no conflict checking!
                gene = _create_pure_random_gene(
                    course_id, group_ids, subsession_duration, course, context
                )
                if gene:
                    genes.append(gene)

        if genes:
            population.append(create_individual(genes))

    return population


def _create_pure_random_individual_wrapper(args):
    """Wrapper for parallel pure random individual creation."""
    individual_idx, course_group_pairs, silent = args

    try:
        worker_data = get_worker_context()
        context = worker_data["context"]
    except RuntimeError:
        return None

    genes = []
    for course_id, group_ids, _session_type, _num_quanta in course_group_pairs:
        course = context.courses.get(course_id)
        if not course:
            continue

        subsession_durations = get_subsession_durations(
            course.quanta_per_week, course.course_type
        )

        for subsession_duration in subsession_durations:
            gene = _create_pure_random_gene(
                course_id, group_ids, subsession_duration, course, context
            )
            if gene:
                genes.append(gene)

    if genes:
        return create_individual(genes)
    return None


def _create_pure_random_gene(course_id, group_ids, num_quanta, course, context):
    """Create gene with completely random assignment (no conflict avoidance)."""
    from src.ga.sessiongene import SessionGene

    # course_id is tuple (course_code, course_type)
    course_code, course_type = course_id

    # Random instructor from qualified instructors
    qualified_instructors = [
        instr_id
        for instr_id, instr in context.instructors.items()
        if course_id in instr.qualified_courses  # Match full tuple
    ]
    if not qualified_instructors:
        return None

    instructor_id = random.choice(qualified_instructors)

    # Random room (ignore suitability - will be caught by constraints)
    if not context.rooms:
        return None
    room_id = random.choice(list(context.rooms.keys()))

    # Random start time (ensure enough space for duration)
    total_quanta = len(context.available_quanta) if context.available_quanta else 42
    max_start = total_quanta - num_quanta
    if max_start < 0:
        return None
    start_quanta = random.randint(0, max_start)

    return SessionGene(
        course_id=course_code,  # Just the code, not the tuple
        course_type=course_type,  # Separate field
        group_ids=list(group_ids),  # SessionGene expects List[str]
        instructor_id=instructor_id,
        room_id=room_id,
        start_quanta=start_quanta,
        num_quanta=num_quanta,
    )


def _create_single_individual_wrapper(args):
    """
    Wrapper function for parallel individual creation.

    Args:
        args: Tuple of (individual_idx, course_group_pairs, silent)
        Note: context is retrieved from worker global state to avoid pickling.

    Returns:
        Individual (list of SessionGenes) or None if creation failed
    """
    individual_idx, course_group_pairs, silent = args

    # Get context from worker global state
    try:
        worker_data = get_worker_context()
        context = worker_data["context"]
    except RuntimeError:
        # Fallback if not running in worker (should not happen in parallel mode)
        # But if someone calls this directly...
        return None

    genes = []
    used_quanta = set()
    instructor_schedule = {}
    group_schedule = {}

    # Create genes for this individual
    for course_id, group_ids in course_group_pairs:
        course = context.courses.get(course_id)
        if not course:
            continue

        session_type = course.course_type
        num_quanta = course.quanta_per_week

        # Create ONE gene per (course, group) pair with full quanta_per_week
        # Even if initialization can't find perfect consecutive slots,
        # repair operators will fix conflicts during evolution
        session_gene = create_session_gene_with_conflict_avoidance(
            course_id,
            group_ids,
            session_type,
            num_quanta,
            course,
            context,
            used_quanta,
            instructor_schedule,
            group_schedule,
        )
        # CRITICAL: Always add gene even if it has conflicts
        # Skipping genes causes course_completeness violations
        if session_gene is not None:
            genes.append(session_gene)

    if genes:
        return create_individual(genes)
    else:
        if not silent:
            print(f"Warning: Individual {individual_idx+1} has no genes!")
        return None


def generate_course_group_aware_population(
    n: int,
    context: SchedulingContext,
    parallel: bool = True,  # NEW: Enable/disable parallelization
) -> list:
    """
    Generate population using simple course-group enrollment structure.

    PARALLELIZED: Individuals are generated concurrently using ProcessPoolExecutor.
    Expected speedup: 3-6x on multi-core systems.

    NEW ARCHITECTURE: No parent groups!
    - Each group is independent (subgroups are just regular groups)
    - Each (course, group) pair gets ONE session gene
    - Simple iteration: for each group, create genes for all enrolled courses

    This ensures no duplicate genes.

    [!]  CRITICAL: GENE ORDERING GUARANTEE

    This function creates genes in DETERMINISTIC ORDER for all individuals:
    1. Iterates through context["groups"] (dict maintains insertion order in Python 3.7+)
    2. For each group, iterates through enrolled_courses
    3. Creates exactly ONE gene per (course, group) pair

    This deterministic ordering ensures ALL individuals have genes at the SAME
    POSITIONS representing the SAME (course, group) pairs. This is required by
    the position-independent crossover operator (crossover_course_group_aware),
    which validates this structure and matches genes by (course_id, group_ids)
    identity rather than relying on position.

    MUTATION REQUIREMENT: Mutation operators MUST NOT reorder genes or change
    (course_id, group_ids) values. See src/ga/operators/mutation.py for details.

    Args:
        n: Population size
        context: Dict containing courses, groups, instructors, rooms, available_quanta
        parallel: Use parallel individual generation (default True)

    Returns:
        List of individuals (each is a list of SessionGenes)
    """
    # Step 1: Analyze group hierarchy and generate proper course-group pairs
    # This respects parent-subgroup relationships:
    # - Theory: Lists all subgroups explicitly (e.g., ["BAE2A", "BAE2B"])
    # - Practical: Each subgroup separately (e.g., ["BAE2A"], then ["BAE2B"])
    hierarchy = analyze_group_hierarchy(context.groups)

    # Always silent=True since warnings already shown in input encoder table
    silent = True

    # Generate course-group pairs using the proper function
    # Returns: List[Tuple[course_key, group_ids, session_type, num_quanta]]
    pair_tuples = generate_course_group_pairs(
        context.courses, context.groups, hierarchy, silent=silent
    )

    # Convert to simpler format for gene creation
    # (course_key, group_ids, session_type, num_quanta) -> (course_key, group_ids)
    course_group_pairs = [
        (course_key, group_ids) for course_key, group_ids, _, _ in pair_tuples
    ]

    if not course_group_pairs:
        if not silent:
            print("Warning: No valid course-group pairs found!")
        return []

    if not silent:
        print(f"Found {len(course_group_pairs)} course-group pairs to schedule")

    # Determine parallelization strategy - USE ALL AVAILABLE CORES
    num_workers = get_cpu_count() if parallel else 1

    # For small populations or debugging, use sequential generation
    if n < 10 or not parallel or num_workers == 1:
        # SEQUENTIAL: Original behavior
        population = []
        for individual_idx in range(n):
            genes = []
            used_quanta: set[int] = set()
            instructor_schedule: dict[str, set[int]] = {}
            group_schedule: dict[str, set[int]] = {}

            for course_id, group_ids in course_group_pairs:
                course = context.courses.get(course_id)
                if not course:
                    continue

                session_type = course.course_type

                # Break into subsessions: theory → multiple 2-quanta blocks, practical → single session
                subsession_durations = get_subsession_durations(
                    course.quanta_per_week, session_type
                )

                # Create one SessionGene per subsession
                for _subsession_idx, num_quanta in enumerate(subsession_durations):
                    session_gene = create_session_gene_with_conflict_avoidance(
                        course_id[0] if isinstance(course_id, tuple) else course_id,
                        group_ids,
                        session_type,
                        num_quanta,
                        course,
                        context,
                        used_quanta,
                        instructor_schedule,
                        group_schedule,
                    )
                    if session_gene:
                        genes.append(session_gene)

            if genes:
                population.append(create_individual(genes))
            else:
                if not silent:
                    print(f"Warning: Individual {individual_idx+1} has no genes!")

    else:
        # PARALLEL: Generate individuals concurrently
        population = []

        # Get data_dir from context config or default
        data_dir = "data"
        if (
            context.config
            and hasattr(context.config, "io")
            and hasattr(context.config.io, "data_dir")
        ):
            data_dir = context.config.io.data_dir

        # Prepare tasks for parallel execution - DO NOT pass context
        tasks = [(idx, course_group_pairs, silent) for idx in range(n)]

        # Use initializer to load data in workers
        # This avoids pickling the large context object
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(data_dir, random.randint(0, 10000)),
        ) as executor:
            # Generate all individuals in parallel
            results = list(executor.map(_create_single_individual_wrapper, tasks))

        # Filter out None results (failed individuals)
        population = [ind for ind in results if ind is not None]

        if len(population) < n and not silent:
            print(
                f"Warning: Only generated {len(population)}/{n} individuals successfully"
            )

    if not silent:
        if population:
            print(
                f"Generated {len(population)} individuals with average {sum(len(ind) for ind in population)/len(population):.1f} genes each"
            )
        else:
            print("Warning: Failed to generate any individuals!")

    return population


def extract_course_group_relationships(
    context: SchedulingContext,
) -> list[tuple[str, str]]:
    """
    Extract valid course-group enrollment pairs from the context.

    IMPORTANT: When a course has both theory and practical components,
    we need to create genes for BOTH. Groups' enrolled_courses contains
    course_codes like "ENSH 252", but the courses dict is keyed by
    (course_code, course_type) tuples.

    Returns:
        List of (course_key, group_id) tuples where course_key is (course_code, course_type)
    """
    course_group_pairs = []

    for group_id, group in context.groups.items():
        # Get enrolled courses for this group (these are course_codes)
        enrolled_courses = getattr(group, "enrolled_courses", [])

        for course_code in enrolled_courses:
            # Check for both theory and practical versions
            # courses dict is keyed by (course_code, course_type) tuples
            theory_key = (course_code, "theory")
            practical_key = (course_code, "practical")

            if theory_key in context.courses:
                course_group_pairs.append((theory_key, group_id))

            if practical_key in context.courses:
                course_group_pairs.append((practical_key, group_id))

    return course_group_pairs  # type: ignore[return-value]


def create_course_component_sessions(
    course_id: str, group_id: str, course, context: SchedulingContext
) -> list[SessionGene]:
    """
    Create session genes for a course-group combination.

    Clean architecture: NO suffixes!
    - All courses have plain course_id (e.g., "ENME 103")
    - course_type attribute distinguishes "theory" vs "practical"
    - Dict keyed by (course_code, course_type) tuples

    Args:
        course_id: Can be either course_code string OR (course_code, course_type) tuple
        group_id: Group identifier
        course: Course object (already has correct quanta_per_week)
        context: GA context with resources

    Returns:
        List of SessionGene objects (typically 1 gene per course-group pair)
    """
    session_genes = []

    # Use the quanta_per_week from Course entity (already correctly set during loading)
    quanta_needed = course.quanta_per_week

    # Get component type from Course object (not from name parsing)
    component_type = course.course_type

    # Create a single session gene for this course-group combination
    gene = create_component_session(
        course_id,
        group_id,
        component_type,
        quanta_needed,
        context,
        require_special_room=(component_type == "practical"),
    )

    if gene:
        session_genes.append(gene)
    else:
        print(f"[!]  Failed to create gene for {course_id} with group {group_id}")

    return session_genes


def create_course_component_sessions_with_conflict_avoidance(
    course_id: str,
    group_id: str,
    course,
    context: SchedulingContext,
    used_quanta: set,
    instructor_schedule: dict,
    group_schedule: dict,
) -> list[SessionGene]:
    """
    Create session genes while avoiding time conflicts.

    Clean architecture: NO suffixes!
    - All courses have plain course_id (e.g., "ENME 103")
    - course_type attribute distinguishes "theory" vs "practical"

    Args:
        course_id: Can be course_code string OR (course_code, course_type) tuple
        group_id: Group identifier
        course: Course object (already has correct quanta_per_week)
        context: GA context with resources
        used_quanta: Set of already used time quanta for this individual
        instructor_schedule: Dict tracking instructor time assignments
        group_schedule: Dict tracking group time assignments

    Returns:
        List of SessionGene objects (typically 1 gene per course-group pair)
    """
    session_genes = []

    # Use the quanta_per_week from Course entity (already correctly set during loading)
    quanta_needed = course.quanta_per_week

    # Get component type from Course object (not from name parsing)
    component_type = course.course_type

    # Create a single session gene for this course-group combination
    gene = create_component_session_with_conflict_avoidance(
        course_id,
        group_id,
        component_type,
        quanta_needed,
        context,
        used_quanta,
        instructor_schedule,
        group_schedule,
        require_special_room=(component_type == "practical"),
    )

    if gene:
        session_genes.append(gene)
    else:
        print(
            f"[!]  Failed to create gene for {course_id} with group {group_id} (conflict-avoidance)"
        )

    return session_genes


def create_session_gene_with_conflict_avoidance(
    course_id: str,
    group_ids: list[str],
    session_type: str,
    num_quanta: int,
    course,
    context: SchedulingContext,
    used_quanta: set,
    instructor_schedule: dict,
    group_schedule: dict,
) -> SessionGene:
    """
    Create ONE session gene for a (course, groups) combination.

    Args:
        course_id: Course identifier
        group_ids: List of group IDs (can be single or multiple)
        session_type: "theory" or "practical"
        num_quanta: Number of quanta needed (from course.quanta_per_week)
        course: Course entity
        context: GA context
        used_quanta, instructor_schedule, group_schedule: Conflict tracking

    Returns:
        SessionGene or None if creation failed
    """
    # Find qualified instructors
    qualified_instructors = find_qualified_instructors(course_id, context)
    if not qualified_instructors:
        qualified_instructors = list(context.instructors.values())

    # CRITICAL: Never return None - always create gene with fallback
    if not qualified_instructors:
        import logging

        logging.warning(f"No instructors available for {course_id}, using placeholder")
        # Create a placeholder instructor to ensure gene is created
        from src.entities.instructor import Instructor

        placeholder = Instructor(
            instructor_id="PLACEHOLDER",
            name="Unassigned",
            qualified_courses=[],
        )
        qualified_instructors = [placeholder]

    instructor = random.choice(qualified_instructors)

    # Find suitable rooms
    is_practical = session_type == "practical"
    suitable_rooms = find_suitable_rooms(
        course, session_type, context, require_special_room=is_practical
    )
    if not suitable_rooms:
        suitable_rooms = list(context.rooms.values())

    # CRITICAL: Always create a gene even if no suitable rooms
    # Use any room as fallback - repair will fix room suitability later
    if not suitable_rooms:
        suitable_rooms = list(context.rooms.values())

    if not suitable_rooms:
        # Absolute fallback - use first room
        suitable_rooms = [list(context.rooms.values())[0]] if context.rooms else []

    if not suitable_rooms:
        # No rooms at all - create gene anyway with placeholder
        import logging

        logging.warning(f"No rooms available for {course_id}, creating gene anyway")

    room = random.choice(suitable_rooms) if suitable_rooms else None

    # Find available quanta that don't conflict
    available_quanta = [q for q in context.available_quanta if q not in used_quanta]

    if len(available_quanta) < num_quanta:
        # Fall back to all if not enough free - allow conflicts, repair will fix later
        available_quanta = list(context.available_quanta)

    # CRITICAL: NEVER reduce quanta_needed - must always equal course.quanta_per_week
    # Initial conflicts are acceptable; repair operators will resolve them during evolution
    quanta_needed = num_quanta

    # CRITICAL: Always assign some quanta, never return None
    if quanta_needed == 0:
        quanta_needed = 1  # Minimum 1 quantum

    # Assign time quanta
    assigned_quanta = assign_conflict_free_quanta(
        quanta_needed, available_quanta, used_quanta
    )

    # DEBUG: Log what happened
    import logging

    if assigned_quanta and len(assigned_quanta) != quanta_needed:
        print(
            f" BUG DETECTED: {course_id} {session_type}: assigned_quanta has {len(assigned_quanta)} but needed {quanta_needed}"
        )
        print(f"   assigned_quanta={assigned_quanta[:10]}...")
        logging.warning(
            f"{course_id}: assign_conflict_free_quanta returned {len(assigned_quanta)} but needed {quanta_needed}"
        )

    # CRITICAL: If assignment fails, wrap around or duplicate quanta as needed
    # Repair operators will fix overlaps later - maintaining correct num_quanta is priority
    if not assigned_quanta:
        print(
            f"️ WRAP-AROUND for {course_id} {session_type}: need {quanta_needed} from {len(context.available_quanta)} available"
        )
        logging.info(
            f"{course_id}: Using fallback (needed {quanta_needed}, available {len(context.available_quanta)})"
        )
        # Use random consecutive block from all available quanta
        if len(context.available_quanta) >= quanta_needed:
            start_idx = random.randint(0, len(context.available_quanta) - quanta_needed)
            assigned_quanta = context.available_quanta[
                start_idx : start_idx + quanta_needed
            ]
        else:
            # Wrap around if needed to get exactly quanta_needed
            assigned_quanta = []
            while len(assigned_quanta) < quanta_needed:
                assigned_quanta.extend(context.available_quanta)
            assigned_quanta = assigned_quanta[:quanta_needed]
        print(f"  ✓ Fallback gave {len(assigned_quanta)} quanta")

    # VERIFICATION: Ensure we got exactly quanta_needed
    if len(assigned_quanta) != quanta_needed:
        logging.error(
            f"BUG: Got {len(assigned_quanta)} quanta but needed {quanta_needed} for {course_id}"
        )

    # Update tracking structures
    used_quanta.update(assigned_quanta)
    instructor_id = instructor.instructor_id
    if instructor_id not in instructor_schedule:
        instructor_schedule[instructor_id] = set()
    instructor_schedule[instructor_id].update(assigned_quanta)

    # Update group schedules for ALL groups in this session
    for gid in group_ids:
        if gid not in group_schedule:
            group_schedule[gid] = set()
        group_schedule[gid].update(assigned_quanta)

    # Create session gene with multi-group support
    # Extract actual course_id from course object (plain code, no tuple)
    actual_course_id = (
        course.course_id
        if hasattr(course, "course_id")
        else (course_id[0] if isinstance(course_id, tuple) else course_id)
    )

    # Get course_type from course object
    actual_course_type = (
        course.course_type if hasattr(course, "course_type") else session_type
    )

    # Convert quanta list to contiguous representation
    from src.ga.quanta_converter import quanta_list_to_contiguous

    start_q, num_q = quanta_list_to_contiguous(assigned_quanta)

    # DEBUG: Verify num_q matches quanta_needed
    if num_q != quanta_needed:
        print(
            f" CONVERSION BUG: {course_id} {session_type}: quanta_list_to_contiguous gave num_q={num_q} but quanta_needed={quanta_needed}"
        )
        print(f"   assigned_quanta length={len(assigned_quanta)}")

    session_gene = SessionGene(
        course_id=actual_course_id,
        course_type=actual_course_type,
        instructor_id=instructor_id,
        group_ids=group_ids,  # Can be multiple groups
        room_id=room.room_id if room else "UNASSIGNED",
        start_quanta=start_q,
        num_quanta=num_q,
    )

    return session_gene


def create_component_session_with_conflict_avoidance(
    course_id: str,
    group_id: str,
    component_type: str,
    hours: int,
    context: SchedulingContext,
    used_quanta: set,
    instructor_schedule: dict,
    group_schedule: dict,
    require_special_room: bool = False,
) -> SessionGene:
    """
    Create a single session while avoiding instructor and group conflicts.

    DEPRECATED: Use create_session_gene_with_conflict_avoidance instead.
    Kept for backwards compatibility with old population generators.
    """
    course = context.courses.get((course_id, component_type)) or context.courses.get(course_id)  # type: ignore[arg-type, call-overload]
    if not course:
        return None  # type: ignore[return-value]

    # Find qualified instructors
    qualified_instructors = find_qualified_instructors(course_id, context)
    if not qualified_instructors:
        qualified_instructors = list(context.instructors.values())

    if not qualified_instructors:
        return None  # type: ignore[return-value]

    instructor = random.choice(qualified_instructors)

    # Find suitable rooms
    suitable_rooms = find_suitable_rooms(
        course, component_type, context, require_special_room
    )
    if not suitable_rooms:
        suitable_rooms = list(context.rooms.values())

    if not suitable_rooms:
        return None  # type: ignore[return-value]

    room = random.choice(suitable_rooms)

    # FIXED: Parameter 'hours' is actually quanta_needed (course.quanta_per_week)
    # Previous bug: treated quanta as hours, multiplied by 4, then capped to 8
    # Now: use quanta_needed directly without multiplication or capping
    quanta_needed = hours  # Rename to clarify it's actually quanta, not hours

    # Find available quanta that don't conflict with used ones
    available_quanta = [q for q in context.available_quanta if q not in used_quanta]

    if len(available_quanta) < quanta_needed:
        # If not enough conflict-free quanta, use all (allow conflicts, repair will fix)
        available_quanta = list(context.available_quanta)

    # CRITICAL: NEVER reduce quanta_needed - must equal course.quanta_per_week
    # (removed: quanta_needed = min(quanta_needed, len(available_quanta)))

    if quanta_needed == 0:
        return None  # type: ignore[return-value]

    # Assign time quanta
    assigned_quanta = assign_conflict_free_quanta(
        quanta_needed, available_quanta, used_quanta
    )

    # FIXED: Add wrap-around fallback if assignment fails
    if not assigned_quanta:
        if len(context.available_quanta) >= quanta_needed:
            # Use module-level random import
            start_idx = random.randint(0, len(context.available_quanta) - quanta_needed)
            assigned_quanta = context.available_quanta[
                start_idx : start_idx + quanta_needed
            ]
        else:
            # Wrap around to get exactly quanta_needed
            assigned_quanta = []
            while len(assigned_quanta) < quanta_needed:
                assigned_quanta.extend(context.available_quanta)
            assigned_quanta = assigned_quanta[:quanta_needed]

    # Update tracking structures
    used_quanta.update(assigned_quanta)
    instructor_id = instructor.instructor_id
    if instructor_id not in instructor_schedule:
        instructor_schedule[instructor_id] = set()
    instructor_schedule[instructor_id].update(assigned_quanta)

    if group_id not in group_schedule:
        group_schedule[group_id] = set()
    group_schedule[group_id].update(assigned_quanta)

    # Create session gene
    # Extract course_id and course_type
    actual_course_id = course_id[0] if isinstance(course_id, tuple) else course_id
    actual_course_type = component_type

    # Convert quanta list to contiguous representation
    from src.ga.quanta_converter import quanta_list_to_contiguous

    start_q, num_q = quanta_list_to_contiguous(assigned_quanta)

    session_gene = SessionGene(
        course_id=actual_course_id,
        course_type=actual_course_type,
        instructor_id=instructor_id,
        group_ids=[group_id],  # Changed to list for multi-group support
        room_id=room.room_id,
        start_quanta=start_q,
        num_quanta=num_q,
    )

    return session_gene


def assign_conflict_free_quanta(
    quanta_needed: int, available_quanta: list, used_quanta: set
) -> list:
    """
    Assign time quanta while avoiding conflicts with already used quanta.

    CRITICAL: SessionGene enforces temporal contiguity (start_quanta + num_quanta).
    This function MUST return consecutive blocks of EXACTLY quanta_needed.

    Non-contiguous assignments or partial allocations cause course_completeness violations.

    Strategy:
    1. Try to find a consecutive block of exact size in free quanta
    2. If not found, try in ALL available quanta (allow conflicts for init)
    3. Return None if impossible (gene creation will fail)

    Args:
        quanta_needed: Number of consecutive quanta required
        available_quanta: All potentially available time quanta
        used_quanta: Quanta already occupied by other sessions

    Returns:
        List of CONSECUTIVE quanta of EXACTLY quanta_needed length, or None
    """
    if quanta_needed <= 0:
        return []

    # Filter out already used quanta
    used_set = set(used_quanta)
    free_quanta = [q for q in available_quanta if q not in used_set]

    # First attempt: Find consecutive block in free quanta (conflict-free)
    consecutive_block = _find_consecutive_block(free_quanta, quanta_needed)
    if consecutive_block:
        if len(consecutive_block) != quanta_needed:
            print(
                f" BUG: _find_consecutive_block returned {len(consecutive_block)} but needed {quanta_needed}"
            )
        return consecutive_block

    # Second attempt: Try in ALL available quanta (may have conflicts)
    # Repair operators will fix conflicts later
    consecutive_block = _find_consecutive_block(available_quanta, quanta_needed)
    if consecutive_block:
        if len(consecutive_block) != quanta_needed:
            print(
                f" BUG: _find_consecutive_block (all quanta) returned {len(consecutive_block)} but needed {quanta_needed}"
            )
        return consecutive_block

    # Cannot satisfy requirement - fail gene creation
    return None  # type: ignore[return-value]


def _find_consecutive_block(free_quanta: list, block_size: int) -> list | None:  # type: ignore[type-arg]
    """
    Find a single consecutive block of specified size.
    Returns None if not found.
    """
    if len(free_quanta) < block_size:
        return None  # type: ignore[return-value]

    sorted_free = sorted(free_quanta)

    for i in range(len(sorted_free) - block_size + 1):
        candidates = sorted_free[i : i + block_size]

        # Check if truly consecutive
        is_consecutive = all(
            candidates[j] - candidates[j - 1] == 1 for j in range(1, len(candidates))
        )

        if is_consecutive:
            return candidates

    return None  # type: ignore[return-value]


def create_component_session(
    course_id: str,
    group_id: str,
    component_type: str,
    hours: int,
    context: SchedulingContext,
    require_special_room: bool = False,
) -> SessionGene:
    """
    Create a single session for a specific course component.

    Args:
        course_id: Course identifier
        group_id: Group identifier
        component_type: "lecture", "tutorial", "practical", or "default"
        hours: Number of hours per week for this component
        context: GA context
        require_special_room: Whether this component needs special room features

    Returns:
        SessionGene object for this component (or None if creation fails)
    """
    course = context.courses.get((course_id, component_type)) or context.courses.get(course_id)  # type: ignore[arg-type, call-overload]
    if not course:
        return None  # type: ignore[return-value]

    # Find qualified instructors
    qualified_instructors = find_qualified_instructors(course_id, context)
    if not qualified_instructors:
        # Fallback to any instructor if none qualified
        qualified_instructors = list(context.instructors.values())

    if not qualified_instructors:
        print(f"Warning: No instructors available for course {course_id}")
        return None  # type: ignore[return-value]

    instructor = random.choice(qualified_instructors)

    # Find suitable rooms
    suitable_rooms = find_suitable_rooms(
        course, component_type, context, require_special_room
    )
    if not suitable_rooms:
        # Fallback to any room if none suitable
        suitable_rooms = list(context.rooms.values())

    if not suitable_rooms:
        print(f"Warning: No rooms available for course {course_id}")
        return None  # type: ignore[return-value]

    room = random.choice(suitable_rooms)

    # FIXED: Parameter 'hours' is actually quanta_needed (course.quanta_per_week)
    # Previous bug: treated quanta as hours, multiplied by 4, then capped to 8
    # Now: use quanta_needed directly without multiplication or capping
    quanta_needed = hours  # Rename to clarify it's actually quanta, not hours

    # CRITICAL: NEVER reduce quanta_needed - must equal course.quanta_per_week
    # (removed all capping: max_session_length, len(available_quanta))

    # Assign time quanta with intelligence
    assigned_quanta = assign_intelligent_quanta(quanta_needed, context.available_quanta)

    # FIXED: Add wrap-around fallback if assignment fails
    if not assigned_quanta:
        if len(context.available_quanta) >= quanta_needed:
            # Use module-level random import
            start_idx = random.randint(0, len(context.available_quanta) - quanta_needed)
            assigned_quanta = context.available_quanta[
                start_idx : start_idx + quanta_needed
            ]
        else:
            # Wrap around to get exactly quanta_needed
            assigned_quanta = []
            while len(assigned_quanta) < quanta_needed:
                assigned_quanta.extend(context.available_quanta)
            assigned_quanta = assigned_quanta[:quanta_needed]

    # Create session gene
    # Extract course_id and course_type
    actual_course_id = course_id[0] if isinstance(course_id, tuple) else course_id
    actual_course_type = component_type

    # Convert quanta list to contiguous representation
    from src.ga.quanta_converter import quanta_list_to_contiguous

    start_q, num_q = quanta_list_to_contiguous(assigned_quanta)

    session_gene = SessionGene(
        course_id=actual_course_id,
        course_type=actual_course_type,
        instructor_id=instructor.instructor_id,
        group_ids=[group_id],  # Changed to list for multi-group support
        room_id=room.room_id,
        start_quanta=start_q,
        num_quanta=num_q,
    )

    return session_gene


def find_qualified_instructors(course_id, context: SchedulingContext) -> list:
    """
    Find instructors qualified to teach this course.

    Args:
        course_id: Can be plain string or (course_code, course_type) tuple
    """
    qualified = []

    for instructor in context.instructors.values():
        # Check if instructor is qualified for this course
        # instructor.qualified_courses now contains tuples (course_code, course_type)
        qualified_courses = getattr(instructor, "qualified_courses", [])
        if course_id in qualified_courses:
            qualified.append(instructor)

    return qualified


def find_suitable_rooms(
    course,
    component_type: str,
    context: SchedulingContext,
    require_special_room: bool = False,
) -> list:
    """
    Find rooms suitable for this course component with intelligent prioritization.

    Strategy:
    1. Exact match: room.room_features matches course.required_room_features
    2. Flexible match: compatible room types (using Room.is_suitable_for_course_type)
    3. Capacity check: room must accommodate largest enrolled group

    Returns rooms in priority order: exact matches first, then flexible matches.
    """
    exact_matches = []
    flexible_matches = []

    # Get required room features from course
    required_features = getattr(course, "required_room_features", "classroom")
    course_id = getattr(course, "course_id", "")

    # Find the group size for capacity matching
    max_group_size = 0  # Start at 0, will use min capacity if no groups found
    for group in context.groups.values():
        if course_id in getattr(group, "enrolled_courses", []):
            group_size = getattr(group, "student_count", 30)
            max_group_size = max(max_group_size, group_size)

    # If no enrolled groups found, use a minimal default (don't filter out small rooms)
    if max_group_size == 0:
        max_group_size = 1  # Accept any room with capacity >= 1

    # Evaluate each room
    for room in context.rooms.values():
        room_capacity = getattr(room, "capacity", 50)

        # Check capacity requirement (hard constraint)
        if room_capacity < max_group_size:
            continue

        room_features = getattr(room, "room_features", "classroom")

        # Handle required_features as string (normalized during data loading)
        required_str = (
            (
                required_features
                if isinstance(required_features, str)
                else str(required_features)
            )
            .lower()
            .strip()
        )

        room_str = (
            (room_features if isinstance(room_features, str) else str(room_features))
            .lower()
            .strip()
        )

        # PRIORITY 1: Exact match
        if room_str == required_str:
            exact_matches.append(room)
            continue

        # PRIORITY 2: Flexible match using Room's built-in method
        if hasattr(
            room, "is_suitable_for_course_type"
        ) and room.is_suitable_for_course_type(required_str):
            flexible_matches.append(room)
            continue

        # PRIORITY 3: Fallback compatibility rules
        # Lab courses can use any lab variant
        if required_str in ["lab", "laboratory"] and any(
            lab_type in room_str for lab_type in ["lab", "computer", "science"]
        ):
            flexible_matches.append(room)
            continue

        # Lecture/theory courses have flexibility
        if required_str in ["lecture", "classroom", "theory"] and room_str in [
            "lecture",
            "classroom",
            "auditorium",
            "seminar",
        ]:
            flexible_matches.append(room)
            continue

    # Return prioritized list: exact matches first, then flexible
    result = exact_matches + flexible_matches

    # If no suitable rooms found, fallback to capacity-only check
    # BUT still try to match room type if possible
    if not result:
        # Last resort: any room with adequate capacity
        # But prefer rooms that match the course type
        fallback_exact = []
        fallback_any = []

        for r in context.rooms.values():
            if getattr(r, "capacity", 50) >= max_group_size:
                room_str = getattr(r, "room_features", "").lower().strip()
                # Try to match course_type at least
                if (component_type == "practical" and "practical" in room_str) or (
                    component_type in ["theory", "lecture"] and "lecture" in room_str
                ):
                    fallback_exact.append(r)
                else:
                    fallback_any.append(r)

        result = fallback_exact + fallback_any

    return result


def assign_intelligent_quanta(quanta_needed: int, available_quanta: list) -> list:
    """
    Assign time quanta with clustering intelligence to minimize fragmentation.
    CLUSTER-AWARE: Uses same logic as assign_conflict_free_quanta.
    """
    if quanta_needed <= 0:
        return []

    available_list = list(available_quanta)

    # CRITICAL: NEVER reduce quanta_needed - must equal course.quanta_per_week
    # (removed: if quanta_needed > len(available_list): quanta_needed = len(available_list))
    # assign_conflict_free_quanta will return None if it can't find consecutive blocks
    # Then caller's wrap-around fallback will handle it

    if quanta_needed == 0:
        return []

    # Reuse cluster-aware assignment with empty used_quanta set
    # This ensures consistent clustering behavior across all initialization paths
    return assign_conflict_free_quanta(quanta_needed, available_list, set())


def generate_random_gene(
    possible_courses,
    possible_instructors,
    possible_groups,
    possible_rooms,
    available_quanta,
) -> SessionGene:
    """Legacy function for backward compatibility."""
    course = random.choice(possible_courses)

    # Try to find a qualified instructor (constraint-aware)
    qualified_instructors = [
        inst
        for inst in possible_instructors
        if course.course_id in getattr(inst, "qualified_courses", [course.course_id])
    ]
    instructor = random.choice(
        qualified_instructors if qualified_instructors else possible_instructors
    )

    # Try to find a compatible group (constraint-aware)
    compatible_groups = [
        grp
        for grp in possible_groups
        if course.course_id in getattr(grp, "enrolled_courses", [course.course_id])
    ]
    group = random.choice(compatible_groups if compatible_groups else possible_groups)

    room = random.choice(possible_rooms)

    # Use course's required quanta count if available
    num_quanta = getattr(course, "quanta_per_week", random.randint(1, 4))
    num_quanta = min(
        num_quanta, len(available_quanta)
    )  # Ensure we don't exceed available

    # You may want to base this on course.quanta_per_week
    quanta = random.sample(list(available_quanta), num_quanta)

    # Convert quanta list to contiguous representation
    from src.ga.quanta_converter import quanta_list_to_contiguous

    start_q, num_q = quanta_list_to_contiguous(quanta)

    return SessionGene(
        course_id=course.course_id,
        course_type=getattr(course, "course_type", "theory"),  # Default to theory
        instructor_id=instructor.instructor_id,
        group_ids=[group.group_id],  # Changed to list for multi-group support
        room_id=room.room_id,
        start_quanta=start_q,
        num_quanta=num_q,
    )


def generate_population(
    n: int, session_count: int | None = None, context: SchedulingContext | None = None
) -> list:
    """
    Wrapper function for backward compatibility.
    Now uses course-group aware generation instead of random.
    """
    if context is None:
        raise ValueError("Context must be provided for population generation")

    return generate_course_group_aware_population(n, context)
