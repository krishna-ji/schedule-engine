"""Population cache — serialisation and deserialisation of GA populations."""

from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from src.domain.gene import SessionGene
from src.ga.core.domain_store import GeneDomainStore
from src.ga.core.usage_tracker import UsageTracker
from src.utils.console_service import get_console
from src.utils.parallel_worker import get_worker_context, init_worker
from src.utils.system_info import get_cpu_count

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.group import Group
    from src.domain.instructor import Instructor
    from src.domain.room import Room
    from src.domain.types import Individual, SchedulingContext

type CourseKey = tuple[str, str]
type DetailedPair = tuple[CourseKey, list[str], str, int]
type CourseGroupPair = tuple[CourseKey, list[str]]
type ScheduleMap = dict[str, set[int]]
type ResourceUsage = dict[tuple[str, int], bool]
type HierarchyMap = dict[str, list[str] | dict[str, list[str]] | dict[str, str]]


def analyze_group_hierarchy_from_json(json_path: str) -> HierarchyMap:
    """
    Analyze group hierarchy by reading explicit subgroups from Groups.json.

    The JSON structure has parent groups with a "subgroups" array, e.g.:
        {"group_id": "BME1AB", "subgroups": [{"id": "BME1A"}, {"id": "BME1B"}]}
    """
    with Path(json_path).open() as handle:
        raw_data = json.load(handle)

    parents: set[str] = set()
    subgroups_dict: dict[str, list[str]] = {}
    parent_map: dict[str, str] = {}
    all_group_ids: set[str] = set()

    for item in raw_data:
        group_id = item.get("group_id", "")
        if group_id:
            all_group_ids.add(group_id)

        subgroups_raw = item.get("subgroups")
        if subgroups_raw and len(subgroups_raw) >= 1:
            parent_id = group_id
            parents.add(parent_id)

            subgroup_ids = _extract_subgroup_ids(subgroups_raw)
            if subgroup_ids:
                subgroups_dict[parent_id] = subgroup_ids
                for sg_id in subgroup_ids:
                    parent_map[sg_id] = parent_id
                    all_group_ids.add(sg_id)

    all_subgroups = set(parent_map.keys())
    standalone = sorted(all_group_ids - parents - all_subgroups)

    return {
        "parents": sorted(parents),
        "subgroups": subgroups_dict,
        "parent_map": parent_map,
        "standalone": standalone,
    }


def _extract_subgroup_ids(subgroups: list[Any]) -> list[str]:
    """Normalize subgroup entries to clean string identifiers."""
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_entry in subgroups:
        subgroup_id: str | None
        if isinstance(raw_entry, dict):
            subgroup_id = raw_entry.get("id")
        else:
            subgroup_id = str(raw_entry)

        if subgroup_id is None:
            continue

        clean_id = subgroup_id.strip()
        if not clean_id:
            continue

        canonical = clean_id.lower()
        if canonical in seen:
            continue

        seen.add(canonical)
        normalized.append(clean_id)

    return normalized


def analyze_group_hierarchy(groups: dict[str, Group]) -> HierarchyMap:
    """
    DEPRECATED: Use analyze_group_hierarchy_from_json instead.

    Fallback uses pattern matching which may not work for all naming conventions.
    """
    parents: set[str] = set()
    subgroups_dict: dict[str, list[str]] = {}
    parent_map: dict[str, str] = {}
    all_group_ids = set(groups.keys())

    for group_id in all_group_ids:
        if len(group_id) > 1 and group_id[-1].isalpha():
            potential_parent = group_id[:-1]

            if potential_parent in all_group_ids:
                parents.add(potential_parent)
                parent_map[group_id] = potential_parent

                if potential_parent not in subgroups_dict:
                    subgroups_dict[potential_parent] = []
                subgroups_dict[potential_parent].append(group_id)

    parents_list = sorted(parents)
    all_subgroups = set(parent_map.keys())
    standalone = sorted(all_group_ids - parents - all_subgroups)

    return {
        "parents": parents_list,
        "subgroups": subgroups_dict,
        "parent_map": parent_map,
        "standalone": standalone,
    }


def is_parent_group(group_id: str, hierarchy: HierarchyMap) -> bool:
    """Check if a group is a parent group."""
    return group_id in hierarchy["parents"]


def is_subgroup(group_id: str, hierarchy: HierarchyMap) -> bool:
    """Check if a group is a subgroup."""
    return group_id in hierarchy["parent_map"]


def get_parent(group_id: str, hierarchy: HierarchyMap) -> str:
    """Get parent group ID for a subgroup."""
    parent_map: dict[str, str] = hierarchy["parent_map"]  # type: ignore[assignment]
    result = parent_map.get(group_id)
    return str(result) if result is not None else ""


def get_subgroups(parent_id: str, hierarchy: HierarchyMap) -> list[str]:
    """Get list of subgroup IDs for a parent."""
    subgroups: dict[str, list[str]] = hierarchy["subgroups"]  # type: ignore[assignment]
    result: list[str] = subgroups.get(parent_id, [])
    return result


def has_subgroups(group_id: str, hierarchy: HierarchyMap) -> bool:
    """Check if a group has subgroups."""
    return group_id in hierarchy["subgroups"]


def get_sibling_groups(group_id: str, hierarchy: HierarchyMap) -> list[str]:
    """
    Get sibling groups (other subgroups of the same parent).
    """
    parent_id = get_parent(group_id, hierarchy)
    if not parent_id:
        return []

    all_subgroups = get_subgroups(parent_id, hierarchy)
    return [g for g in all_subgroups if g != group_id]


def get_all_related_groups(group_id: str, hierarchy: HierarchyMap) -> set[str]:
    """
    Get all groups related to this one (parent, siblings, and self).
    """
    related = {group_id}

    parent_id = get_parent(group_id, hierarchy)
    if parent_id:
        related.add(parent_id)
        subgroups: dict[str, list[str]] = hierarchy["subgroups"]  # type: ignore[assignment]
        related.update(subgroups.get(parent_id, []))

    if has_subgroups(group_id, hierarchy):
        subgroups_dict: dict[str, list[str]] = hierarchy["subgroups"]  # type: ignore[assignment]
        related.update(subgroups_dict.get(group_id, []))

    return related


def build_group_family_map(hierarchy: HierarchyMap) -> dict[str, set[str]]:
    """Pre-compute a map from each group_id to all related groups."""
    all_groups: set[str] = set()

    parents: list[str] = hierarchy["parents"]  # type: ignore[assignment]
    parent_map: dict[str, str] = hierarchy["parent_map"]  # type: ignore[assignment]
    standalone: list[str] = hierarchy["standalone"]  # type: ignore[assignment]

    all_groups.update(parents)
    all_groups.update(parent_map.keys())
    all_groups.update(standalone)

    family_map: dict[str, set[str]] = {}

    for group_id in all_groups:
        family_map[group_id] = get_all_related_groups(group_id, hierarchy)

    return family_map


def groups_conflict(
    group_ids_a: list[str],
    group_ids_b: list[str],
    family_map: dict[str, set[str]],
) -> bool:
    """Check if two sets of groups have any family overlap (conflict)."""
    a_family: set[str] = set()
    for gid in group_ids_a:
        a_family.update(family_map.get(gid, {gid}))

    for gid in group_ids_b:
        b_family = family_map.get(gid, {gid})
        if a_family & b_family:
            return True

    return False


_cached_hierarchy: HierarchyMap | None = None
_cached_family_map: dict[str, set[str]] | None = None
_cached_json_path: str | None = None


def get_hierarchy_from_json(json_path: str = "data/Groups.json") -> HierarchyMap:
    """Get the group hierarchy, loading from JSON and caching the result."""
    global _cached_hierarchy, _cached_json_path

    if _cached_hierarchy is None or _cached_json_path != json_path:
        _cached_hierarchy = analyze_group_hierarchy_from_json(json_path)
        _cached_json_path = json_path

    return _cached_hierarchy


def _get_family_map_from_json(
    json_path: str = "data/Groups.json",
) -> dict[str, set[str]]:
    """Get the pre-computed family map, loading from JSON and caching."""
    global _cached_family_map

    hierarchy = get_hierarchy_from_json(json_path)

    if _cached_family_map is None or _cached_json_path != json_path:
        _cached_family_map = build_group_family_map(hierarchy)

    return _cached_family_map


def get_family_map_from_json(
    json_path: str = "data/Groups.json",
) -> dict[str, set[str]]:
    """Backward-compatible alias for group family map loader."""
    return _get_family_map_from_json(json_path)


def clear_hierarchy_cache() -> None:
    """Clear the cached hierarchy data (useful for testing or data reload)."""
    global _cached_hierarchy, _cached_family_map, _cached_json_path
    _cached_hierarchy = None
    _cached_family_map = None
    _cached_json_path = None


def generate_course_group_pairs(
    courses: dict[tuple[str, str], Course],
    groups: dict[str, Group],
    hierarchy: HierarchyMap,
    silent: bool = False,
) -> list[tuple[tuple[str, str], list[str], str, int]]:
    """
    Generate (course_id, group_ids, session_type, num_quanta) tuples.
    """
    pairs: list[tuple[tuple[str, str], list[str], str, int]] = []

    from collections import defaultdict

    parent_to_subgroups = defaultdict(list)

    for group_id in groups:
        if len(group_id) > 1 and group_id[-1].isalpha():
            parent_prefix = group_id[:-1]
            parent_to_subgroups[parent_prefix].append(group_id)
        elif group_id not in parent_to_subgroups:
            parent_to_subgroups[group_id] = [group_id]

    for parent_prefix, sibling_ids in parent_to_subgroups.items():
        first_sibling = groups[sibling_ids[0]]
        enrolled_courses = first_sibling.enrolled_courses

        for course_code in enrolled_courses:
            theory_key = (course_code, "theory")
            practical_key = (course_code, "practical")

            matching_courses = []
            if theory_key in courses:
                matching_courses.append((theory_key, courses[theory_key]))
            if practical_key in courses:
                matching_courses.append((practical_key, courses[practical_key]))

            if not matching_courses:
                if not silent:
                    logger.warning(
                        "Course %s not found for group %s", course_code, parent_prefix
                    )
                continue

            for course_key, course in matching_courses:
                if course.course_type == "theory":
                    theory_quanta = course.quanta_per_week
                    pairs.append(
                        (course_key, sorted(sibling_ids), "theory", theory_quanta)
                    )

                elif course.course_type == "practical":
                    practical_quanta = course.quanta_per_week
                    pairs.extend(
                        (course_key, [sibling_id], "practical", practical_quanta)
                        for sibling_id in sibling_ids
                    )

    return pairs


def count_total_genes(pairs: list[tuple[tuple[str, str], list[str], str, int]]) -> int:
    """Count total number of genes that will be created."""
    return sum(num_quanta for _, _, _, num_quanta in pairs)


def group_pairs_by_course(
    pairs: list[tuple[tuple[str, str], list[str], str, int]],
) -> dict[tuple[str, str], list[tuple[tuple[str, str], list[str], str, int]]]:
    """Group pairs by course for analysis."""
    from collections import defaultdict

    course_pairs = defaultdict(list)
    for pair in pairs:
        course_key = pair[0]
        course_pairs[course_key].append(pair)
    return dict(course_pairs)


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
    # Theory: Break into 2-quanta blocks with remainder
    if quanta_per_week % 2 == 0:
        # Even: All 2-quanta blocks (e.g., 6 → [2,2,2])
        return [2] * (quanta_per_week // 2)
    # Odd: 2-quanta blocks + 1-quanta remainder (e.g., 5 → [2,2,1])
    blocks = [2] * (quanta_per_week // 2)
    blocks.append(1)
    return blocks


console = get_console()


def generate_pure_random_population(
    n: int,
    context: SchedulingContext,
    parallel: bool = True,
) -> list[Individual]:
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
            population.append(genes)

    return population


def _create_pure_random_individual_wrapper(
    args: tuple[int, list[DetailedPair], bool],
) -> Individual | None:
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
        return genes
    return None


def _create_pure_random_gene(
    course_id: CourseKey,
    group_ids: list[str],
    num_quanta: int,
    course: Course,
    context: SchedulingContext,
) -> SessionGene | None:
    """Create gene with completely random assignment (no conflict avoidance)."""
    from src.domain.gene import SessionGene

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

    # Room: suitability-aware selection (type + capacity + features)
    from src.ga.operators.mutation import find_suitable_rooms_for_course

    primary_group = group_ids[0] if group_ids else ""
    suitable_rooms = find_suitable_rooms_for_course(
        course_code, course_type, primary_group, context
    )
    if not suitable_rooms:
        return None
    room_id = random.choice(suitable_rooms)

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


def _create_single_individual_wrapper(
    args: tuple[int, list[CourseGroupPair], bool],
) -> Individual | None:
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
    used_quanta: set[int] = set()
    instructor_schedule: ScheduleMap = {}
    group_schedule: ScheduleMap = {}

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
        _assign_practical_co_instructors(genes, context)
        return genes
    if not silent:
        logger.warning("Individual %s has no genes!", individual_idx + 1)
    return None


def generate_course_group_aware_population(
    n: int,
    context: SchedulingContext,
    parallel: bool = True,  # NEW: Enable/disable parallelization
) -> list[Individual]:
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
            logger.warning("No valid course-group pairs found!")
        return []

    if not silent:
        logger.info("Found %s course-group pairs to schedule", len(course_group_pairs))

    # Determine parallelization strategy - USE ALL AVAILABLE CORES
    num_workers = get_cpu_count() if parallel else 1

    # For small populations or debugging, use sequential generation
    if n < 10 or not parallel or num_workers == 1:
        # SEQUENTIAL: Spreading-aware initialization
        population = []
        for individual_idx in range(n):
            ind = _create_individual_with_spreading(
                course_group_pairs, context, silent=silent
            )
            if ind is not None:
                population.append(ind)
            elif not silent:
                logger.warning("Individual %s has no genes!", individual_idx + 1)

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
            logger.warning(
                "Only generated %s/%s individuals successfully", len(population), n
            )

    if not silent:
        if population:
            logger.info(
                "Generated %s individuals with average %.1f genes each",
                len(population),
                sum(len(ind) for ind in population) / len(population),
            )
        else:
            logger.warning("Failed to generate any individuals!")

    return population


def _create_individual_with_spreading(
    course_group_pairs: list[CourseGroupPair],
    context: SchedulingContext,
    *,
    silent: bool = True,
) -> Individual | None:
    """Create one individual using usage-aware spreading.

    For each gene we:
    1. Pick the LEAST-LOADED qualified instructor
    2. Pick a time-start where groups + instructor are FREE and time-load is lowest
    3. Pick a room that is FREE at the chosen time and least-loaded overall

    This replaces the old ``random.choice()`` pattern that caused clustering.
    """
    from src.domain.gene import SessionGene, get_time_system

    qts = get_time_system()
    domain_store = GeneDomainStore(context, qts)
    tracker = UsageTracker()

    # Pre-compute day boundaries for valid-start generation
    # (reuse the store's internal data)

    genes: list[SessionGene] = []
    gene_idx = 0

    for course_id, group_ids in course_group_pairs:
        course = context.courses.get(course_id)
        if not course:
            continue

        session_type = course.course_type
        subsession_durations = get_subsession_durations(
            course.quanta_per_week, session_type
        )

        for num_quanta in subsession_durations:
            # Build a temporary "phantom" gene so we can compute its domain
            # We'll replace inst/room/time below
            phantom = SessionGene(
                course_id=course_id[0] if isinstance(course_id, tuple) else course_id,
                course_type=(
                    course_id[1] if isinstance(course_id, tuple) else session_type
                ),
                instructor_id=next(iter(context.instructors.keys())),
                group_ids=list(group_ids),
                room_id=next(iter(context.rooms.keys())),
                start_quanta=0,
                num_quanta=num_quanta,
            )
            domain_store.refresh_domain(gene_idx, phantom)
            domain = domain_store.get_domain(gene_idx)

            # --- INSTRUCTOR: least-loaded qualified ---
            instructor_id = tracker.pick_least_used_instructor(domain.instructors)

            # --- TIME: cascading narrowing with graceful fallback ---
            # Priority: instructor-available + group-free + inst-free > group-free + inst-free > group-free > avail > any
            avail_starts = domain_store.instructor_available_starts(
                gene_idx, instructor_id
            )
            all_starts = domain.valid_starts

            # Try ideal: avail ∩ group-free ∩ instructor-free
            pool = avail_starts if avail_starts else all_starts
            grp_free = tracker.group_free_starts(
                group_ids,
                pool,
                num_quanta,
                family_map=context.family_map or None,
            )
            candidates = (
                tracker.instructor_free_starts(
                    instructor_id,
                    grp_free if grp_free else pool,
                    num_quanta,
                )
                if grp_free
                else []
            )

            if not candidates:
                # Fallback 1: group-free from full domain (ignore inst availability)
                grp_free_full = tracker.group_free_starts(
                    group_ids,
                    all_starts,
                    num_quanta,
                    family_map=context.family_map or None,
                )
                candidates = (
                    tracker.instructor_free_starts(
                        instructor_id,
                        grp_free_full if grp_free_full else all_starts,
                        num_quanta,
                    )
                    if grp_free_full
                    else []
                )

            if not candidates:
                # Fallback 2: just group-free (allow instructor double-book over avail conflict)
                candidates = (
                    grp_free_full
                    if grp_free_full
                    else (avail_starts if avail_starts else all_starts)
                )

            start_quanta = tracker.pick_least_used_start(
                candidates,
                num_quanta,
                top_k=5,
            )
            if start_quanta is None:
                # Final fallback: random from available
                if context.available_quanta:
                    max_s = len(context.available_quanta) - num_quanta
                    start_quanta = context.available_quanta[
                        random.randint(0, max(0, max_s))
                    ]
                else:
                    start_quanta = 0

            # --- ROOM: free at chosen time, then least-loaded ---
            room_free = [
                r
                for r in domain.rooms
                if all(
                    tracker.room_load.get(r, Counter()).get(q, 0) == 0
                    for q in range(start_quanta, start_quanta + num_quanta)
                )
            ]
            room_pool = room_free if room_free else domain.rooms
            room_id = tracker.pick_least_used_room(
                room_pool if room_pool else list(context.rooms.keys()),
                start_quanta,
                num_quanta,
            )

            actual_course_id = (
                course_id[0] if isinstance(course_id, tuple) else course_id
            )
            actual_course_type = (
                course_id[1] if isinstance(course_id, tuple) else session_type
            )

            gene = SessionGene(
                course_id=actual_course_id,
                course_type=actual_course_type,
                instructor_id=instructor_id,
                group_ids=list(group_ids),
                room_id=room_id,
                start_quanta=start_quanta,
                num_quanta=num_quanta,
            )
            genes.append(gene)
            tracker.add_gene(gene)
            gene_idx += 1

    if genes:
        _assign_practical_co_instructors(genes, context)
        return genes
    return None


def _assign_practical_co_instructors(
    genes: list[SessionGene],
    context: SchedulingContext,
) -> None:
    """Assign one co-instructor to every practical gene (in-place).

    Structural invariant: each practical session ALWAYS has exactly
    2 instructors (1 main + 1 co-instructor). This is not an optimisation
    objective — it is enforced at construction and preserved by every
    operator (mutation, crossover).
    """
    for gene in genes:
        if gene.course_type != "practical":
            continue
        course_key = (gene.course_id, gene.course_type)
        qualified = [
            iid
            for iid, inst in context.instructors.items()
            if course_key in getattr(inst, "qualified_courses", [])
            and iid != gene.instructor_id
        ]
        if qualified:
            gene.co_instructor_ids = [random.choice(qualified)]
        else:
            # Data guarantees enough instructors; pick any other as fallback
            others = [iid for iid in context.instructors if iid != gene.instructor_id]
            gene.co_instructor_ids = (
                [random.choice(others)] if others else [gene.instructor_id]
            )


def extract_course_group_relationships(
    context: SchedulingContext,
) -> list[tuple[CourseKey, str]]:
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

    return course_group_pairs


def create_course_component_sessions(
    course_id: str, group_id: str, course: Course, context: SchedulingContext
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
        logger.warning(
            "Failed to create gene for %s with group %s", course_id, group_id
        )

    return session_genes


def create_course_component_sessions_with_conflict_avoidance(
    course_id: str,
    group_id: str,
    course: Course,
    context: SchedulingContext,
    used_quanta: set[int],
    instructor_schedule: ScheduleMap,
    group_schedule: ScheduleMap,
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
        logger.warning(
            "Failed to create gene for %s with group %s (conflict-avoidance)",
            course_id,
            group_id,
        )

    return session_genes


def create_session_gene_with_conflict_avoidance(
    course_id: str | CourseKey,
    group_ids: list[str],
    session_type: str,
    num_quanta: int,
    course: Course,
    context: SchedulingContext,
    used_quanta: set[int],
    instructor_schedule: ScheduleMap,
    group_schedule: ScheduleMap,
) -> SessionGene:
    """
    Create ONE session gene for a (course, groups) combination.

    ENHANCED: Uses instructor availability as a HARD FILTER when possible.

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

    # ENHANCED: Try to find instructor-time pairs that respect availability
    # Build set of quanta already used by THIS individual's instructors
    used_by_instructors: set[int] = set()
    for inst_quanta in instructor_schedule.values():
        used_by_instructors.update(inst_quanta)

    # Build set of quanta used by groups in this session
    used_by_groups: set[int] = set()
    for gid in group_ids:
        if gid in group_schedule:
            used_by_groups.update(group_schedule[gid])

    # Find instructors with availability
    qualified_with_availability = find_qualified_instructors_with_availability(
        course_id,
        context,
        num_quanta,
        exclude_quanta=used_by_instructors | used_by_groups,
    )

    instructor = None
    assigned_quanta: list[int] = []

    if qualified_with_availability:
        # Pick randomly from instructors with most flexibility
        # Take top 3 by availability count and pick one randomly
        top_instructors = qualified_with_availability[:3]
        chosen_instructor, available_starts = random.choice(top_instructors)
        instructor = chosen_instructor

        # Pick a random valid start time from available slots
        if available_starts:
            start_q = random.choice(available_starts)
            assigned_quanta = list(range(start_q, start_q + num_quanta))
    # FALLBACK: Use original logic if availability-based selection fails
    if instructor is None:
        # Find qualified instructors (may not be available)
        qualified_instructors = find_qualified_instructors(course_id, context)
        if not qualified_instructors:
            qualified_instructors = list(context.instructors.values())

        # CRITICAL: Never return None - always create gene with fallback
        if not qualified_instructors:
            logger.warning(
                "No instructors available for %s, using placeholder", course_id
            )
            from src.domain.instructor import Instructor

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
    if not suitable_rooms:
        suitable_rooms = list(context.rooms.values())

    if not suitable_rooms and context.rooms:
        suitable_rooms = [next(iter(context.rooms.values()))]

    if not suitable_rooms:
        logger.warning("No rooms available for %s, creating gene anyway", course_id)

    room = random.choice(suitable_rooms) if suitable_rooms else None
    # If we didn't get assigned_quanta from availability check, use fallback
    quanta_needed = num_quanta if num_quanta > 0 else 1

    if not assigned_quanta:
        # FIX: Use PER-RESOURCE conflict tracking instead of global used_quanta.
        # A quantum is only blocked for THIS gene if the gene's groups or
        # instructor already have a session at that time.
        gene_blocked: set[int] = set()
        for gid in group_ids:
            gene_blocked.update(group_schedule.get(gid, set()))
        if instructor is not None:
            gene_blocked.update(
                instructor_schedule.get(instructor.instructor_id, set())
            )

        # Find quanta not blocked for this specific gene
        available_quanta = [
            q for q in context.available_quanta if q not in gene_blocked
        ]

        if len(available_quanta) < quanta_needed:
            # Not enough conflict-free quanta — use all quanta
            available_quanta = list(context.available_quanta)

        # Assign time quanta (use per-gene blocked set, not global)
        assigned_quanta = assign_conflict_free_quanta(
            quanta_needed, available_quanta, gene_blocked
        )

        if assigned_quanta and len(assigned_quanta) != quanta_needed:
            logger.warning(
                "%s: assign_conflict_free_quanta returned %d but needed %d",
                course_id,
                len(assigned_quanta),
                quanta_needed,
            )

        # CRITICAL: If assignment fails, pick a random start (not always 0)
        if not assigned_quanta:
            if len(context.available_quanta) >= quanta_needed:
                start_idx = random.randint(
                    0, len(context.available_quanta) - quanta_needed
                )
                assigned_quanta = context.available_quanta[
                    start_idx : start_idx + quanta_needed
                ]
            else:
                assigned_quanta = []
                while len(assigned_quanta) < quanta_needed:
                    assigned_quanta.extend(context.available_quanta)
                assigned_quanta = assigned_quanta[:quanta_needed]

    # VERIFICATION: Ensure we got exactly quanta_needed
    if len(assigned_quanta) != quanta_needed:
        logger.error(
            "BUG: Got %d quanta but needed %d for %s",
            len(assigned_quanta),
            quanta_needed,
            course_id,
        )

    # Create session gene with multi-group support
    actual_course_id = (
        course.course_id
        if hasattr(course, "course_id")
        else (course_id[0] if isinstance(course_id, tuple) else course_id)
    )

    actual_course_type = (
        course.course_type if hasattr(course, "course_type") else session_type
    )

    # Convert quanta list to contiguous representation
    from src.ga.core.quanta_converter import quanta_list_to_contiguous

    instructor_id = instructor.instructor_id
    start_q, num_q = quanta_list_to_contiguous(assigned_quanta)

    # DEBUG: Verify num_q matches quanta_needed
    if num_q != quanta_needed:
        logger.error(
            "CONVERSION BUG: %s %s: quanta_list_to_contiguous gave num_q=%s but quanta_needed=%s",
            course_id,
            session_type,
            num_q,
            quanta_needed,
        )
        logger.error("assigned_quanta length=%s", len(assigned_quanta))

    session_gene = SessionGene(
        course_id=actual_course_id,
        course_type=actual_course_type,
        instructor_id=instructor_id,
        group_ids=group_ids,  # Can be multiple groups
        room_id=room.room_id if room else "UNASSIGNED",
        start_quanta=start_q,
        num_quanta=num_q,
    )

    # Update tracking structures using the gene's contiguous quanta
    assigned_quanta = session_gene.get_quanta_list()
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

    return session_gene


def create_component_session_with_conflict_avoidance(
    course_id: str,
    group_id: str,
    component_type: str,
    hours: int,
    context: SchedulingContext,
    used_quanta: set[int],
    instructor_schedule: ScheduleMap,
    group_schedule: ScheduleMap,
    require_special_room: bool = False,
) -> SessionGene:
    """
    Create a single session while avoiding instructor and group conflicts.

    DEPRECATED: Use create_session_gene_with_conflict_avoidance instead.
    Kept for backwards compatibility with old population generators.
    """
    course = context.courses.get((course_id, component_type)) or context.courses.get(  # type: ignore[call-overload]
        course_id
    )
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

    # Create session gene
    # Extract course_id and course_type
    actual_course_id = course_id[0] if isinstance(course_id, tuple) else course_id  # type: ignore[unreachable]
    actual_course_type = component_type

    # Convert quanta list to contiguous representation
    from src.ga.core.quanta_converter import quanta_list_to_contiguous

    instructor_id = instructor.instructor_id
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

    # Update tracking structures using the gene's contiguous quanta
    assigned_quanta = session_gene.get_quanta_list()
    used_quanta.update(assigned_quanta)
    instructor_id = instructor.instructor_id
    if instructor_id not in instructor_schedule:
        instructor_schedule[instructor_id] = set()
    instructor_schedule[instructor_id].update(assigned_quanta)

    if group_id not in group_schedule:
        group_schedule[group_id] = set()
    group_schedule[group_id].update(assigned_quanta)

    return session_gene


def assign_conflict_free_quanta(
    quanta_needed: int, available_quanta: list[int], used_quanta: set[int]
) -> list[int]:
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
            logger.error(
                "BUG: _find_consecutive_block returned %s but needed %s",
                len(consecutive_block),
                quanta_needed,
            )
        return consecutive_block

    # Second attempt: Try in ALL available quanta (may have conflicts)
    # Repair operators will fix conflicts later
    consecutive_block = _find_consecutive_block(available_quanta, quanta_needed)
    if consecutive_block:
        if len(consecutive_block) != quanta_needed:
            logger.error(
                "BUG: _find_consecutive_block (all quanta) returned %s but needed %s",
                len(consecutive_block),
                quanta_needed,
            )
        return consecutive_block

    # Cannot satisfy requirement - fail gene creation
    return None  # type: ignore[return-value]


def _find_consecutive_block(
    free_quanta: list[int], block_size: int
) -> list[int] | None:
    """
    Find a single consecutive block of specified size.
    Returns None if not found.
    """
    if len(free_quanta) < block_size:
        return None

    sorted_free = sorted(free_quanta)

    for i in range(len(sorted_free) - block_size + 1):
        candidates = sorted_free[i : i + block_size]

        # Check if truly consecutive
        is_consecutive = all(
            candidates[j] - candidates[j - 1] == 1 for j in range(1, len(candidates))
        )

        if is_consecutive:
            return candidates

    return None


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
    course = context.courses.get((course_id, component_type)) or context.courses.get(  # type: ignore[call-overload]
        course_id
    )
    if not course:
        return None  # type: ignore[return-value]

    # Find qualified instructors
    qualified_instructors = find_qualified_instructors(course_id, context)
    if not qualified_instructors:
        # Fallback to any instructor if none qualified
        qualified_instructors = list(context.instructors.values())

    if not qualified_instructors:
        logger.warning("No instructors available for course %s", course_id)
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
        logger.warning("No rooms available for course %s", course_id)
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
    actual_course_id = course_id[0] if isinstance(course_id, tuple) else course_id  # type: ignore[unreachable]
    actual_course_type = component_type

    # Convert quanta list to contiguous representation
    from src.ga.core.quanta_converter import quanta_list_to_contiguous

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


def find_qualified_instructors(
    course_id: str | CourseKey, context: SchedulingContext
) -> list[Instructor]:
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


def find_qualified_instructors_with_availability(
    course_id: str | CourseKey,
    context: SchedulingContext,
    required_quanta: int,
    exclude_quanta: set[int] | None = None,
) -> list[tuple[Instructor, list[int]]]:
    """
    Find instructors qualified AND available for a session.

    This enforces instructor availability as a HARD FILTER during initialization,
    not just relying on repair operators.

    Args:
        course_id: Can be plain string or (course_code, course_type) tuple
        context: Scheduling context
        required_quanta: Number of consecutive quanta needed
        exclude_quanta: Quanta to exclude (already used by other sessions)

    Returns:
        List of (Instructor, available_slots) tuples where available_slots
        is a list of valid start quanta for this instructor.
        Sorted by availability (more options first).
    """
    exclude_set = exclude_quanta or set()
    results: list[tuple[Instructor, list[int]]] = []

    for instructor in context.instructors.values():
        # Check if qualified
        qualified_courses = getattr(instructor, "qualified_courses", [])
        if course_id not in qualified_courses:
            continue

        # Find available start slots
        available_starts: list[int] = []

        if instructor.is_full_time:
            # Full-time: available during all operating hours
            inst_available = set(context.available_quanta)
        else:
            # Part-time: only during their specified availability
            inst_available = set(instructor.available_quanta)

        # Remove excluded quanta
        inst_available -= exclude_set

        # Convert to sorted list for consecutive block search
        available_list = sorted(inst_available)

        # Find valid start positions for consecutive blocks
        for _i, start_q in enumerate(available_list):
            end_q = start_q + required_quanta

            # Check if all quanta in range are available
            valid = True
            for q in range(start_q, end_q):
                if q not in inst_available:
                    valid = False
                    break

            if valid:
                available_starts.append(start_q)

        if available_starts:
            results.append((instructor, available_starts))

    # Sort by number of available slots (more flexibility first)
    results.sort(key=lambda x: len(x[1]), reverse=True)

    return results


def find_suitable_rooms(
    course: Course,
    component_type: str,
    context: SchedulingContext,
    require_special_room: bool = False,
) -> list[Room]:
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

        # Course-level specific lab features (for practical courses)
        lab_feats = getattr(course, "specific_lab_features", None) or []

        # PRIORITY 1: Exact type match + specific feature match
        if room_str == required_str:
            if lab_feats:
                # For practical courses with lab requirements, room must have a matching feature
                room_spec = getattr(room, "specific_features", None) or []
                room_feat_set = {f.lower().strip() for f in room_spec}
                if any(f.lower().strip() in room_feat_set for f in lab_feats):
                    exact_matches.append(room)
                # else: type matches but wrong lab — skip
            else:
                exact_matches.append(room)
            continue

        # PRIORITY 2: Flexible match using Room's built-in method
        # (includes specific_lab_features matching for practical courses)
        if hasattr(
            room, "is_suitable_for_course_type"
        ) and room.is_suitable_for_course_type(required_str, lab_feats or None):
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


def assign_intelligent_quanta(
    quanta_needed: int, available_quanta: list[int]
) -> list[int]:
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
    possible_courses: list[Course],
    possible_instructors: list[Instructor],
    possible_groups: list[Group],
    possible_rooms: list[Room],
    available_quanta: list[int],
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
    from src.ga.core.quanta_converter import quanta_list_to_contiguous

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
) -> list[Individual]:
    """
    Wrapper function for backward compatibility.
    Now uses course-group aware generation instead of random.
    """
    if context is None:
        raise ValueError("Context must be provided for population generation")

    return generate_course_group_aware_population(n, context)


def generate_hybrid_population(n: int, context: SchedulingContext) -> list[Individual]:
    """
    Generate population with hybrid initialization strategy.

    Composition (default):
    - 40% greedy
    - 40% constraint-aware
    - 20% random
    """
    population: list[Individual] = []

    from src.config import get_config_or_default
    from src.ga.heuristics.construction import (
        earliest_deadline_first,
        largest_degree_first,
        most_constrained_first,
    )

    # Gracefully handle missing config (use defaults)
    cfg = get_config_or_default()
    enhancement_cfg = getattr(cfg, "enhancements", None)
    if enhancement_cfg and getattr(enhancement_cfg, "master_enabled", False):
        greedy_percent = getattr(enhancement_cfg, "greedy_initialization_percent", 0.4)
    else:
        greedy_percent = 0.4  # Default: 40% greedy

    greedy_count = int(n * greedy_percent)
    random_count = max(1, int(n * 0.2))
    smart_count = n - greedy_count - random_count

    silent = os.environ.get("_GA_WORKER_PROCESS") == "1"
    if not silent:
        logger.info(
            "Hybrid initialization: %s greedy, %s smart, %s random",
            greedy_count,
            smart_count,
            random_count,
        )

    hierarchy = analyze_group_hierarchy(context.groups)
    pair_tuples = generate_course_group_pairs(
        context.courses, context.groups, hierarchy, silent=True
    )

    num_workers = get_cpu_count()
    use_parallel = num_workers > 1 and n >= 10

    construction_heuristics = [
        largest_degree_first,
        most_constrained_first,
        earliest_deadline_first,
    ]

    for i in range(greedy_count):
        heuristic = construction_heuristics[i % len(construction_heuristics)]
        try:
            genes = heuristic(context)
            if genes:
                population.append(genes)
        except Exception:
            fallback = generate_course_group_aware_population(1, context)
            if fallback:
                population.append(fallback[0])

    smart_population = generate_course_group_aware_population(smart_count, context)
    population.extend(smart_population)

    if use_parallel:
        random_tasks = [(context, pair_tuples) for _ in range(random_count)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_random_construction_wrapper, random_tasks))
        population.extend([ind for ind in results if ind is not None])
    else:
        for _i in range(random_count):
            individual = _random_construction(context, pair_tuples)
            if individual:
                population.append(individual)

    while len(population) < n:
        extra = generate_course_group_aware_population(1, context)
        population.extend(extra)

    return population[:n]


def _random_construction_wrapper(
    args: tuple[SchedulingContext, list[DetailedPair]],
) -> Individual | None:
    context, pair_tuples = args
    individual = _random_construction(context, pair_tuples)
    if individual:
        return individual
    return None


def _random_construction(
    context: SchedulingContext, pair_tuples: list[DetailedPair]
) -> list[SessionGene]:
    """Generate one individual using pure random assignment."""
    genes: list[SessionGene] = []

    for course_key, group_ids, _session_type, num_quanta in pair_tuples:
        if num_quanta == 0:
            continue

        course = context.courses.get(course_key)
        if not course:
            continue

        subsession_durations = get_subsession_durations(
            course.quanta_per_week, course.course_type
        )

        for subsession_duration in subsession_durations:
            gene = _random_gene(course_key, group_ids, subsession_duration, context)
            if gene:
                genes.append(gene)

    return genes


def _random_gene(
    course_key: tuple[str, str],
    group_ids: list[str],
    num_quanta: int,
    context: SchedulingContext,
) -> SessionGene | None:
    course = context.courses.get(course_key)
    if not course:
        return None

    qualified = list(course.qualified_instructor_ids)
    if not qualified:
        return None

    instructor_id = random.choice(qualified)

    if not context.rooms:
        return None
    room_id = random.choice(list(context.rooms.keys()))

    available_quanta = list(context.available_quanta)
    if not available_quanta:
        return None

    max_start = len(available_quanta) - num_quanta
    if max_start < 0:
        return None
    start_idx = random.randint(0, max_start)
    start_quanta = available_quanta[start_idx]

    return SessionGene(
        course_id=course_key[0],
        course_type=course_key[1],
        group_ids=list(group_ids),
        instructor_id=instructor_id,
        room_id=room_id,
        start_quanta=start_quanta,
        num_quanta=num_quanta,
    )
