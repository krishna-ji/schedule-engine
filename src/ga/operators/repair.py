"""
Repair Heuristics for Constraint Violation Restoration (Updated for Nov 2025 Architecture)

Deterministic repair operators that fix hard constraint violations in GA individuals.
Applied after mutation/crossover to project invalid solutions onto feasible region.

KEY UPDATE: Now uses SessionGene's contiguous representation (start_quanta + num_quanta)
instead of the old array-based quanta representation.

Constraint-to-Repair Mapping (6 of 8 hard constraints have repairs):
┌──────┬─────────────────────────────────┬────────────────────────────────────┬──────────┐
│ Code │ Constraint                      │ Repair Operator                    │ Priority │
├──────┼─────────────────────────────────┼────────────────────────────────────┼──────────┤
│ HC1  │ student_group_exclusivity       │ repair_group_overlaps              │    2     │
│ HC2  │ instructor_exclusivity          │ repair_instructor_conflicts        │    5     │
│ HC3  │ instructor_qualifications       │ repair_instructor_qualifications   │    6     │
│ HC4  │ room_suitability                │ repair_room_type_mismatches        │    7     │
│ HC5  │ instructor_time_availability    │ repair_instructor_availability     │    1     │
│ HC6  │ room_time_availability          │  NO REPAIR (always available)    │    -     │
│ HC7  │ course_completeness             │  NO REPAIR (structural integrity)│    -     │
│ HC8  │ room_exclusivity                │ repair_room_conflicts +            │   4,3    │
│      │                                 │ repair_room_overlap_reassign       │          │
└──────┴─────────────────────────────────┴────────────────────────────────────┴──────────┘

Soft Constraint Repairs (1 of 4):
  SC4 (session_continuity): repair_session_clustering_selective (priority 8)

NOTE: HC7 repair removed - not needed because:
- Population initialization creates correct gene counts per (course, group)
- Crossover only swaps attributes, never adds/removes genes
- Mutation only changes attributes, never adds/removes genes
- course_completeness constraint verifies correctness (should be 0)

Availability Model:
- Instructor availability: Checked (part-time may have restrictions)
- Room availability: NOT checked (always available during operating hours)
- Group availability: NOT checked (default to all operating hours)

Architecture:
- Decorator-based registry: Auto-register repair operators (like constraints)
- Priority-ordered: Lower priority number executes first (1-7 for hard, 8 for soft)
- In-place modification: Invalidate fitness after repair
- Unified interface: repair_individual_unified() with selective optimization

Repair Modes:
- Full: Scans all genes (thorough, slower)
- Selective: Only repairs violated genes (3-4x faster, recommended)

Usage:
    from src.ga.operators.repair import repair_individual_unified

    # Recommended: Use selective mode
    stats = repair_individual_unified(individual, context, selective=True)
    print(f"Fixed {stats['total_fixes']} violations")

    # Mode B (Memetic): Applied to elite 20% with max_iterations=100
    # Mode C-E: Applied with adaptive triggers (stagnation, periodic)
"""

from collections import defaultdict
from collections.abc import Iterable

from src.core.types import SchedulingContext
from src.entities.instructor import Instructor
from src.ga.operators.repair_wrappers import repair_operator
from src.ga.sessiongene import SessionGene

# ================
# 1. INSTRUCTOR AVAILABILITY REPAIR (Priority 1)
# ================


@repair_operator(
    name="repair_instructor_availability",
    description="Fix instructor availability violations (shift sessions to instructor-available times)",
    priority=1,
    modifies_length=False,
)
def repair_instructor_availability(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """
    Fix instructor availability violations by shifting genes to valid time slots.

    Uses NEW API: gene.start_quanta, gene.num_quanta (contiguous representation)

    Args:
        individual: List of SessionGene objects (GA chromosome)
        context: Scheduling context with entities and available quanta

    Returns:
        Number of genes repaired
    """
    fixes = 0

    for gene in individual:
        # Get instructor object
        instructor = context.instructors.get(gene.instructor_id)
        if not instructor:
            continue

        # Check if current quanta violate instructor availability
        needs_repair = False
        for q in range(gene.start_quanta, gene.end_quanta):
            if q not in instructor.available_quanta:
                needs_repair = True
                break

        if not needs_repair:
            continue

        # Find valid replacement quanta
        new_start = _find_instructor_available_slot(
            individual, gene, gene.num_quanta, instructor, context.available_quanta
        )

        if new_start is not None:
            gene.start_quanta = new_start
            # num_quanta stays the same (preserve duration)
            fixes += 1

    return fixes


def _find_instructor_available_slot(
    individual: list[SessionGene],
    current_gene: SessionGene,
    duration: int,
    instructor: Instructor,
    available_quanta: list[int],
) -> int | None:
    """
    Find a valid start quantum where instructor is available and no conflicts exist.

    Returns:
        Start quantum if valid slot found, None otherwise
    """
    # Build conflict map from other genes
    occupied = _build_occupied_quanta_map(individual, current_gene)

    # Get room and group IDs from current gene
    room_id = current_gene.room_id
    group_ids = current_gene.group_ids

    # Try to find consecutive available quanta
    for start_q in available_quanta:
        # Check if we can fit duration quanta starting at start_q
        end_q = start_q + duration

        # Check if all quanta in range are valid operating times
        if end_q > max(available_quanta) + 1:
            continue

        # Check instructor availability (PRIMARY CHECK)
        if not all(q in instructor.available_quanta for q in range(start_q, end_q)):
            continue

        # Check no conflicts with other genes
        conflict_free = True
        for q in range(start_q, end_q):
            # Instructor conflict check
            if instructor.instructor_id in occupied["instructors"].get(q, set()):
                conflict_free = False
                break

            # Room conflict check
            if room_id in occupied["rooms"].get(q, set()):
                conflict_free = False
                break

            # Group conflict check
            for group_id in group_ids:
                if group_id in occupied["groups"].get(q, set()):
                    conflict_free = False
                    break

            if not conflict_free:
                break

        if conflict_free:
            return start_q

    return None  # No valid slot found


# ================
# 2. GROUP OVERLAP REPAIR (Priority 2)
# ================


@repair_operator(
    name="repair_group_overlaps",
    description="Fix group schedule overlaps (same group in multiple sessions)",
    priority=2,
    modifies_length=False,
)
def repair_group_overlaps(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """
    Resolve time conflicts where same group is scheduled in multiple sessions.

    Uses NEW API: gene.start_quanta, gene.num_quanta
    """
    fixes = 0
    # occupied = _build_occupied_quanta_map(individual)  # Unused

    for gene in individual:
        # Check if any group in this gene has conflicts
        has_conflict = False
        for group_id in gene.group_ids:
            for q in range(gene.start_quanta, gene.end_quanta):
                # Count how many genes use this group at this quantum
                genes_at_q = [
                    g
                    for g in individual
                    if group_id in g.group_ids and g.start_quanta <= q < g.end_quanta
                ]
                if len(genes_at_q) > 1:
                    has_conflict = True
                    break
            if has_conflict:
                break

        if has_conflict:
            # Try to shift to conflict-free time
            new_start = _find_conflict_free_slot(
                individual, gene, context.available_quanta
            )
            if new_start is not None:
                gene.start_quanta = new_start
                fixes += 1

    return fixes


# ================
# 3. ROOM OVERLAP REASSIGNMENT (Priority 3)
# ================


@repair_operator(
    name="repair_room_overlap_reassign",
    description="Resolve room overlaps by moving sessions into idle rooms before shifting times",
    priority=3,
    modifies_length=False,
)
def repair_room_overlap_reassign(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """Prefer room swaps over time shifts when plenty of rooms exist."""

    fixes = 0
    occupied = _build_occupied_quanta_map(individual)

    for gene in individual:
        if not _has_room_conflict(gene, occupied):
            continue

        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        required_type = (
            getattr(course, "required_room_features", "lecture").lower().strip()
            if course
            else "lecture"
        )

        candidate_room = _find_compatible_room(individual, gene, context, required_type)
        if candidate_room is None or candidate_room == gene.room_id:
            continue

        gene.room_id = candidate_room
        fixes += 1
        occupied = _build_occupied_quanta_map(individual)

    return fixes


# ================
# 4. ROOM CONFLICT REPAIR (Priority 4)
# ================


@repair_operator(
    name="repair_room_conflicts",
    description="Fallback room conflict repair that shifts sessions when no alternate room exists",
    priority=4,
    modifies_length=False,
)
def repair_room_conflicts(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """Resolve room conflicts by moving sessions or selecting compatible rooms."""
    fixes = 0
    occupied = _build_occupied_quanta_map(individual)

    for gene in individual:
        if not _has_room_conflict(gene, occupied):
            continue

        # Try shifting first to keep same room assignment (room swap ran earlier)
        new_start = _find_conflict_free_slot(individual, gene, context.available_quanta)

        if new_start is not None:
            gene.start_quanta = new_start
            fixes += 1
            occupied = _build_occupied_quanta_map(individual)
            continue

        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        required_type = (
            getattr(course, "required_room_features", "lecture").lower().strip()
            if course
            else "lecture"
        )
        new_room = _find_compatible_room(individual, gene, context, required_type)
        if new_room is not None:
            gene.room_id = new_room
            fixes += 1
            occupied = _build_occupied_quanta_map(individual)

    return fixes


# ================
# 5. INSTRUCTOR CONFLICT REPAIR (Priority 5)
# ================


@repair_operator(
    name="repair_instructor_conflicts",
    description="Resolve instructor double-bookings by shifting sessions or swapping instructors",
    priority=5,
    modifies_length=False,
)
def repair_instructor_conflicts(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """Resolve instructor conflicts by time shift or instructor swap."""
    fixes = 0
    occupied = _build_occupied_quanta_map(individual)

    for gene in individual:
        has_conflict = any(
            len(occupied["instructors"].get(q, set())) > 1
            for q in range(gene.start_quanta, gene.end_quanta)
        )

        if not has_conflict:
            continue

        # Phase 1: Try time shift (preserve instructor)
        new_start = _find_conflict_free_slot(individual, gene, context.available_quanta)

        if new_start is not None:
            gene.start_quanta = new_start
            fixes += 1
            occupied = _build_occupied_quanta_map(individual)
            continue

        # Phase 2: Try instructor swap (preserve time)
        course_key = (gene.course_id, gene.course_type)
        new_instructor = _find_available_instructor(
            individual, gene, context, course_key
        )

        if new_instructor is not None:
            gene.instructor_id = new_instructor
            fixes += 1
            occupied = _build_occupied_quanta_map(individual)

    return fixes


# ================
# 6. INSTRUCTOR QUALIFICATION REPAIR (Priority 6)
# ================


@repair_operator(
    name="repair_instructor_qualifications",
    description="Reassign sessions to qualified instructors",
    priority=6,
    modifies_length=False,
)
def repair_instructor_qualifications(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """Ensure instructors assigned to sessions are properly qualified."""
    fixes = 0

    for gene in individual:
        course_key = (gene.course_id, gene.course_type)
        instructor = context.instructors.get(gene.instructor_id)
        course = context.courses.get(course_key)

        if not course:
            continue

        if instructor and course_key in instructor.qualified_courses:
            continue

        replacement = _find_available_instructor(individual, gene, context, course_key)
        if replacement is not None:
            gene.instructor_id = replacement
            fixes += 1

    return fixes


# ================
# 7. ROOM TYPE MISMATCH REPAIR (Priority 7)
# ================


@repair_operator(
    name="repair_room_type_mismatches",
    description="Match course requirements with compatible room types",
    priority=7,
    modifies_length=False,
)
def repair_room_type_mismatches(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """Swap rooms when course type and room features disagree."""
    fixes = 0

    for gene in individual:
        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        room = context.rooms.get(gene.room_id)

        if not course or not room:
            continue

        # Get required and actual room types
        required_type = (
            getattr(course, "required_room_features", "lecture").lower().strip()
        )
        room_type = getattr(room, "room_features", "lecture").lower().strip()

        # Check if already compatible using flexible matching
        from src.utils.room_compatibility import is_room_type_compatible

        if is_room_type_compatible(required_type, room_type):
            continue  # Already matches

        # Find compatible room (pass required_type directly)
        replacement_room = _find_compatible_room(
            individual, gene, context, required_type
        )
        if replacement_room is not None:
            gene.room_id = replacement_room
            fixes += 1

    return fixes


@repair_operator(
    name="repair_paired_cohort_practicals",
    description=(
        "Improve alignment of practical sessions for paired cohorts by shifting "
        "sessions to parallel time slots when feasible."
    ),
    priority=8,
    modifies_length=False,
)
def repair_paired_cohort_practicals(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """Align practical sessions for paired cohorts where possible.

    This operator targets practical courses shared by configured cohort
    pairs (e.g., bei1a/bei1b). For each such pair and course, it attempts
    to move one cohort's practical sessions so that both cohorts attend the
    course in parallel time windows, while preserving all hard constraints
    (group, room, and instructor exclusivity plus availability).
    """

    cohort_pairs: Iterable[tuple[str, str]] = context.cohort_pairs or []
    if not cohort_pairs:
        return 0

    fixes = 0

    # Index practical sessions per (course_id, group_id)
    practical_map: dict[tuple[str, str], list[SessionGene]] = defaultdict(list)

    for gene in individual:
        if gene.course_type.lower() != "practical":
            continue

        for group_id in gene.group_ids:
            key = (gene.course_id, group_id)
            practical_map[key].append(gene)

    if not practical_map:
        return 0

    # Helper to collect occupied quanta for one cohort and course
    def _collect_quanta(
        course_id: str,
        group_id: str,
    ) -> set[int]:
        key = (course_id, group_id)
        result: set[int] = set()
        for g in practical_map.get(key, []):
            result.update(range(g.start_quanta, g.end_quanta))
        return result

    # Local conflict checks against hard constraints
    def _is_move_feasible(
        target_gene: SessionGene,
        group_id: str,
        new_start: int,
        duration: int,
    ) -> bool:
        new_end = new_start + duration
        instructor = context.instructors.get(target_gene.instructor_id)

        for gene in individual:
            if gene is target_gene:
                continue

            # Precompute intersection interval once per gene for performance
            if gene.end_quanta <= new_start or gene.start_quanta >= new_end:
                continue

            for _q in range(
                max(new_start, gene.start_quanta), min(new_end, gene.end_quanta)
            ):
                # Group exclusivity
                if group_id in gene.group_ids:
                    return False

                # Room exclusivity
                if (
                    target_gene.room_id is not None
                    and gene.room_id == target_gene.room_id
                ):
                    return False

                # Instructor exclusivity
                if gene.instructor_id == target_gene.instructor_id:
                    return False

        # Instructor time availability
        if instructor is not None and not instructor.is_full_time:
            for q in range(new_start, new_end):
                if q not in instructor.available_quanta:
                    return False

        return True

    # Main loop over cohort pairs and shared practical courses
    for left_id, right_id in cohort_pairs:
        # Discover shared practical courses between the pair
        course_ids: set[str] = set()
        for course_id, group_id in practical_map:
            if group_id in (left_id, right_id):
                course_ids.add(course_id)

        for course_id in course_ids:
            left_key = (course_id, left_id)
            right_key = (course_id, right_id)

            if left_key not in practical_map or right_key not in practical_map:
                continue

            left_quanta = _collect_quanta(course_id, left_id)
            right_quanta = _collect_quanta(course_id, right_id)

            if not left_quanta and not right_quanta:
                continue

            current_diff = left_quanta.symmetric_difference(right_quanta)
            if not current_diff:
                continue

            # Use left cohort's pattern as anchor and move right cohort's sessions
            anchor_quanta = left_quanta

            for gene in practical_map[right_key]:
                duration = gene.num_quanta

                # Generate candidate starts sorted by overlap with anchor pattern
                candidates: list[tuple[int, int]] = []
                for start_q in context.available_quanta:
                    end_q = start_q + duration
                    if end_q > max(context.available_quanta) + 1:
                        continue

                    overlap = 0
                    for q in range(start_q, end_q):
                        if q in anchor_quanta:
                            overlap += 1

                    if overlap > 0:
                        candidates.append((start_q, overlap))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: item[1], reverse=True)

                moved = False
                for candidate_start, _overlap in candidates:
                    if not _is_move_feasible(gene, right_id, candidate_start, duration):
                        continue

                    # Compute new right-quanta pattern and see if misalignment improves
                    new_right_quanta = set(right_quanta)
                    for q in range(gene.start_quanta, gene.end_quanta):
                        if q in new_right_quanta:
                            new_right_quanta.remove(q)
                    for q in range(candidate_start, candidate_start + duration):
                        new_right_quanta.add(q)

                    new_diff = left_quanta.symmetric_difference(new_right_quanta)
                    if len(new_diff) < len(current_diff):
                        gene.start_quanta = candidate_start
                        right_quanta = new_right_quanta
                        current_diff = new_diff
                        fixes += 1
                        moved = True
                        break

                if moved:
                    # Limit work per pair-course to keep runtime under control
                    break

    return fixes


def _find_conflict_free_slot(
    individual: list[SessionGene],
    current_gene: SessionGene,
    available_quanta: list[int],
) -> int | None:
    """Find a time slot with no group/room/instructor conflicts."""
    occupied = _build_occupied_quanta_map(individual, current_gene)
    duration = current_gene.num_quanta

    for start_q in available_quanta:
        end_q = start_q + duration
        if end_q > max(available_quanta) + 1:
            continue

        # Check no conflicts
        conflict_free = True
        for q in range(start_q, end_q):
            # Check instructor, room, and group conflicts
            if current_gene.instructor_id in occupied["instructors"].get(q, set()):
                conflict_free = False
                break
            if current_gene.room_id in occupied["rooms"].get(q, set()):
                conflict_free = False
                break
            for group_id in current_gene.group_ids:
                if group_id in occupied["groups"].get(q, set()):
                    conflict_free = False
                    break
            if not conflict_free:
                break

        if conflict_free:
            return start_q

    return None


def _find_available_slot(
    individual: list[SessionGene],
    current_gene: SessionGene,
    duration: int,
    available_quanta: list[int],
) -> int | None:
    """
    Find a valid time slot with specified duration (used by repair_selective).

    Returns:
        Start quantum if valid slot found, None otherwise
    """
    return _find_conflict_free_slot(individual, current_gene, available_quanta)


def _find_available_instructor(
    individual: list[SessionGene],
    current_gene: SessionGene,
    context: SchedulingContext,
    course_key: tuple[str, str],
) -> str | None:
    """Find a qualified instructor who is available for the session window."""
    occupied = _build_occupied_quanta_map(individual, current_gene)
    duration_range = range(current_gene.start_quanta, current_gene.end_quanta)

    for instructor in context.instructors.values():
        if course_key not in getattr(instructor, "qualified_courses", set()):
            continue

        if not all(q in instructor.available_quanta for q in duration_range):
            continue

        conflict = False
        for q in duration_range:
            if instructor.instructor_id in occupied["instructors"].get(q, set()):
                conflict = True
                break

        if conflict:
            continue

        return instructor.instructor_id

    return None


def _find_compatible_room(
    individual: list[SessionGene],
    current_gene: SessionGene,
    context: SchedulingContext,
    required_type: str,
) -> str | None:
    """Find a room matching required type without conflicts.

    Args:
        individual: Current schedule
        current_gene: Gene to find room for
        context: Scheduling context
        required_type: Required room type ("lecture", "practical", etc.)

    Returns:
        Room ID if found, None otherwise
    """
    occupied = _build_occupied_quanta_map(individual, current_gene)
    duration_range = range(current_gene.start_quanta, current_gene.end_quanta)

    from src.utils.room_compatibility import is_room_type_compatible

    # Calculate total enrollment for capacity check
    total_enrollment = sum(
        context.groups[gid].student_count
        for gid in current_gene.group_ids
        if gid in context.groups
    )

    # Collect compatible candidates and sort by preference
    candidates = []
    for room in context.rooms.values():
        # Check room type compatibility using centralized logic
        room_type = getattr(room, "room_features", "lecture").lower().strip()
        if not is_room_type_compatible(required_type, room_type):
            continue

        # Check capacity (hard requirement)
        if room.capacity < total_enrollment:
            continue

        # Check time conflicts
        conflict = False
        for q in duration_range:
            if room.room_id in occupied["rooms"].get(q, set()):
                conflict = True
                break

        if conflict:
            continue

        # Calculate fit quality (prefer rooms close to enrollment size)
        capacity_ratio = total_enrollment / room.capacity if room.capacity > 0 else 0
        candidates.append((room.room_id, capacity_ratio))

    # Return best fit (highest capacity utilization)
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


# ================
# HELPER FUNCTIONS
# ================


def _build_occupied_quanta_map(
    individual: list[SessionGene], exclude_gene: SessionGene | None = None
) -> dict[str, dict[int, set[str]]]:
    """
    Build occupation map for detecting conflicts.

    Uses NEW API: range(gene.start_quanta, gene.end_quanta)

    Returns:
        {
            "groups": {quantum: {group_id, ...}},
            "rooms": {quantum: {room_id, ...}},
            "instructors": {quantum: {instructor_id, ...}}
        }
    """
    occupied: dict[str, dict[int, set[str]]] = {
        "groups": defaultdict(set),
        "rooms": defaultdict(set),
        "instructors": defaultdict(set),
    }

    for gene in individual:
        if exclude_gene and gene is exclude_gene:
            continue

        for q in range(gene.start_quanta, gene.end_quanta):
            occupied["rooms"][q].add(gene.room_id)
            occupied["instructors"][q].add(gene.instructor_id)
            for group_id in gene.group_ids:
                occupied["groups"][q].add(group_id)

    return occupied


def _has_room_conflict(
    gene: SessionGene, occupied_map: dict[str, dict[int, set[str]]]
) -> bool:
    """Check whether a gene shares any room quanta with another session."""

    for q in range(gene.start_quanta, gene.end_quanta):
        if len(occupied_map["rooms"].get(q, set())) > 1:
            return True
    return False


# ================
# ORCHESTRATION
# ================


def repair_individual_unified(
    individual: list[SessionGene],
    context: SchedulingContext,
    selective: bool = True,
    max_iterations: int = 3,
) -> dict:
    """
    Apply enabled repair heuristics using registry pattern.

    Args:
        individual: GA individual (chromosome) to repair
        context: Scheduling context with entities
        selective: Use selective mode (faster, recommended)
        max_iterations: Maximum repair passes

    Returns:
        Dict with repair statistics
    """
    import logging

    logger = logging.getLogger(__name__)

    if selective:
        try:
            from src.config import get_config
            from src.ga.operators.repair_selective import repair_individual_selective

            detection_strategy = get_config().repair.detection_strategy
            logger.debug(f" Applying selective repair (strategy={detection_strategy})")
            selective_stats = repair_individual_selective(
                individual,
                context,
                max_iterations=max_iterations,
                detection_strategy=detection_strategy,
            )
            if selective_stats is not None:
                return selective_stats
        except Exception:  # pragma: no cover - fallback to full scan
            pass

    from src.ga.operators.repair_wrappers import get_enabled_repair_operators

    stats = {
        "iterations": 0,
        "total_fixes": 0,
    }

    # Get enabled repair operators
    enabled_repairs = get_enabled_repair_operators()

    if not enabled_repairs:
        logger.debug(" Repair system: No operators enabled")
        return stats

    logger.debug(f" Applying full repair ({len(enabled_repairs)} operators enabled)")

    # Apply repairs iteratively
    for _iteration in range(max_iterations):
        iteration_fixes = 0

        for repair_name, repair_meta in enabled_repairs.items():
            repair_func = repair_meta.function
            fixes = repair_func(individual, context)

            stats[f"{repair_name}_fixes"] = stats.get(f"{repair_name}_fixes", 0) + fixes
            iteration_fixes += fixes

        stats["iterations"] += 1
        stats["total_fixes"] += iteration_fixes

        # Stop if no fixes were made
        if iteration_fixes == 0:
            break

    return stats


# Alias for backward compatibility
def repair_individual(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 3,
) -> dict:
    """Legacy interface - calls repair_individual_unified."""
    return repair_individual_unified(
        individual, context, selective=True, max_iterations=max_iterations
    )
