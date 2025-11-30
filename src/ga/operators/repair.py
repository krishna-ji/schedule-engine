"""
Repair Heuristics for Constraint Violation Restoration (Updated for Nov 2025 Architecture)

Deterministic repair operators that fix hard constraint violations in GA individuals.
Applied after mutation/crossover to project invalid solutions onto feasible region.

KEY UPDATE: Now uses SessionGene's contiguous representation (start_quanta + num_quanta)
instead of the old array-based quanta representation.

Repair Strategies:
1. Instructor Availability: Shift sessions to respect instructor schedules
2. Group Overlaps: Resolve time conflicts for same group
3. Room Conflicts: Fix double-bookings by shifting times or changing rooms
4. Instructor Conflicts: Resolve instructor double-bookings
5. Instructor Qualification: Reassign unqualified instructors
6. Room Type Mismatch: Match course requirements (lab vs classroom)
7. Session Clustering: Group isolated sessions into blocks (soft penalty reducer)

NOTE: repair_incomplete_or_extra_sessions REMOVED - not needed because:
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
- Priority-ordered: Lower priority number executes first
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
"""

from collections import defaultdict

from src.core.types import SchedulingContext
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
    instructor,
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
# 3. ROOM CONFLICT REPAIR (Priority 3)
# ================


@repair_operator(
    name="repair_room_conflicts",
    description="Fix room double-bookings by shifting sessions or reassigning rooms",
    priority=3,
    modifies_length=False,
)
def repair_room_conflicts(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """Resolve room conflicts by moving sessions or selecting compatible rooms."""
    fixes = 0
    occupied = _build_occupied_quanta_map(individual)

    for gene in individual:
        has_conflict = any(
            len(occupied["rooms"].get(q, set())) > 1
            for q in range(gene.start_quanta, gene.end_quanta)
        )

        if not has_conflict:
            continue

        # Try shifting first to keep same room assignment
        new_start = _find_conflict_free_slot(individual, gene, context.available_quanta)

        if new_start is not None:
            gene.start_quanta = new_start
            fixes += 1
            occupied = _build_occupied_quanta_map(individual)
            continue

        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        needs_lab = course.course_type == "practical" if course else False
        new_room = _find_compatible_room(individual, gene, context, needs_lab)
        if new_room is not None:
            gene.room_id = new_room
            fixes += 1
            occupied = _build_occupied_quanta_map(individual)

    return fixes


# ================
# 4. INSTRUCTOR CONFLICT REPAIR (Priority 4)
# ================


@repair_operator(
    name="repair_instructor_conflicts",
    description="Resolve instructor double-bookings by shifting sessions",
    priority=4,
    modifies_length=False,
)
def repair_instructor_conflicts(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """Resolve instructor conflicts by finding conflict-free slots."""
    fixes = 0
    occupied = _build_occupied_quanta_map(individual)

    for gene in individual:
        has_conflict = any(
            len(occupied["instructors"].get(q, set())) > 1
            for q in range(gene.start_quanta, gene.end_quanta)
        )

        if not has_conflict:
            continue

        new_start = _find_conflict_free_slot(individual, gene, context.available_quanta)

        if new_start is not None:
            gene.start_quanta = new_start
            fixes += 1
            occupied = _build_occupied_quanta_map(individual)

    return fixes


# ================
# 5. INSTRUCTOR QUALIFICATION REPAIR (Priority 5)
# ================


@repair_operator(
    name="repair_instructor_qualifications",
    description="Reassign sessions to qualified instructors",
    priority=5,
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
# 6. ROOM TYPE MISMATCH REPAIR (Priority 6)
# ================


@repair_operator(
    name="repair_room_type_mismatches",
    description="Match course requirements with compatible room types",
    priority=6,
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

        needs_lab = course.course_type == "practical"
        is_lab = getattr(room, "room_features", None) == "lab"

        if needs_lab == is_lab:
            continue

        replacement_room = _find_compatible_room(individual, gene, context, needs_lab)
        if replacement_room is not None:
            gene.room_id = replacement_room
            fixes += 1

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
    needs_lab: bool,
) -> str | None:
    """Find a room matching lab/theory requirement without conflicts."""
    occupied = _build_occupied_quanta_map(individual, current_gene)
    duration_range = range(current_gene.start_quanta, current_gene.end_quanta)

    # Determine required room type
    required_type = "practical" if needs_lab else "lecture"

    for room in context.rooms.values():
        # Check room type compatibility using proper matching logic
        room_type = getattr(room, "room_features", "lecture").lower().strip()
        if not _room_type_compatible(required_type, room_type):
            continue

        conflict = False
        for q in duration_range:
            if room.room_id in occupied["rooms"].get(q, set()):
                conflict = True
                break

        if conflict:
            continue

        return room.room_id

    return None


def _room_type_compatible(required: str, room_type: str) -> bool:
    """Check if room type satisfies requirement with flexible compatibility."""
    # Exact match
    if required == room_type:
        return True

    # Lecture/theory courses: Accept lecture, classroom, auditorium
    if required in ["lecture", "classroom", "theory"] and room_type in [
        "lecture",
        "classroom",
        "auditorium",
        "seminar",
        "tutorial",
    ]:
        return True

    # Practical/lab courses: Accept practical, lab variants
    return required in ["practical", "lab", "laboratory"] and room_type in [
        "practical",
        "lab",
        "laboratory",
        "computer_lab",
        "science_lab",
    ]


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
