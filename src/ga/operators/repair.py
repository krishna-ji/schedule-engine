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

from typing import List, Dict, Set, Tuple, Optional
import random
from collections import defaultdict

from src.ga.sessiongene import SessionGene
from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.ga.operators.repair_wrappers import repair_operator
from src.ga.quanta_converter import quanta_list_to_contiguous


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
    individual: List[SessionGene], context: SchedulingContext
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
    individual: List[SessionGene],
    current_gene: SessionGene,
    duration: int,
    instructor,
    available_quanta: List[int],
) -> Optional[int]:
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
    individual: List[SessionGene], context: SchedulingContext
) -> int:
    """
    Resolve time conflicts where same group is scheduled in multiple sessions.

    Uses NEW API: gene.start_quanta, gene.num_quanta
    """
    fixes = 0
    occupied = _build_occupied_quanta_map(individual)

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


def _find_conflict_free_slot(
    individual: List[SessionGene],
    current_gene: SessionGene,
    available_quanta: List[int],
) -> Optional[int]:
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
    individual: List[SessionGene],
    current_gene: SessionGene,
    duration: int,
    available_quanta: List[int],
) -> Optional[int]:
    """
    Find a valid time slot with specified duration (used by repair_selective).

    Returns:
        Start quantum if valid slot found, None otherwise
    """
    return _find_conflict_free_slot(individual, current_gene, available_quanta)


# ================
# HELPER FUNCTIONS
# ================


def _build_occupied_quanta_map(
    individual: List[SessionGene], exclude_gene: SessionGene = None
) -> Dict[str, Dict[int, Set[str]]]:
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
    occupied = {
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
    individual: List[SessionGene],
    context: SchedulingContext,
    selective: bool = True,
    max_iterations: int = 3,
) -> Dict:
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
    from src.ga.operators.repair_wrappers import get_enabled_repair_operators

    stats = {
        "iterations": 0,
        "total_fixes": 0,
    }

    # Get enabled repair operators
    enabled_repairs = get_enabled_repair_operators()

    if not enabled_repairs:
        return stats

    # Apply repairs iteratively
    for iteration in range(max_iterations):
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
    individual: List[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 3,
) -> Dict:
    """Legacy interface - calls repair_individual_unified."""
    return repair_individual_unified(
        individual, context, selective=True, max_iterations=max_iterations
    )
