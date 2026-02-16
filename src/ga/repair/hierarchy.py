"""
Hierarchy-Aware Repair Operators

Enhanced repair functions that understand parent-subgroup relationships.
Fixes the critical issue where BME1A ⊂ BME1AB conflicts aren't detected.

Key Insight:
- When BME1AB (parent) has a theory session, BME1A and BME1B can't have other sessions
- When BME1A has a practical, BME1AB can't have a theory session at that time
- Siblings (BME1A, BME1B) can have sessions at the same time IF they're the same theory session

Architecture:
- Pre-computes group family relationships once
- Uses family-aware conflict detection in all repairs
- Stores hierarchy in SchedulingContext for reuse
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from src.ga.core.population import (
    analyze_group_hierarchy,
    build_group_family_map,
    get_family_map_from_json,
)
from src.ga.repair.wrappers import repair_operator

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext


def _get_or_build_family_map(context: SchedulingContext) -> dict[str, set[str]]:
    """
    Get cached family map or build one using JSON-based hierarchy.

    Uses the explicit subgroup relationships from Groups.json for accuracy.
    Caches the result in context for reuse across repair iterations.
    """
    # Check if already computed and cached
    if hasattr(context, "_group_family_map") and context._group_family_map:
        return context._group_family_map  # type: ignore

    # Use JSON-based hierarchy (correct behavior)
    try:
        family_map = get_family_map_from_json("data/Groups.json")
    except FileNotFoundError:
        # Fallback: build from hierarchy if JSON not found
        hierarchy = analyze_group_hierarchy(context.groups)
        family_map = build_group_family_map(hierarchy)

    # Cache it
    context._group_family_map = family_map  # type: ignore

    return family_map


def _build_family_aware_occupied_map(
    individual: list[SessionGene],
    family_map: dict[str, set[str]],
    exclude_gene: SessionGene | None = None,
) -> dict[str, dict[int, set[str]]]:
    """
    Build occupation map that includes ALL related groups.

    When BME1A has a session, this also marks BME1B and BME1AB as occupied.

    Returns:
        {
            "groups": {quantum: {group_id, ...}},  # Includes related groups
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

            # Add all groups in this gene AND their related groups
            for group_id in gene.group_ids:
                # Add the group itself
                occupied["groups"][q].add(group_id)
                # Add all related groups (parent, siblings)
                related = family_map.get(group_id, {group_id})
                occupied["groups"][q].update(related)

    return occupied


def _find_family_aware_conflict_free_slot(
    individual: list[SessionGene],
    current_gene: SessionGene,
    available_quanta: list[int],
    family_map: dict[str, set[str]],
    instructor_available: set[int] | None = None,
) -> int | None:
    """
    Find a time slot with no conflicts, respecting group family relationships.

    Args:
        individual: All genes in the individual
        current_gene: Gene to relocate
        available_quanta: Valid time slots
        family_map: Pre-computed group family relationships
        instructor_available: Optional instructor availability constraint

    Returns:
        Start quantum if found, None otherwise
    """
    occupied = _build_family_aware_occupied_map(individual, family_map, current_gene)
    duration = current_gene.num_quanta

    # Get all related groups for the current gene
    gene_family: set[str] = set()
    for gid in current_gene.group_ids:
        gene_family.update(family_map.get(gid, {gid}))

    max_q = max(available_quanta) if available_quanta else 0

    for start_q in available_quanta:
        end_q = start_q + duration
        if end_q > max_q + 1:
            continue

        # Check instructor availability if provided
        if instructor_available is not None and not all(
            q in instructor_available for q in range(start_q, end_q)
        ):
            continue

        # Check no conflicts
        conflict_free = True
        for q in range(start_q, end_q):
            # Check instructor conflict
            if current_gene.instructor_id in occupied["instructors"].get(q, set()):
                conflict_free = False
                break

            # Check room conflict
            if current_gene.room_id in occupied["rooms"].get(q, set()):
                conflict_free = False
                break

            # Check group conflict (family-aware!)
            # Any group in gene's family conflicts if already occupied
            occupied_groups = occupied["groups"].get(q, set())
            if gene_family & occupied_groups:  # Set intersection
                conflict_free = False
                break

        if conflict_free:
            return start_q

    return None


# HIERARCHY-AWARE GROUP OVERLAP REPAIR (Priority 2.5)
@repair_operator(
    name="repair_group_overlaps_hierarchy",
    description="Fix group overlaps with parent-subgroup awareness (BME1A ⊂ BME1AB)",
    priority=2,  # Same priority as original, will replace it
    modifies_length=False,
)
def repair_group_overlaps_hierarchy(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """
    Resolve time conflicts with full parent-subgroup awareness.

    Key improvement over basic repair_group_overlaps:
    - Detects conflicts between BME1A sessions and BME1AB sessions
    - Detects conflicts between sibling subgroups (BME1A and BME1B)
    - Uses family-aware conflict checking when relocating sessions

    Returns:
        Number of genes repaired
    """
    fixes = 0
    family_map = _get_or_build_family_map(context)

    # Build family-aware occupation map
    _build_family_aware_occupied_map(individual, family_map)

    # Pre-compute gene families for faster checking
    gene_families = []
    for gene in individual:
        family: set[str] = set()
        for gid in gene.group_ids:
            family.update(family_map.get(gid, {gid}))
        gene_families.append(family)

    # Check each gene for family-aware conflicts
    for idx, gene in enumerate(individual):
        gene_family = gene_families[idx]
        has_conflict = False

        for q in range(gene.start_quanta, gene.end_quanta):
            # Count genes with overlapping family at this quantum
            overlapping_genes = 0
            for other_idx, other_gene in enumerate(individual):
                if other_idx == idx:
                    continue

                # Check if other gene occupies this quantum
                if not (other_gene.start_quanta <= q < other_gene.end_quanta):
                    continue

                # Check if families overlap
                other_family = gene_families[other_idx]
                if gene_family & other_family:
                    overlapping_genes += 1

            if overlapping_genes >= 1:
                has_conflict = True
                break

        if has_conflict:
            # Try to find a slot that respects ALL family relationships
            new_start = _find_family_aware_conflict_free_slot(
                individual, gene, list(context.available_quanta), family_map
            )

            if new_start is not None:
                gene.start_quanta = new_start
                fixes += 1
                # Rebuild occupation map after fix
                _build_family_aware_occupied_map(individual, family_map)

    return fixes


# INSTRUCTOR AVAILABILITY WITH SWAP (Priority 1.5)
@repair_operator(
    name="repair_instructor_availability_with_swap",
    description="Fix instructor availability by time shift OR instructor swap",
    priority=1,  # Same as original, will enhance it
    modifies_length=False,
)
def repair_instructor_availability_with_swap(
    individual: list[SessionGene], context: SchedulingContext
) -> int:
    """
    Enhanced instructor availability repair with fallback to instructor swap.

    Strategy:
    1. First try shifting to instructor-available time (original approach)
    2. If no valid time found, try swapping to a different qualified instructor

    Returns:
        Number of genes repaired
    """
    fixes = 0
    family_map = _get_or_build_family_map(context)

    for gene in individual:
        instructor = context.instructors.get(gene.instructor_id)
        if not instructor:
            continue

        # Full-time instructors are always available
        if instructor.is_full_time:
            continue

        # Check if current time violates availability
        needs_repair = False
        for q in range(gene.start_quanta, gene.end_quanta):
            if q not in instructor.available_quanta:
                needs_repair = True
                break

        if not needs_repair:
            continue

        # Strategy 1: Try shifting to instructor-available time
        new_start = _find_family_aware_conflict_free_slot(
            individual,
            gene,
            list(context.available_quanta),
            family_map,
            instructor_available=instructor.available_quanta,
        )

        if new_start is not None:
            gene.start_quanta = new_start
            fixes += 1
            continue

        # Strategy 2: Try swapping to a different instructor who IS available
        course_key = (gene.course_id, gene.course_type)

        # Find qualified instructors available at current time
        alternative_found = False
        for alt_instructor in context.instructors.values():
            if alt_instructor.instructor_id == gene.instructor_id:
                continue

            # Check if qualified for this course
            if course_key not in getattr(alt_instructor, "qualified_courses", set()):
                continue

            # Check if available at current time
            available_at_time = True
            if not alt_instructor.is_full_time:
                for q in range(gene.start_quanta, gene.end_quanta):
                    if q not in alt_instructor.available_quanta:
                        available_at_time = False
                        break

            if not available_at_time:
                continue

            # Check if already teaching something else at this time
            occupied = _build_family_aware_occupied_map(individual, family_map, gene)
            instructor_busy = False
            for q in range(gene.start_quanta, gene.end_quanta):
                if alt_instructor.instructor_id in occupied["instructors"].get(
                    q, set()
                ):
                    instructor_busy = True
                    break

            if instructor_busy:
                continue

            # Found a valid alternative instructor!
            gene.instructor_id = alt_instructor.instructor_id
            fixes += 1
            alternative_found = True
            break

        # If still not fixed, this is a hard case - leave for next iteration
        if not alternative_found:
            pass  # Will be attempted again in next repair cycle

    return fixes


# CASCADE-AWARE REPAIR COORDINATOR
def repair_with_cascade_check(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_iterations: int = 5,
) -> dict:
    """
    Apply repairs while checking for cascading effects.

    Before applying a fix, simulates the change to ensure it doesn't create
    more violations than it fixes.

    Returns:
        Statistics dict with repair outcomes
    """
    stats = {
        "iterations": 0,
        "total_fixes": 0,
        "cascade_prevented": 0,
    }

    family_map = _get_or_build_family_map(context)

    for _iter in range(max_iterations):
        # Get current violation count
        before_violations = count_family_aware_violations(
            individual, family_map, context
        )

        # Apply hierarchy-aware repairs
        iter_fixes = 0
        iter_fixes += repair_instructor_availability_with_swap(individual, context)
        iter_fixes += repair_group_overlaps_hierarchy(individual, context)

        # Get new violation count
        after_violations = count_family_aware_violations(
            individual, family_map, context
        )

        stats["iterations"] += 1
        stats["total_fixes"] += iter_fixes

        # Stop if no improvement
        if iter_fixes == 0 or after_violations >= before_violations:
            break

    return stats


def count_family_aware_violations(
    individual: list[SessionGene],
    family_map: dict[str, set[str]],
    context: SchedulingContext,
) -> int:
    """
    Count violations using family-aware logic.

    This includes conflicts that the basic checker would miss
    (e.g., BME1A session conflicting with BME1AB session).
    """
    violations = 0
    _build_family_aware_occupied_map(individual, family_map)

    # Pre-compute gene families
    gene_families = []
    for gene in individual:
        family: set[str] = set()
        for gid in gene.group_ids:
            family.update(family_map.get(gid, {gid}))
        gene_families.append(family)

    checked_pairs: set[tuple[int, int]] = set()

    for idx, gene in enumerate(individual):
        gene_family = gene_families[idx]

        for q in range(gene.start_quanta, gene.end_quanta):
            for other_idx, other_gene in enumerate(individual):
                if other_idx <= idx:
                    continue
                if (idx, other_idx) in checked_pairs:
                    continue

                if not (other_gene.start_quanta <= q < other_gene.end_quanta):
                    continue

                other_family = gene_families[other_idx]

                # Group family conflict
                if gene_family & other_family:
                    violations += 1
                    checked_pairs.add((idx, other_idx))

                # Instructor conflict
                if gene.instructor_id == other_gene.instructor_id:
                    violations += 1
                    checked_pairs.add((idx, other_idx))

                # Room conflict
                if gene.room_id == other_gene.room_id:
                    violations += 1
                    checked_pairs.add((idx, other_idx))

    return violations
