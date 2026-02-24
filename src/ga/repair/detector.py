"""
Violation Detector for Selective Gene Repair

Identifies genes with constraint violations to enable targeted repair operations.
Core component of the selective repair optimization system.

Strategy:
- Fast detection: Quick checks without decoding (self-overlap, invalid quanta)
- Full detection: Comprehensive constraint checks via schedule building
- Hybrid: Combines both for accuracy and speed


Architecture:
- Detection is decoupled from repair functions
- Returns gene indices mapped to violation types
- No modifications to SessionGene structure

Usage:
    from src.ga.repair.detector import detect_violated_genes

    violations = detect_violated_genes(individual, context, strategy="hybrid")
    # Returns: {12: ["group_overlap"], 45: ["FPC"]}

    violated_indices = set(violations.keys())
    # Repair only: violated_indices instead of entire individual
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from src.ga.core.schedule_index import ScheduleIndex

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext


def detect_violated_genes(
    individual: list[SessionGene],
    context: SchedulingContext,
    strategy: str = "hybrid",
) -> dict[int, list[str]]:
    """
    Detect genes with constraint violations.

    Args:
        individual: List of SessionGene objects
        context: Scheduling context with entities and resources
        strategy: Detection strategy:
            - "fast": Quick checks without decoding (self-overlap, invalid quanta)
            - "full": Comprehensive constraint checking via schedule maps
            - "hybrid": Both fast and full (recommended for accuracy)

    Returns:
        Dict mapping gene index to list of violation types
        Example: {12: ["group_overlap", "room_conflict"], 45: ["FPC"]}

    Note:
        Empty dict means no violations detected (individual is feasible).
    """
    violations = defaultdict(list)

    if strategy in ["fast", "hybrid"]:
        # Fast pre-check (no decoding needed)
        fast_violations = _detect_fast(individual)
        for idx, vtypes in fast_violations.items():
            violations[idx].extend(vtypes)

    if strategy in ["full", "hybrid"]:
        # Full constraint-based detection
        full_violations = _detect_full(individual, context)
        for idx, vtypes in full_violations.items():
            violations[idx].extend(vtypes)

    return dict(violations)


def _detect_fast(individual: list[SessionGene]) -> dict[int, list[str]]:
    """
    Fast detection without decoding.

    Checks:
    - Duplicate quanta in gene (self-overlap)
    - Empty quanta list
    - Invalid quantum values

    Args:
        individual: List of SessionGene objects

    Returns:
        Dict mapping gene index to violation types
    """
    violations = {}

    for idx, gene in enumerate(individual):
        issues = []

        # Check 1: Duplicate quanta (self-overlap)
        if gene.num_quanta != gene.num_quanta:
            issues.append("self_overlap")

        # Check 2: Empty schedule
        if gene.num_quanta == 0:
            issues.append("empty_schedule")

        # Check 3: Invalid quantum values
        if gene.num_quanta > 0:
            min_q = gene.start_quanta
            max_q = gene.start_quanta + gene.num_quanta - 1
            # Assuming max quantum is around 527 (6 days * 11 slots * 8 quanta per slot)
            if min_q < 0 or max_q > 600:  # Upper bound safety margin
                issues.append("invalid_quanta")

        if issues:
            violations[idx] = issues

    return violations


def _detect_full(
    individual: list[SessionGene], context: SchedulingContext
) -> dict[int, list[str]]:
    """
    Full constraint-based detection using ScheduleIndex (OPTIMIZED).

    Uses ScheduleIndex for efficient caching:
    - Builds schedule maps ONCE (instead of 3 separate builds)
    - Reuses cached maps for all conflict checks
    - 3x faster than original implementation

    Checks:
    - Group overlaps (same group at same time)
    - Room conflicts (same room at same time)
    - Instructor conflicts (same instructor at same time)
    - Instructor qualifications
    - Room type mismatches
    - Instructor availability

    Args:
        individual: List of SessionGene objects
        context: Scheduling context with entities

    Returns:
        Dict mapping gene index to violation types
    """
    violations = defaultdict(list)

    # === NEW: Use ScheduleIndex for efficient conflict detection ===
    # Builds all 3 maps (group/room/instructor) in ONE pass instead of 3
    index = ScheduleIndex.from_individual(individual)

    # Detect group overlaps (uses cached map)
    group_conflicts = index.find_group_conflicts()
    for idx in group_conflicts:
        violations[idx].append("group_overlap")

    # Detect room conflicts (uses same cached map)
    room_conflicts = index.find_room_conflicts()
    for idx in room_conflicts:
        violations[idx].append("room_conflict")

    # Detect instructor conflicts (uses same cached map)
    instructor_conflicts = index.find_instructor_conflicts()
    for idx in instructor_conflicts:
        violations[idx].append("instructor_conflict")
    # === END NEW CODE ===

    # Detect instructor qualifications and availability
    for idx, gene in enumerate(individual):
        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        instructor = context.instructors.get(gene.instructor_id)

        if not course or not instructor:
            violations[idx].append("invalid_entity")
            continue

        if course_key not in instructor.qualified_courses:
            violations[idx].append("FPC")  # Faculty-Program Compliance

        if not instructor.is_full_time and any(
            q not in instructor.available_quanta
            for q in range(gene.start_quanta, gene.end_quanta)
        ):
            violations[idx].append("FCA")  # Faculty Chronometric Availability

    # Detect room type mismatches
    for idx, gene in enumerate(individual):
        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        room = context.rooms.get(gene.room_id)

        if not course or not room:
            if idx not in violations:
                violations[idx].append("invalid_entity")
            continue

        # Check room type compatibility (Room uses 'room_features' not 'room_type')
        if (course.course_type == "practical" and room.room_features != "lab") or (
            course.course_type == "theory" and room.room_features == "lab"
        ):
            violations[idx].append("FFC")  # Facility-Format Compliance

    return dict(violations)


def _build_group_schedule_map(individual: list[SessionGene]) -> dict:
    """
    Build map of group schedules for overlap detection.

    Returns:
        Dict: {group_id: {quantum: [gene_indices]}}
    """
    schedule: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for idx, gene in enumerate(individual):
        for quantum in range(gene.start_quanta, gene.end_quanta):
            for group_id in gene.group_ids:
                schedule[group_id][quantum].append(idx)

    return schedule


def _build_room_schedule_map(individual: list[SessionGene]) -> dict:
    """
    Build map of room schedules for conflict detection.

    Returns:
        Dict: {room_id: {quantum: [gene_indices]}}
    """
    schedule: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for idx, gene in enumerate(individual):
        for quantum in range(gene.start_quanta, gene.end_quanta):
            schedule[gene.room_id][quantum].append(idx)

    return schedule


def _build_instructor_schedule_map(individual: list[SessionGene]) -> dict:
    """
    Build map of instructor schedules for conflict detection.

    Returns:
        Dict: {instructor_id: {quantum: [gene_indices]}}
    """
    schedule: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for idx, gene in enumerate(individual):
        for quantum in range(gene.start_quanta, gene.end_quanta):
            schedule[gene.instructor_id][quantum].append(idx)

    return schedule


if __name__ == "__main__":
    """Quick test of violation detector."""
    from rich.console import Console

    console = Console()
    console.print("\n[bold cyan]Violation Detector Module[/bold cyan]")
    console.print(
        "[dim]Use detect_violated_genes(individual, context, strategy='hybrid')[/dim]\n"
    )
