"""
Local Search Algorithms for Intensive Gene Optimization

Implements neighborhood search strategies for gene-level optimization:
- Greedy local search: Fast hill climbing (first improvement)
- Steepest descent: Exhaustive neighborhood evaluation (best improvement)

Used by Intensive Local Search (IGLS) system for thorough optimization
at key generations or during stagnation.

Architecture:
- Operates on individual SessionGene objects
- Generates neighborhoods (time shifts, room changes)
- Evaluates candidates via constraint checking
- Returns best improvement found

Performance:
- Greedy: ~10-20 evaluations per gene (fast)
- Steepest: ~50-200 evaluations per gene (thorough)
- Can be parallelized at population level

Usage:
    from src.ga.operators.local_search import optimize_gene_greedy

    improved_gene = optimize_gene_greedy(
        gene=original_gene,
        individual=individual,
        context=context,
        max_iterations=10
    )
"""

import random
from typing import Any

from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.room import Room
from src.domain.types import SchedulingContext


def optimize_gene_greedy(
    gene: SessionGene,
    individual: list[SessionGene],
    gene_index: int,
    context: SchedulingContext,
    max_iterations: int = 10,
) -> tuple[SessionGene, int]:
    """
    Greedy local search: hill climbing with first improvement acceptance.

    Explores neighborhood and accepts first improvement found. Fast but may
    miss better alternatives. Suitable for stagnation-triggered repair.

    Args:
        gene: SessionGene to optimize
        individual: Full schedule (for conflict checking)
        gene_index: Index of gene in individual
        context: Scheduling context with entities
        max_iterations: Maximum neighborhood exploration iterations

    Returns:
        Tuple of (best_gene, improvement_delta)
        - best_gene: Improved gene (or original if no improvement)
        - improvement_delta: Reduction in violation count (0 if no improvement)

    Strategy:
        1. Evaluate current gene's constraint violations
        2. Generate random neighborhood samples
        3. Accept FIRST gene that improves fitness
        4. Repeat until no improvement or max_iterations reached
    """
    current_gene = gene
    current_violations = _count_gene_violations(
        current_gene, individual, gene_index, context
    )

    total_improvement = 0
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Generate neighborhood samples (random subset for speed)
        neighbors = _generate_neighborhood(current_gene, context, max_samples=30)

        # Shuffle for random exploration
        random.shuffle(neighbors)

        improved = False
        for neighbor in neighbors:
            neighbor_violations = _count_gene_violations(
                neighbor, individual, gene_index, context
            )

            # Accept FIRST improvement (greedy)
            if neighbor_violations < current_violations:
                improvement = current_violations - neighbor_violations
                current_gene = neighbor
                current_violations = neighbor_violations
                total_improvement += improvement
                improved = True
                break  # Take first improvement

        # Stop if no improvement found
        if not improved:
            break

    return current_gene, total_improvement


def optimize_gene_exhaustive(
    gene: SessionGene,
    individual: list[SessionGene],
    gene_index: int,
    context: SchedulingContext,
    max_neighborhood_size: int = 100,
) -> tuple[SessionGene, int]:
    """
    Steepest descent: exhaustive neighborhood search with best improvement.

    Evaluates ALL neighbors and selects the best one. Thorough but expensive.
    Suitable for fixed-generation intensive optimization (e.g., gen 3, 25).

    Args:
        gene: SessionGene to optimize
        individual: Full schedule (for conflict checking)
        gene_index: Index of gene in individual
        context: Scheduling context with entities
        max_neighborhood_size: Maximum neighbors to evaluate (performance limit)

    Returns:
        Tuple of (best_gene, improvement_delta)
        - best_gene: Best gene found (or original if no improvement)
        - improvement_delta: Reduction in violation count

    Strategy:
        1. Evaluate current gene's constraint violations
        2. Generate full neighborhood (up to max_neighborhood_size)
        3. Evaluate ALL neighbors
        4. Select BEST neighbor (steepest descent)
        5. Repeat until local optimum reached
    """
    current_gene = gene
    current_violations = _count_gene_violations(
        current_gene, individual, gene_index, context
    )

    total_improvement = 0
    max_iter_without_improvement = 3
    no_improvement_count = 0

    while no_improvement_count < max_iter_without_improvement:
        # Generate full neighborhood
        neighbors = _generate_neighborhood(
            current_gene, context, max_samples=max_neighborhood_size
        )

        if not neighbors:
            break

        # Evaluate ALL neighbors
        best_neighbor = None
        best_violations = current_violations

        for neighbor in neighbors:
            neighbor_violations = _count_gene_violations(
                neighbor, individual, gene_index, context
            )

            if neighbor_violations < best_violations:
                best_neighbor = neighbor
                best_violations = neighbor_violations

        # Accept best improvement (if any)
        if best_neighbor is not None:
            improvement = current_violations - best_violations
            current_gene = best_neighbor
            current_violations = best_violations
            total_improvement += improvement
            no_improvement_count = 0  # Reset counter
        else:
            no_improvement_count += 1  # No improvement this iteration

    return current_gene, total_improvement


def _generate_neighborhood(
    gene: SessionGene,
    context: SchedulingContext,
    max_samples: int | None = None,
) -> list[SessionGene]:
    """
    Generate neighborhood of alternative gene assignments.

    Neighborhood types:
    1. Time neighbors: Different time slots (same room/instructor)
    2. Room neighbors: Different rooms (same time/instructor)
    3. Instructor neighbors: Different instructors (same time/room)
    4. Time+Instructor combined: New time AND new instructor

    Args:
        gene: Current gene
        context: Scheduling context
        max_samples: Maximum neighbors to generate (None = all)

    Returns:
        List of neighbor SessionGene objects
    """
    neighbors: list[SessionGene] = []

    # Get course and metadata
    course_key = (gene.course_id, gene.course_type)
    course = context.courses.get(course_key)

    if not course:
        return neighbors

    duration = gene.num_quanta

    # 1. Generate TIME neighbors (shift time slots)
    time_neighbors = _generate_time_neighbors(gene, duration, context)
    neighbors.extend(time_neighbors)

    # 2. Generate ROOM neighbors (change room)
    room_neighbors = _generate_room_neighbors(gene, course, context)
    neighbors.extend(room_neighbors)

    # 3. Generate INSTRUCTOR neighbors (change instructor, same time/room)
    instructor_neighbors = _generate_instructor_neighbors(gene, context)
    neighbors.extend(instructor_neighbors)

    # 4. Generate TIME+INSTRUCTOR combined neighbors (for instructor_time_availability)
    combined_neighbors = _generate_time_instructor_neighbors(gene, duration, context)
    neighbors.extend(combined_neighbors)

    # 5. Sample if too many neighbors
    if max_samples and len(neighbors) > max_samples:
        neighbors = random.sample(neighbors, max_samples)

    return neighbors


def _generate_time_neighbors(
    gene: SessionGene,
    duration: int,
    context: SchedulingContext,
) -> list[SessionGene]:
    """Generate neighbors by shifting time slots."""
    neighbors = []

    # Convert to sorted list if it's a set
    available_quanta = sorted(context.available_quanta)

    # Try all possible time slots of same duration
    for start_idx in range(len(available_quanta) - duration + 1):
        new_quanta = available_quanta[start_idx : start_idx + duration]

        # Skip if same as current
        if new_quanta == gene.get_quanta_list():
            continue

        # Create time neighbor (same room, instructor)
        from src.ga.core.quanta_converter import quanta_list_to_contiguous

        start_q, num_q = quanta_list_to_contiguous(new_quanta)

        neighbor = SessionGene(
            course_id=gene.course_id,
            course_type=gene.course_type,
            group_ids=gene.group_ids,
            instructor_id=gene.instructor_id,
            room_id=gene.room_id,
            start_quanta=start_q,
            num_quanta=num_q,
            co_instructor_ids=list(gene.co_instructor_ids),
        )
        neighbors.append(neighbor)

    return neighbors


def _generate_room_neighbors(
    gene: SessionGene,
    course: Course,
    context: SchedulingContext,
) -> list[SessionGene]:
    """Generate neighbors by changing room."""
    neighbors = []

    # Try all suitable rooms
    for room in context.rooms.values():
        # Skip current room
        if room.room_id == gene.room_id:
            continue

        # Check room suitability
        if not _is_room_suitable(course, room, context, gene.group_ids):
            continue

        # Create room neighbor (same time, instructor)
        neighbor = SessionGene(
            course_id=gene.course_id,
            course_type=gene.course_type,
            group_ids=gene.group_ids,
            instructor_id=gene.instructor_id,
            room_id=room.room_id,
            start_quanta=gene.start_quanta,
            num_quanta=gene.num_quanta,
            co_instructor_ids=list(gene.co_instructor_ids),
        )
        neighbors.append(neighbor)

    return neighbors


def _generate_instructor_neighbors(
    gene: SessionGene,
    context: SchedulingContext,
) -> list[SessionGene]:
    """Generate neighbors by changing instructor (same time/room).

    Targets instructor_time_availability and instructor_exclusivity violations.
    """
    neighbors = []
    course_key = (gene.course_id, gene.course_type)

    for instructor in context.instructors.values():
        if instructor.instructor_id == gene.instructor_id:
            continue
        # Must be qualified
        qualified: set[Any] = getattr(instructor, "qualified_courses", set())
        if course_key not in qualified and gene.course_id not in qualified:
            continue
        # Must be available at current time
        if not instructor.is_full_time and not all(
            q in instructor.available_quanta
            for q in range(gene.start_quanta, gene.end_quanta)
        ):
            continue
        neighbor = SessionGene(
            course_id=gene.course_id,
            course_type=gene.course_type,
            group_ids=gene.group_ids,
            instructor_id=instructor.instructor_id,
            room_id=gene.room_id,
            start_quanta=gene.start_quanta,
            num_quanta=gene.num_quanta,
            co_instructor_ids=_pick_co_instructors(
                gene, instructor.instructor_id, context
            ),
        )
        neighbors.append(neighbor)

    return neighbors


def _generate_time_instructor_neighbors(
    gene: SessionGene,
    duration: int,
    context: SchedulingContext,
    max_combined: int = 15,
) -> list[SessionGene]:
    """Generate combined time+instructor neighbors.

    For genes where the current instructor is unavailable at the current time,
    this finds new (time, instructor) pairs. Sampled to keep neighborhood bounded.
    """

    neighbors: list[SessionGene] = []
    course_key = (gene.course_id, gene.course_type)
    available_quanta = sorted(context.available_quanta)

    # Find qualified instructors (different from current)
    qualified_instructors = []
    for instructor in context.instructors.values():
        if instructor.instructor_id == gene.instructor_id:
            continue
        q_courses: set[Any] = getattr(instructor, "qualified_courses", set())
        if course_key in q_courses or gene.course_id in q_courses:
            qualified_instructors.append(instructor)

    if not qualified_instructors:
        return neighbors

    # Sample some time slots
    possible_starts = []
    for start_idx in range(len(available_quanta) - duration + 1):
        start_q = available_quanta[start_idx]
        end_q = start_q + duration
        if any(q not in context.available_quanta for q in range(start_q, end_q)):
            continue
        if start_q == gene.start_quanta:
            continue
        possible_starts.append(start_q)

    # Sample to limit combinatorial explosion
    sampled_starts = (
        random.sample(possible_starts, min(5, len(possible_starts)))
        if possible_starts
        else []
    )
    sampled_instructors = random.sample(
        qualified_instructors, min(3, len(qualified_instructors))
    )

    for start_q in sampled_starts:
        end_q = start_q + duration
        for instructor in sampled_instructors:
            # Must be available at this new time
            if not instructor.is_full_time and not all(
                q in instructor.available_quanta for q in range(start_q, end_q)
            ):
                continue
            neighbor = SessionGene(
                course_id=gene.course_id,
                course_type=gene.course_type,
                group_ids=gene.group_ids,
                instructor_id=instructor.instructor_id,
                room_id=gene.room_id,
                start_quanta=start_q,
                num_quanta=duration,
                co_instructor_ids=_pick_co_instructors(
                    gene, instructor.instructor_id, context
                ),
            )
            neighbors.append(neighbor)
            if len(neighbors) >= max_combined:
                return neighbors

    return neighbors


def _is_room_suitable(
    course: Course, room: Room, context: SchedulingContext, group_ids: list[str]
) -> bool:
    """Check if room is suitable for course and groups."""
    # Get max group size from all groups
    max_group_size = 0
    for group_id in group_ids:
        group = context.groups.get(group_id)
        if group:
            group_size = getattr(group, "student_count", 30)
            max_group_size = max(max_group_size, group_size)

    if max_group_size == 0:
        max_group_size = 30  # Default fallback

    # Capacity check
    if room.capacity < max_group_size:
        return False

    # Feature compatibility check
    if course.course_type == "practical":
        if room.room_features != "lab":
            return False
    elif course.course_type == "theory" and room.room_features == "lab":
        return False  # Don't use labs for theory

    return True


def _pick_co_instructors(
    gene: SessionGene,
    new_main_instructor: str,
    context: SchedulingContext,
) -> list[str]:
    """Return co-instructor list preserving the structural invariant.

    For practical genes: keep existing co-instructor if valid (different from
    new main), otherwise pick another qualified one.
    For theory genes: always empty.
    """
    if gene.course_type != "practical":
        return []
    # Keep current co-instructor if it differs from new main
    current_co = gene.co_instructor_ids
    if current_co and current_co[0] != new_main_instructor:
        return list(current_co)
    # Pick a new co-instructor
    course_key = (gene.course_id, gene.course_type)
    candidates = [
        iid
        for iid, inst in context.instructors.items()
        if course_key in getattr(inst, "qualified_courses", [])
        and iid != new_main_instructor
    ]
    if candidates:
        return [random.choice(candidates)]
    others = [iid for iid in context.instructors if iid != new_main_instructor]
    return [random.choice(others)] if others else list(current_co)


def _count_gene_violations(
    gene: SessionGene,
    individual: list[SessionGene],
    gene_index: int,
    context: SchedulingContext,
) -> int:
    """
    Count constraint violations for a specific gene.

    Fast violation counting without full fitness evaluation.
    Checks:
    - Group overlaps
    - Room conflicts
    - Instructor conflicts
    - Instructor availability
    - Instructor qualification
    - Room type mismatch

    Args:
        gene: Gene to evaluate
        individual: Full schedule
        gene_index: Index of gene in individual
        context: Scheduling context

    Returns:
        Total violation count (lower is better)
    """
    violations = 0

    # Create temporary individual with candidate gene
    temp_individual = individual.copy()
    temp_individual[gene_index] = gene

    # Check group overlaps
    for other_idx, other_gene in enumerate(temp_individual):
        if other_idx == gene_index:
            continue

        # Check if same group at overlapping time
        if _has_group_overlap(gene, other_gene):
            violations += 1

    # Check room conflicts
    for other_idx, other_gene in enumerate(temp_individual):
        if other_idx == gene_index:
            continue

        if _has_room_conflict(gene, other_gene):
            violations += 1

    # Check instructor conflicts
    for other_idx, other_gene in enumerate(temp_individual):
        if other_idx == gene_index:
            continue

        if _has_instructor_conflict(gene, other_gene):
            violations += 1

    # Check instructor availability (full-time instructors are always available)
    instructor = context.instructors.get(gene.instructor_id)
    if instructor and not instructor.is_full_time:
        for q in range(gene.start_quanta, gene.end_quanta):
            if q not in instructor.available_quanta:
                violations += 1

    # Check instructor qualification
    course_key = (gene.course_id, gene.course_type)
    course = context.courses.get(course_key)
    if instructor and course and course_key not in instructor.qualified_courses:
        violations += 2  # Higher weight for qualification

    # Check room type mismatch
    room = context.rooms.get(gene.room_id)
    if course and room and not _is_room_suitable(course, room, context, gene.group_ids):
        violations += 1

    return violations


def _has_group_overlap(gene1: SessionGene, gene2: SessionGene) -> bool:
    """Check if two genes have group overlap at same time."""
    # Check for common groups
    common_groups = set(gene1.group_ids) & set(gene2.group_ids)
    if not common_groups:
        return False

    # Check for time overlap using efficient range comparison
    # Two ranges overlap if: start1 < end2 AND start2 < end1
    return (
        gene1.start_quanta < gene2.end_quanta and gene2.start_quanta < gene1.end_quanta
    )


def _has_room_conflict(gene1: SessionGene, gene2: SessionGene) -> bool:
    """Check if two genes use same room at same time."""
    if gene1.room_id != gene2.room_id:
        return False

    # Check for time overlap using efficient range comparison
    return (
        gene1.start_quanta < gene2.end_quanta and gene2.start_quanta < gene1.end_quanta
    )


def _has_instructor_conflict(gene1: SessionGene, gene2: SessionGene) -> bool:
    """Check if two genes use same instructor at same time."""
    if gene1.instructor_id != gene2.instructor_id:
        return False

    # Check for time overlap using efficient range comparison
    return (
        gene1.start_quanta < gene2.end_quanta and gene2.start_quanta < gene1.end_quanta
    )


if __name__ == "__main__":
    """Quick test of local search module."""
    from rich.console import Console

    console = Console()
    console.print("\n[bold cyan]Local Search Module[/bold cyan]")
    console.print("[dim]Provides greedy and exhaustive gene optimization[/dim]")
    console.print("\nAvailable functions:")
    console.print("  • optimize_gene_greedy() - Fast hill climbing")
    console.print("  • optimize_gene_exhaustive() - Thorough steepest descent")
    console.print()
