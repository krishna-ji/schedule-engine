"""
Constraint-Guided Mutation Operator

PHASE 2: Priority 2 Enhancement

Targets sessions with constraint violations for focused repair.
Instead of random mutation, identifies problematic sessions and mutates those.

Strategy:
1. Decode individual to CourseSession objects
2. Identify sessions causing hard violations
3. Mutate violating sessions preferentially (80% probability)
4. Fallback to random mutation (20% for diversity)

Expected Impact: 20-30% faster convergence to zero violations.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

from src.io.decoder import decode_individual

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.session import CourseSession
    from src.domain.types import Individual, SchedulingContext
    from src.ga.core.domain_store import GeneDomainStore
    from src.ga.core.usage_tracker import UsageTracker


def constraint_guided_mutation(
    individual: Individual, context: SchedulingContext
) -> tuple[Individual, dict[str, int]]:
    """
    Mutate genes corresponding to sessions with violations.

    Now uses UsageTracker + GeneDomainStore for spreading-aware selection
    instead of random.choice().

    Args:
        individual: List of SessionGene
        context: SchedulingContext with courses, groups, instructors, rooms

    Returns:
        Tuple of (modified individual, mutation stats dict)
    """
    from src.domain.gene import get_time_system
    from src.ga.core.domain_store import GeneDomainStore
    from src.ga.core.usage_tracker import UsageTracker

    # Build spreading infrastructure
    qts = get_time_system()
    domain_store = GeneDomainStore(context, qts)
    domain_store.build_domains(individual)
    tracker = UsageTracker()
    tracker.build_from_individual(individual)

    # Decode to identify violations
    decoded = decode_individual(
        individual,
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    # Find sessions with violations (returns {gene_idx: violation_type})
    violations = _find_violating_sessions(decoded, context)

    # Repair multiple violating genes per call (not just 1)
    # This makes mutation strong enough to overcome crossover disruption
    violating_indices = list(violations.keys())
    max_repairs = min(len(violating_indices), max(3, len(violating_indices) // 5))
    targeted = 0
    rand_mut = 0

    if violating_indices:
        # Shuffle to avoid always fixing the same genes first
        repair_targets = random.sample(violating_indices, max_repairs)
        for target_idx in repair_targets:
            if random.random() < 0.8:
                vtype = violations[target_idx]
                _mutate_session_spreading(
                    individual,
                    target_idx,
                    context,
                    domain_store,
                    tracker,
                    force_component=vtype,
                )
                targeted += 1
            else:
                # Random mutation for diversity
                rand_idx = random.randint(0, len(individual) - 1)
                _mutate_session_spreading(
                    individual, rand_idx, context, domain_store, tracker
                )
                rand_mut += 1
    # No violations found — random mutation for diversity
    elif len(individual) > 0:
        target_idx = random.randint(0, len(individual) - 1)
        _mutate_session_spreading(
            individual, target_idx, context, domain_store, tracker
        )
        rand_mut = 1

    return individual, {"targeted_mutations": targeted, "random_mutations": rand_mut}


def _find_violating_sessions(
    decoded_sessions: list[CourseSession], context: SchedulingContext
) -> dict[int, str]:
    """
    Identify indices of sessions causing hard constraint violations.

    Uses O(N) index-based detection instead of O(N²) all-pairs scanning.

    Checks:
    - Group overlaps (double-booking)      → "time"
    - Room conflicts (double-booking)       → "room"
    - Instructor conflicts (double-booking) → "time"
    - Instructor qualification mismatches   → "instructor"
    - Room suitability (type + features)    → "room"
    - Instructor time availability          → "time"

    Returns:
        Dict mapping gene index → violation category ("time", "room", "instructor")
    """
    from collections import defaultdict

    violations: dict[int, str] = {}

    # ── O(N) indexed clash detection ──────────────────────────────
    # Build occupancy maps: entity → quantum → list[session_idx]
    group_occ: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    room_occ: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    inst_occ: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for idx, session in enumerate(decoded_sessions):
        for q in session.session_quanta:
            for gid in session.group_ids:
                group_occ[gid][q].append(idx)
            room_occ[session.room_id][q].append(idx)
            inst_occ[session.instructor_id][q].append(idx)

    # Detect clashes: any quantum with >1 session is a conflict
    for q_map in group_occ.values():
        for indices in q_map.values():
            if len(indices) > 1:
                for i in indices:
                    violations.setdefault(i, "time")

    for q_map in room_occ.values():
        for indices in q_map.values():
            if len(indices) > 1:
                for i in indices:
                    violations.setdefault(i, "room")

    for q_map in inst_occ.values():
        for indices in q_map.values():
            if len(indices) > 1:
                for i in indices:
                    violations.setdefault(i, "time")

    # ── O(N) per-gene checks ─────────────────────────────────────
    for idx, session in enumerate(decoded_sessions):
        if idx in violations:
            continue

        # Instructor qualification
        if not _is_instructor_qualified(session, context):
            violations[idx] = "instructor"
            continue

        # Room suitability (type + specific features)
        if not _is_room_suitable(session, context):
            violations[idx] = "room"
            continue

        # Instructor time availability
        if not _is_instructor_available(session, context):
            violations[idx] = "time"

    return violations


def _is_instructor_qualified(
    session: CourseSession, context: SchedulingContext
) -> bool:
    """Check if instructor is qualified to teach the course."""
    course_key = (session.course_id, session.course_type)
    course = context.courses.get(course_key)
    if not course:
        return True  # Unknown course, assume OK

    # If course has no qualification requirements, anyone can teach
    if not course.qualified_instructor_ids:
        return True

    # Check if instructor is in the qualified list
    return session.instructor_id in course.qualified_instructor_ids


def _is_room_suitable(session: CourseSession, context: SchedulingContext) -> bool:
    """Check if the assigned room is suitable for the course type and features."""
    from src.utils.room_compatibility import is_room_suitable_for_course

    course_key = (
        (session.course_id, session.course_type)
        if isinstance(session.course_id, str)
        else session.course_id
    )
    course = context.courses.get(course_key)
    if not course:
        return True

    room = context.rooms.get(session.room_id)
    if not room:
        return True

    required = getattr(course, "required_room_features", "lecture")
    room_type = getattr(room, "room_features", "lecture")
    req_str = (required if isinstance(required, str) else str(required)).lower().strip()
    room_str = (
        (room_type if isinstance(room_type, str) else str(room_type)).lower().strip()
    )
    course_lab_feats = getattr(course, "specific_lab_features", None)
    room_spec_feats = getattr(room, "specific_features", None)

    return is_room_suitable_for_course(
        req_str, room_str, course_lab_feats, room_spec_feats
    )


def _is_instructor_available(
    session: CourseSession, context: SchedulingContext
) -> bool:
    """Check if instructor is available during the session's time slots."""
    instructor = context.instructors.get(session.instructor_id)
    if not instructor:
        return True

    available = getattr(instructor, "available_quanta", None)
    if not available:
        return True  # No availability info means always available

    available_set = set(available) if not isinstance(available, set) else available
    return all(q in available_set for q in session.session_quanta)


def _has_group_overlap(
    session: CourseSession, all_sessions: list[CourseSession], current_idx: int
) -> bool:
    """Check if group has overlapping sessions."""
    for idx, other in enumerate(all_sessions):
        if idx == current_idx:
            continue

        # Check if any group in session overlaps with any group in other
        # group_ids is always a list[str] per SessionGene definition
        session_groups = session.group_ids
        other_groups = other.group_ids

        # Same group and overlapping time?
        if (set(session_groups) & set(other_groups)) and (
            set(session.session_quanta) & set(other.session_quanta)
        ):
            return True
    return False


def _has_room_conflict(
    session: CourseSession, all_sessions: list[CourseSession], current_idx: int
) -> bool:
    """Check if room is double-booked."""
    for idx, other in enumerate(all_sessions):
        if idx == current_idx:
            continue

        # Same room and overlapping time?
        if session.room_id == other.room_id and (
            set(session.session_quanta) & set(other.session_quanta)
        ):
            return True
    return False


def _has_instructor_conflict(
    session: CourseSession, all_sessions: list[CourseSession], current_idx: int
) -> bool:
    """Check if instructor is double-booked."""
    for idx, other in enumerate(all_sessions):
        if idx == current_idx:
            continue

        # Same instructor and overlapping time?
        if session.instructor_id == other.instructor_id and (
            set(session.session_quanta) & set(other.session_quanta)
        ):
            return True
    return False


def _mutate_session_spreading(
    individual: list[SessionGene],
    gene_idx: int,
    context: SchedulingContext,
    domain_store: GeneDomainStore,
    tracker: UsageTracker,
    force_component: str | None = None,
) -> None:
    """Mutate a gene in-place using domain buckets + usage-aware spreading.

    Args:
        force_component: If set, forces mutation of this component:
            "time"       → change time (+ room secondarily)
            "room"       → change room
            "instructor" → change instructor (+ time secondarily)
            None         → weighted random (default)

    Default strategy (weighted random):
    - 40% chance: change time (least-used conflict-free)
    - 30% chance: change room (free at current time, least-loaded)
    - 20% chance: change instructor (least-loaded qualified)
    - 10% chance: change ALL three (aggressive)
    """
    gene = individual[gene_idx]
    domain = domain_store.get_domain(gene_idx)

    # When force_component is set, bias heavily toward that component
    if force_component == "room":
        mutation_type = 0.5  # → room branch
    elif force_component == "time":
        mutation_type = 0.95  # → change ALL (time + room) for clash resolution
    elif force_component == "instructor":
        mutation_type = 0.75  # → instructor branch
    else:
        mutation_type = random.random()

    # Remove gene from tracker before re-assigning
    tracker.remove_gene(gene)

    # --- Build blocked set for time narrowing ---
    blocked: set[int] = set()
    family_map = context.family_map or {}
    expanded_groups: set[str] = set(gene.group_ids)
    for gid in gene.group_ids:
        expanded_groups.update(family_map.get(gid, set()))

    for j, other in enumerate(individual):
        if j == gene_idx:
            continue
        if expanded_groups & set(other.group_ids):
            for q in range(other.start_quanta, other.start_quanta + other.num_quanta):
                blocked.add(q)
        if other.instructor_id == gene.instructor_id:
            for q in range(other.start_quanta, other.start_quanta + other.num_quanta):
                blocked.add(q)

    new_start = gene.start_quanta
    new_room = gene.room_id
    new_instructor = gene.instructor_id

    def _pick_time() -> int:
        # Rebuild instructor-blocked quanta for the (possibly new) instructor
        time_blocked = set()
        for j, other in enumerate(individual):
            if j == gene_idx:
                continue
            if expanded_groups & set(other.group_ids):
                for q in range(
                    other.start_quanta, other.start_quanta + other.num_quanta
                ):
                    time_blocked.add(q)
            if other.instructor_id == new_instructor:
                for q in range(
                    other.start_quanta, other.start_quanta + other.num_quanta
                ):
                    time_blocked.add(q)
        free_starts = domain_store.narrow_time_domain(gene_idx, time_blocked)
        # Also narrow by instructor native availability
        free_starts = domain_store.instructor_available_starts(
            gene_idx, new_instructor, free_starts if free_starts else None
        )
        if not free_starts:
            free_starts = domain.valid_starts
        picked = tracker.pick_least_used_start(free_starts, gene.num_quanta, top_k=5)
        return picked if picked is not None else gene.start_quanta

    def _pick_room(start: int) -> str:
        room_free = [
            r
            for r in domain.rooms
            if all(
                tracker.room_load.get(r, Counter()).get(q, 0) == 0
                for q in range(start, start + gene.num_quanta)
            )
        ]
        pool = room_free if room_free else domain.rooms
        picked = tracker.pick_least_used_room(
            pool if pool else [gene.room_id], start, gene.num_quanta
        )
        return picked if picked else gene.room_id

    def _pick_instructor(for_start: int | None = None) -> str:
        """Pick least-loaded instructor, preferring those available at for_start."""
        if for_start is not None:
            # Prefer instructors available at the target time
            avail_insts = [
                iid
                for iid in domain.instructors
                if all(
                    q
                    in domain_store._instructor_available.get(
                        iid, domain_store._available_set
                    )
                    for q in range(for_start, for_start + gene.num_quanta)
                )
            ]
            if avail_insts:
                picked = tracker.pick_least_used_instructor(avail_insts)
                if picked:
                    return picked
        picked = tracker.pick_least_used_instructor(domain.instructors)
        return str(picked) if picked else gene.instructor_id

    if mutation_type < 0.4:
        new_start = _pick_time()
    elif mutation_type < 0.7:
        new_room = _pick_room(gene.start_quanta)
    elif mutation_type < 0.9:
        new_instructor = _pick_instructor(for_start=gene.start_quanta)
    else:
        new_instructor = _pick_instructor()
        new_start = _pick_time()
        new_room = _pick_room(new_start)

    gene.instructor_id = new_instructor
    gene.start_quanta = new_start
    gene.room_id = new_room
    gene.__post_init__()  # Re-validate day boundaries after in-place mutation

    tracker.add_gene(gene)


def _mutate_session(
    gene: SessionGene,
    context: SchedulingContext,
    individual: list[SessionGene] | None = None,
) -> None:
    """Legacy mutate (kept for backward compat). Uses random.choice."""
    mutation_type = random.random()

    available_quanta_list = list(context.available_quanta)
    blocked: set[int] = set()
    if individual is not None:
        gene_groups = set(gene.group_ids)
        for other in individual:
            if other is gene:
                continue
            if gene_groups & set(other.group_ids):
                for q in range(
                    other.start_quanta, other.start_quanta + other.num_quanta
                ):
                    blocked.add(q)
            if other.instructor_id == gene.instructor_id:
                for q in range(
                    other.start_quanta, other.start_quanta + other.num_quanta
                ):
                    blocked.add(q)

    free_quanta = [q for q in available_quanta_list if q not in blocked]
    time_pool = (
        free_quanta if len(free_quanta) >= gene.num_quanta else available_quanta_list
    )

    if mutation_type < 0.4:
        num_quanta = gene.num_quanta
        if num_quanta > 0 and len(time_pool) >= num_quanta:
            valid_starts = [
                q
                for q in time_pool
                if all((q + i) in time_pool for i in range(num_quanta))
            ]
            if valid_starts:
                gene.start_quanta = random.choice(valid_starts)

    elif mutation_type < 0.7:
        if context.rooms:
            from src.ga.operators.mutation import find_suitable_rooms_for_course

            primary_group = gene.group_ids[0] if gene.group_ids else ""
            suitable = find_suitable_rooms_for_course(
                gene.course_id, gene.course_type, primary_group, context
            )
            if suitable:
                gene.room_id = random.choice(suitable)
            # STRICT: never assign unsuitable room — keep current if none suitable

    elif mutation_type < 0.9:
        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        if course and course.qualified_instructor_ids:
            gene.instructor_id = random.choice(course.qualified_instructor_ids)
        # STRICT: never assign unqualified — keep current if no qualified found

    else:
        num_quanta = gene.num_quanta
        if num_quanta > 0 and len(time_pool) >= num_quanta:
            valid_starts = [
                q
                for q in time_pool
                if all((q + i) in time_pool for i in range(num_quanta))
            ]
            if valid_starts:
                gene.start_quanta = random.choice(valid_starts)

        if context.rooms:
            from src.ga.operators.mutation import find_suitable_rooms_for_course

            primary_group = gene.group_ids[0] if gene.group_ids else ""
            suitable = find_suitable_rooms_for_course(
                gene.course_id, gene.course_type, primary_group, context
            )
            if suitable:
                gene.room_id = random.choice(suitable)
            # STRICT: never assign unsuitable room — keep current if none suitable
