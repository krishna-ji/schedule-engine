"""
Heuristic-based repair for LNS (Large Neighborhood Search).

Provides fast, greedy repair strategies as an alternative to CP-SAT.
"""

import logging
import random
import time

from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.ga.core.evaluator import evaluate

logger = logging.getLogger(__name__)


def repair_with_heuristic(
    conflicted_sessions: list[SessionGene],
    partial_schedule: list[SessionGene],
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
    max_iterations: int = 500,
    time_limit: float = 5.0,
) -> list[SessionGene] | None:
    """
    Repair conflicted sessions using greedy heuristic + local search.

    Strategy:
    1. Greedy assignment: place sessions in feasible time/room slots
    2. If greedy fails, run local search (random swaps/moves)
    3. Accept improvements or use simulated annealing acceptance

    Args:
        conflicted_sessions: Sessions to repair
        partial_schedule: Fixed sessions (already scheduled)
        courses: Course dictionary
        instructors: Instructor dictionary
        groups: Group dictionary
        rooms: Room dictionary
        max_iterations: Max local search iterations
        time_limit: Time limit in seconds

    Returns:
        List of repaired SessionGene objects, or None if repair fails
    """
    start_time = time.time()

    # Phase 1: Greedy assignment
    logger.debug(
        f"Heuristic repair: attempting greedy assignment for {len(conflicted_sessions)} sessions"
    )

    repaired = _greedy_assign(
        conflicted_sessions, partial_schedule, courses, instructors, groups, rooms
    )

    if repaired is not None:
        elapsed = time.time() - start_time
        logger.info(f"Heuristic repair: GREEDY SUCCESS (time={elapsed:.2f}s)")
        return repaired

    # Phase 2: Local search with random restarts
    logger.debug("Heuristic repair: greedy failed, trying local search")

    repaired = _local_search_repair(
        conflicted_sessions,
        partial_schedule,
        courses,
        instructors,
        groups,
        rooms,
        max_iterations,
        time_limit - (time.time() - start_time),
    )

    elapsed = time.time() - start_time
    if repaired is not None:
        logger.info(f"Heuristic repair: LOCAL SEARCH SUCCESS (time={elapsed:.2f}s)")
        return repaired
    logger.warning(f"Heuristic repair: FAILED (time={elapsed:.2f}s)")
    return None


def _greedy_assign(
    conflicted_sessions: list[SessionGene],
    partial_schedule: list[SessionGene],
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
) -> list[SessionGene] | None:
    """
    Greedy assignment: place each session in first feasible time/room slot.

    Order sessions by difficulty (fewest options first).
    """
    # Sort by difficulty (smallest domain first)
    ordered = _order_by_difficulty(
        conflicted_sessions, instructors, groups, rooms, courses
    )

    assigned: list[SessionGene] = []
    current_schedule = partial_schedule + assigned

    for session in ordered:
        # Try to find first feasible assignment
        candidate = _find_first_feasible(
            session, current_schedule, instructors, groups, rooms, courses
        )

        if candidate is None:
            return None  # Greedy failed

        assigned.append(candidate)
        current_schedule = partial_schedule + assigned

    return assigned


def _order_by_difficulty(
    sessions: list[SessionGene],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
    courses: dict[tuple, Course],
) -> list[SessionGene]:
    """Order sessions by difficulty (fewest options first)."""

    def difficulty_score(session: SessionGene) -> tuple[int, int, int]:
        # Count available times (instructor & groups intersection)
        instructor = instructors[session.instructor_id]
        common_quanta = set(instructor.available_quanta)
        for gid in session.group_ids:
            group = groups[gid]
            common_quanta &= set(group.available_quanta)

        # Count suitable rooms
        course_key = (session.course_id, session.course_type)
        course = courses[course_key]
        total_size = sum(groups[gid].student_count for gid in session.group_ids)
        suitable_rooms = sum(
            1
            for room in rooms.values()
            if room.capacity >= total_size
            and room.is_suitable_for_course_type(
                course.required_room_features,
                getattr(course, "specific_lab_features", None),
            )
        )

        # Return tuple: (time_options, room_options, group_count)
        # Smaller = more difficult (sort ascending)
        return (len(common_quanta), suitable_rooms, len(session.group_ids))

    return sorted(sessions, key=difficulty_score)


def _find_first_feasible(
    session: SessionGene,
    current_schedule: list[SessionGene],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
    courses: dict[tuple, Course],
) -> SessionGene | None:
    """Find first feasible time/room assignment for a session."""

    instructor = instructors[session.instructor_id]
    course_key = (session.course_id, session.course_type)
    course = courses[course_key]

    # Get common available quanta
    common_quanta = set(instructor.available_quanta)
    for gid in session.group_ids:
        group = groups[gid]
        common_quanta &= set(group.available_quanta)

    # Filter out occupied quanta
    occupied_instructor: set[int] = set()
    occupied_groups: dict[str, set[int]] = {gid: set() for gid in session.group_ids}
    occupied_rooms: dict[str, set[int]] = {}

    for fixed in current_schedule:
        if fixed.instructor_id == session.instructor_id:
            # Add all quanta in the contiguous block to occupied set
            occupied_instructor.update(range(fixed.start_quanta, fixed.end_quanta))
        for gid in session.group_ids:
            if gid in fixed.group_ids:
                occupied_groups[gid].update(range(fixed.start_quanta, fixed.end_quanta))
        if fixed.room_id not in occupied_rooms:
            occupied_rooms[fixed.room_id] = set()
        occupied_rooms[fixed.room_id].update(
            range(fixed.start_quanta, fixed.end_quanta)
        )

    # Try start times
    session_duration = session.num_quanta
    for start_q in sorted(common_quanta):
        span = list(range(start_q, start_q + session_duration))
        if not all(q in common_quanta for q in span):
            continue
        # Check not occupied
        if any(q in occupied_instructor for q in span):
            continue
        if any(
            any(q in occupied_groups[gid] for q in span) for gid in session.group_ids
        ):
            continue

        # Try rooms
        total_size = sum(groups[gid].student_count for gid in session.group_ids)
        for room_id, room in rooms.items():
            if room.capacity >= total_size and room.is_suitable_for_course_type(
                course.required_room_features,
                getattr(course, "specific_lab_features", None),
            ):
                # Check room not occupied
                if room_id in occupied_rooms and any(
                    q in occupied_rooms[room_id] for q in span
                ):
                    continue

                # Found feasible assignment
                return SessionGene(
                    course_id=session.course_id,
                    course_type=session.course_type,
                    instructor_id=session.instructor_id,
                    group_ids=session.group_ids,
                    room_id=room_id,
                    start_quanta=span[0],
                    num_quanta=len(span),
                    co_instructor_ids=list(getattr(session, "co_instructor_ids", [])),
                )

    return None


def _local_search_repair(
    conflicted_sessions: list[SessionGene],
    partial_schedule: list[SessionGene],
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
    max_iterations: int,
    time_limit: float,
) -> list[SessionGene] | None:
    """
    Local search repair with random moves and acceptance.

    Start from random/partial assignment, try random moves,
    accept if improves or use simulated annealing.
    """
    start_time = time.time()

    # Initialize with random assignment (best effort)
    current: list[SessionGene] = []
    for session in conflicted_sessions:
        candidate = _random_assignment(
            session, partial_schedule + current, instructors, groups, rooms, courses
        )
        if candidate:
            current.append(candidate)
        else:
            # Use original (may be infeasible)
            current.append(session)

    # Evaluate current
    full_individual = partial_schedule + current
    current_fitness = evaluate(full_individual, courses, instructors, groups, rooms)

    best = current
    best_fitness = current_fitness

    temperature = 10.0  # Simulated annealing temperature

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break

        # Propose random move
        neighbor = _propose_move(
            current, partial_schedule, instructors, groups, rooms, courses
        )

        if neighbor is None:
            continue

        # Evaluate neighbor
        full_neighbor = partial_schedule + neighbor
        neighbor_fitness = evaluate(full_neighbor, courses, instructors, groups, rooms)

        # Accept if better or via simulated annealing
        delta_hard = neighbor_fitness[0] - current_fitness[0]
        delta_soft = neighbor_fitness[1] - current_fitness[1]

        accept = False
        if delta_hard < 0 or (
            delta_hard == 0 and delta_soft < 0
        ):  # Hard constraint improvement
            accept = True
        elif temperature > 0.1:  # Simulated annealing
            prob = min(1.0, 2.718281828 ** (-abs(delta_hard) / temperature))
            if random.random() < prob:
                accept = True

        if accept:
            current = neighbor
            current_fitness = neighbor_fitness
            if current_fitness[0] < best_fitness[0] or (
                current_fitness[0] == best_fitness[0]
                and current_fitness[1] < best_fitness[1]
            ):
                best = current
                best_fitness = current_fitness

        # Decay temperature
        temperature *= 0.995

        # Early exit if feasible
        if best_fitness[0] == 0:
            logger.debug(
                f"Local search found feasible solution at iteration {iteration}"
            )
            return best

    # Return best found (even if still infeasible - caller will check)
    logger.debug(
        f"Local search completed: best_fitness=({best_fitness[0]}, {best_fitness[1]:.1f})"
    )
    return best


def _random_assignment(
    session: SessionGene,
    current_schedule: list[SessionGene],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
    courses: dict[tuple, Course],
) -> SessionGene | None:
    """Random feasible assignment (best effort)."""
    instructor = instructors[session.instructor_id]
    common_quanta = set(instructor.available_quanta)
    for gid in session.group_ids:
        common_quanta &= set(groups[gid].available_quanta)

    if not common_quanta:
        return None

    # Pick random start
    candidates = sorted(common_quanta)
    random.shuffle(candidates)

    session_duration = session.num_quanta
    for start_q in candidates:
        span = list(range(start_q, start_q + session_duration))
        if all(q in common_quanta for q in span):
            # Pick random room
            course_key = (session.course_id, session.course_type)
            course = courses[course_key]
            total_size = sum(groups[gid].student_count for gid in session.group_ids)
            suitable = [
                rid
                for rid, room in rooms.items()
                if room.capacity >= total_size
                and room.is_suitable_for_course_type(
                    course.required_room_features,
                    getattr(course, "specific_lab_features", None),
                )
            ]
            if suitable:
                return SessionGene(
                    course_id=session.course_id,
                    course_type=session.course_type,
                    instructor_id=session.instructor_id,
                    group_ids=session.group_ids,
                    room_id=random.choice(suitable),
                    start_quanta=start_q,
                    num_quanta=session_duration,
                    co_instructor_ids=list(getattr(session, "co_instructor_ids", [])),
                )

    return None


def _propose_move(
    current: list[SessionGene],
    partial_schedule: list[SessionGene],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
    courses: dict[tuple, Course],
) -> list[SessionGene] | None:
    """Propose a random move (time shift or room swap)."""
    if not current:
        return None

    neighbor = list(current)
    idx = random.randint(0, len(neighbor) - 1)
    session = neighbor[idx]

    # Try random time shift or room swap
    if random.random() < 0.5:
        # Time shift
        new_session = _random_assignment(
            session,
            partial_schedule + neighbor[:idx] + neighbor[idx + 1 :],
            instructors,
            groups,
            rooms,
            courses,
        )
        if new_session:
            neighbor[idx] = new_session
            return neighbor
    else:
        # Room swap
        course_key = (session.course_id, session.course_type)
        course = courses[course_key]
        total_size = sum(groups[gid].student_count for gid in session.group_ids)
        suitable = [
            rid
            for rid, room in rooms.items()
            if room.capacity >= total_size
            and room.is_suitable_for_course_type(
                course.required_room_features,
                getattr(course, "specific_lab_features", None),
            )
            and rid != session.room_id
        ]
        if suitable:
            # new_room = random.choice(suitable)  # Unused - directly use in SessionGene
            neighbor[idx] = SessionGene(
                course_id=session.course_id,
                course_type=session.course_type,
                instructor_id=session.instructor_id,
                group_ids=session.group_ids,
                room_id=session.room_id,
                start_quanta=session.start_quanta,
                num_quanta=session.num_quanta,
                co_instructor_ids=list(getattr(session, "co_instructor_ids", [])),
            )
            return neighbor

    return None
