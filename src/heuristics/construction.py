"""
Construction Heuristics - Build Schedules Greedily

Provides greedy construction algorithms that build schedules from scratch
by intelligently ordering session assignments based on domain knowledge.

Construction heuristics are useful for:
1. Initial population generation (better than random)
2. Repair operations (rebuild portions of schedule)
3. Warm-starting the GA with feasible solutions

Strategies:
1. Largest Degree First: Schedule most conflicting courses first
2. Most Constrained First: Schedule sessions with fewest options first
3. Earliest Deadline First: Prioritize sessions by scheduling urgency

Architecture:
- Decorator-based registration with @construction_heuristic
- Returns fully constructed individual (List[SessionGene])
- Does not modify existing individuals (pure construction)
- Can be used standalone or integrated into GA population initialization

Usage:
    from src.heuristics.construction import largest_degree_first

    # Build a schedule using largest degree first
    individual = largest_degree_first(context)
"""

from typing import List, Dict, Tuple, Set
import random
from collections import defaultdict

from src.ga.sessiongene import SessionGene
from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.heuristics.registry import construction_heuristic


# ============================================================================
# LARGEST DEGREE FIRST (Schedule most conflicting courses first)
# ============================================================================


@construction_heuristic(
    name="largest_degree_first",
    description="Schedule courses with most conflicts/constraints first (graph coloring heuristic)",
    priority=1,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=False,
)
def largest_degree_first(context: SchedulingContext) -> List[SessionGene]:
    """
    Build schedule by scheduling most conflicting courses first.

    Based on graph coloring heuristic: courses with more conflicts (edges)
    are scheduled first because they have fewer valid time slots.

    Conflict degree measured by:
    - Number of shared instructors with other courses
    - Number of shared student groups with other courses
    - Room type constraints (lab vs classroom)

    Algorithm:
    1. Calculate conflict degree for each course
    2. Sort courses by degree (descending)
    3. For each course (high to low degree):
        - Find earliest valid time slot
        - Assign room and instructor
        - Add SessionGene to schedule

    Args:
        context: Scheduling context with entities and available times

    Returns:
        List[SessionGene] representing a complete schedule
    """
    time_system = QuantumTimeSystem(context.config)
    individual = []

    # Calculate conflict degrees
    course_degrees = _calculate_conflict_degrees(context)

    # Sort courses by degree (descending - most conflicts first)
    sorted_courses = sorted(course_degrees.items(), key=lambda x: x[1], reverse=True)

    # Track assignments for conflict checking
    assigned_times = defaultdict(set)  # {entity_id: {time_quanta}}
    assigned_rooms = defaultdict(set)  # {room_id: {time_quanta}}

    for course_id, _degree in sorted_courses:
        course = context.courses[course_id]

        # Build sessions for this course
        for session_idx in range(course.sessions_per_week):
            # Find valid time slot
            time_quantum = _find_earliest_valid_time(
                context,
                course,
                time_system,
                assigned_times,
                assigned_rooms,
            )

            if time_quantum is None:
                # No valid time found - assign random (will be repaired later)
                time_quantum = random.choice(time_system.available_quanta)

            # Find suitable room
            room_id = _find_suitable_room(context, course, time_quantum, assigned_rooms)

            # Select qualified instructor
            instructor_id = _select_qualified_instructor(
                context, course, time_quantum, assigned_times
            )

            # Create gene
            gene = SessionGene(
                course_id=course_id,
                group_ids=course.group_ids,
                time_quantum=time_quantum,
                duration_quanta=course.duration_quanta,
                room_id=room_id,
                instructor_id=instructor_id,
            )

            individual.append(gene)

            # Update assignments
            for q in range(time_quantum, time_quantum + course.duration_quanta):
                for group_id in course.group_ids:
                    assigned_times[group_id].add(q)
                assigned_times[instructor_id].add(q)
                assigned_rooms[room_id].add(q)

    return individual


# ============================================================================
# MOST CONSTRAINED FIRST (Schedule sessions with fewest options first)
# ============================================================================


@construction_heuristic(
    name="most_constrained_first",
    description="Schedule sessions with fewest valid time slots first (minimum remaining values)",
    priority=2,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=False,
)
def most_constrained_first(context: SchedulingContext) -> List[SessionGene]:
    """
    Build schedule by scheduling most constrained sessions first.

    Based on Minimum Remaining Values (MRV) heuristic from CSP:
    sessions with fewer valid options are scheduled first to avoid
    backtracking later.

    Constraint level measured by:
    - Instructor availability restrictions
    - Room availability (lab requirements)
    - Existing assignments reducing options

    Algorithm:
    1. Calculate initial constraint levels for all sessions
    2. While sessions remain:
        - Pick session with fewest valid time slots
        - Find best time slot for this session
        - Assign and update remaining constraints

    Args:
        context: Scheduling context with entities and available times

    Returns:
        List[SessionGene] representing a complete schedule
    """
    time_system = QuantumTimeSystem(context.config)
    individual = []

    # Track assignments
    assigned_times = defaultdict(set)
    assigned_rooms = defaultdict(set)

    # Build list of all sessions to schedule
    sessions_to_schedule = []
    for course_id, course in context.courses.items():
        for session_idx in range(course.sessions_per_week):
            sessions_to_schedule.append((course_id, session_idx, course))

    # Schedule sessions in constraint order
    while sessions_to_schedule:
        # Find most constrained session
        most_constrained = None
        min_options = float("inf")

        for course_id, session_idx, course in sessions_to_schedule:
            # Count valid time slots
            valid_slots = _count_valid_time_slots(
                context,
                course,
                time_system,
                assigned_times,
                assigned_rooms,
            )

            if valid_slots < min_options:
                min_options = valid_slots
                most_constrained = (course_id, session_idx, course)

        # Remove from pending list
        sessions_to_schedule.remove(most_constrained)
        course_id, session_idx, course = most_constrained

        # Find best time slot
        time_quantum = _find_earliest_valid_time(
            context,
            course,
            time_system,
            assigned_times,
            assigned_rooms,
        )

        if time_quantum is None:
            time_quantum = random.choice(time_system.available_quanta)

        # Find room and instructor
        room_id = _find_suitable_room(context, course, time_quantum, assigned_rooms)
        instructor_id = _select_qualified_instructor(
            context, course, time_quantum, assigned_times
        )

        # Create gene
        gene = SessionGene(
            course_id=course_id,
            group_ids=course.group_ids,
            time_quantum=time_quantum,
            duration_quanta=course.duration_quanta,
            room_id=room_id,
            instructor_id=instructor_id,
        )

        individual.append(gene)

        # Update assignments
        for q in range(time_quantum, time_quantum + course.duration_quanta):
            for group_id in course.group_ids:
                assigned_times[group_id].add(q)
            assigned_times[instructor_id].add(q)
            assigned_rooms[room_id].add(q)

    return individual


# ============================================================================
# EARLIEST DEADLINE FIRST (Prioritize by scheduling urgency)
# ============================================================================


@construction_heuristic(
    name="earliest_deadline_first",
    description="Schedule courses with more sessions per week first (higher frequency = higher priority)",
    priority=3,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=False,
)
def earliest_deadline_first(context: SchedulingContext) -> List[SessionGene]:
    """
    Build schedule prioritizing courses with higher session frequency.

    Courses with more sessions per week are scheduled first because
    they have more scheduling constraints (need to fit more sessions
    in the weekly timetable).

    Priority order:
    1. Courses with most sessions per week
    2. Lab courses (need specific room types)
    3. Courses with part-time instructors (limited availability)

    Algorithm:
    1. Calculate urgency score for each course
    2. Sort courses by urgency (descending)
    3. For each course (urgent to less urgent):
        - Schedule all sessions for this course
        - Space sessions appropriately in the week

    Args:
        context: Scheduling context with entities and available times

    Returns:
        List[SessionGene] representing a complete schedule
    """
    time_system = QuantumTimeSystem(context.config)
    individual = []

    # Calculate urgency scores
    course_urgency = _calculate_urgency_scores(context)

    # Sort by urgency (descending)
    sorted_courses = sorted(course_urgency.items(), key=lambda x: x[1], reverse=True)

    # Track assignments
    assigned_times = defaultdict(set)
    assigned_rooms = defaultdict(set)

    for course_id, _urgency in sorted_courses:
        course = context.courses[course_id]

        # Schedule all sessions for this course
        for session_idx in range(course.sessions_per_week):
            # Find valid time slot
            time_quantum = _find_earliest_valid_time(
                context,
                course,
                time_system,
                assigned_times,
                assigned_rooms,
            )

            if time_quantum is None:
                time_quantum = random.choice(time_system.available_quanta)

            # Find room and instructor
            room_id = _find_suitable_room(context, course, time_quantum, assigned_rooms)
            instructor_id = _select_qualified_instructor(
                context, course, time_quantum, assigned_times
            )

            # Create gene
            gene = SessionGene(
                course_id=course_id,
                group_ids=course.group_ids,
                time_quantum=time_quantum,
                duration_quanta=course.duration_quanta,
                room_id=room_id,
                instructor_id=instructor_id,
            )

            individual.append(gene)

            # Update assignments
            for q in range(time_quantum, time_quantum + course.duration_quanta):
                for group_id in course.group_ids:
                    assigned_times[group_id].add(q)
                assigned_times[instructor_id].add(q)
                assigned_rooms[room_id].add(q)

    return individual


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _calculate_conflict_degrees(context: SchedulingContext) -> Dict[str, int]:
    """Calculate conflict degree for each course (for largest degree first)."""
    degrees = {}

    for course_id, course in context.courses.items():
        degree = 0

        # Count instructor conflicts
        for other_id, other_course in context.courses.items():
            if other_id == course_id:
                continue

            # Check for shared qualified instructors
            shared_instructors = set(course.qualified_instructor_ids) & set(
                other_course.qualified_instructor_ids
            )
            if shared_instructors:
                degree += len(shared_instructors)

            # Check for shared groups
            shared_groups = set(course.group_ids) & set(other_course.group_ids)
            if shared_groups:
                degree += len(shared_groups) * 2  # Group conflicts more critical

        # Lab courses have higher degree (fewer room options)
        if course.is_lab:
            degree += 5

        degrees[course_id] = degree

    return degrees


def _calculate_urgency_scores(context: SchedulingContext) -> Dict[str, float]:
    """Calculate urgency score for each course (for earliest deadline first)."""
    urgency = {}

    for course_id, course in context.courses.items():
        score = 0.0

        # More sessions = higher urgency
        score += course.sessions_per_week * 10

        # Lab courses higher urgency (limited room options)
        if course.is_lab:
            score += 5

        # Courses with part-time instructors have higher urgency
        for instructor_id in course.qualified_instructor_ids:
            instructor = context.instructors.get(instructor_id)
            if instructor and instructor.employment_type == "part_time":
                score += 3

        urgency[course_id] = score

    return urgency


def _count_valid_time_slots(
    context: SchedulingContext,
    course,
    time_system: QuantumTimeSystem,
    assigned_times: Dict,
    assigned_rooms: Dict,
) -> int:
    """Count number of valid time slots for a course session."""
    valid_count = 0

    for time_quantum in time_system.available_quanta:
        # Check if slot is long enough
        if time_quantum + course.duration_quanta > max(time_system.available_quanta):
            continue

        # Check group conflicts
        time_range = range(time_quantum, time_quantum + course.duration_quanta)
        has_conflict = False

        for group_id in course.group_ids:
            if any(q in assigned_times[group_id] for q in time_range):
                has_conflict = True
                break

        if not has_conflict:
            valid_count += 1

    return valid_count


def _find_earliest_valid_time(
    context: SchedulingContext,
    course,
    time_system: QuantumTimeSystem,
    assigned_times: Dict,
    assigned_rooms: Dict,
) -> int:
    """Find earliest valid time slot for a course session."""
    for time_quantum in time_system.available_quanta:
        # Check if slot is long enough
        if time_quantum + course.duration_quanta > max(time_system.available_quanta):
            continue

        # Check conflicts
        time_range = range(time_quantum, time_quantum + course.duration_quanta)
        has_conflict = False

        # Check group conflicts
        for group_id in course.group_ids:
            if any(q in assigned_times[group_id] for q in time_range):
                has_conflict = True
                break

        if not has_conflict:
            return time_quantum

    return None


def _find_suitable_room(
    context: SchedulingContext,
    course,
    time_quantum: int,
    assigned_rooms: Dict,
) -> str:
    """Find suitable room for course session."""
    time_range = range(time_quantum, time_quantum + course.duration_quanta)

    # Filter rooms by type and capacity
    suitable_rooms = [
        room_id
        for room_id, room in context.rooms.items()
        if room.room_type == course.required_room_type
        and room.capacity >= course.expected_students
    ]

    # Find available room
    for room_id in suitable_rooms:
        if not any(q in assigned_rooms[room_id] for q in time_range):
            return room_id

    # Fallback: return first suitable room (will conflict, but repair will fix)
    return suitable_rooms[0] if suitable_rooms else list(context.rooms.keys())[0]


def _select_qualified_instructor(
    context: SchedulingContext,
    course,
    time_quantum: int,
    assigned_times: Dict,
) -> str:
    """Select qualified instructor for course session."""
    time_range = range(time_quantum, time_quantum + course.duration_quanta)

    # Find available qualified instructors
    available_instructors = []

    for instructor_id in course.qualified_instructor_ids:
        instructor = context.instructors.get(instructor_id)
        if not instructor:
            continue

        # Check availability
        is_available = all(instructor.is_available(time_quantum) for q in time_range)

        # Check time conflicts
        has_conflict = any(q in assigned_times[instructor_id] for q in time_range)

        if is_available and not has_conflict:
            available_instructors.append(instructor_id)

    if available_instructors:
        return random.choice(available_instructors)

    # Fallback: return first qualified instructor
    return (
        course.qualified_instructor_ids[0]
        if course.qualified_instructor_ids
        else list(context.instructors.keys())[0]
    )
