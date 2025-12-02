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

import random
from collections import defaultdict

from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.entities.course import Course
from src.entities.instructor import Instructor
from src.entities.room import Room
from src.ga.sessiongene import SessionGene
from src.heuristics.registry import construction_heuristic

type AssignedTimes = dict[str, set[int]]
type AssignedRooms = dict[str, set[int]]

# ================
# LARGEST DEGREE FIRST (Schedule most conflicting courses first)
# ================


@construction_heuristic(
    name="largest_degree_first",
    description="Schedule courses with most conflicts/constraints first (graph coloring heuristic)",
    priority=1,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=False,
)
def largest_degree_first(context: SchedulingContext) -> list[SessionGene]:
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
    # Import subsession breaker and course-group pair generator
    from src.ga.course_group_pairs import generate_course_group_pairs
    from src.ga.group_hierarchy import analyze_group_hierarchy
    from src.ga.population import get_subsession_durations

    time_system = QuantumTimeSystem()
    individual = []

    # Generate course-group pairs (ensures consistent structure with smart population)
    hierarchy = analyze_group_hierarchy(context.groups)
    pair_tuples = generate_course_group_pairs(
        context.courses, context.groups, hierarchy, silent=True
    )

    # Calculate conflict degrees for course-group pairs
    pair_degrees = _calculate_pair_conflict_degrees(pair_tuples, context)

    # Sort pairs by degree (descending - most conflicts first)
    sorted_pairs = sorted(pair_degrees.items(), key=lambda x: x[1], reverse=True)

    # Track assignments for conflict checking
    assigned_times: dict[str, set[int]] = defaultdict(set)  # {entity_id: {time_quanta}}
    assigned_rooms: dict[str, set[int]] = defaultdict(set)  # {room_id: {time_quanta}}

    for (course_key, group_ids, _session_type, _num_quanta), _degree in sorted_pairs:
        course = context.courses.get(course_key)
        if not course:
            continue

        # Unpack tuple - course_key is (course_code, course_type)
        course_code: str
        course_type: str
        if isinstance(course_key, tuple) and len(course_key) == 2:
            course_code, course_type = course_key
        else:
            # Fallback for legacy string keys
            course_code = str(course_key)
            course_type = "theory"

        # Break into subsessions using canonical logic
        subsession_durations = get_subsession_durations(
            course.quanta_per_week, course.course_type
        )

        # Build one gene per subsession
        for _subsession_idx, subsession_duration in enumerate(subsession_durations):
            # Find valid time slot for THIS subsession
            time_quantum = _find_earliest_valid_time(
                context,
                course,
                time_system,
                assigned_times,
                assigned_rooms,
                required_duration=subsession_duration,
            )

            if time_quantum is None:
                # No valid time found - assign random (will be repaired later)
                all_quanta = list(time_system.get_all_operating_quanta())
                # Ensure we can fit the subsession
                valid_starts = [
                    q
                    for q in all_quanta
                    if q + subsession_duration <= time_system.total_quanta
                ]
                time_quantum = (
                    random.choice(valid_starts)
                    if valid_starts
                    else (all_quanta[0] if all_quanta else 0)
                )

            # Find suitable room
            room_id = _find_suitable_room(context, course, time_quantum, assigned_rooms)

            # Select qualified instructor
            instructor_id = _select_qualified_instructor(
                context, course, time_quantum, assigned_times
            )

            # Create gene with subsession duration and correct group_ids from pair
            gene = SessionGene(
                course_id=course_code,
                course_type=course_type,
                group_ids=group_ids,  # Use group_ids from pair, not course.enrolled_group_ids
                room_id=room_id,
                instructor_id=instructor_id,
                start_quanta=time_quantum,
                num_quanta=subsession_duration,  # Use subsession duration!
            )

            individual.append(gene)

            # Update assignments for THIS subsession
            for q in range(time_quantum, time_quantum + subsession_duration):
                for group_id in group_ids:  # Use group_ids from pair
                    assigned_times[group_id].add(q)
                assigned_times[instructor_id].add(q)
                assigned_rooms[room_id].add(q)

    return individual


# ================
# MOST CONSTRAINED FIRST (Schedule sessions with fewest options first)
# ================


@construction_heuristic(
    name="most_constrained_first",
    description="Schedule sessions with fewest valid time slots first (minimum remaining values)",
    priority=2,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=False,
)
def most_constrained_first(context: SchedulingContext) -> list[SessionGene]:
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
    # Import utilities for course-group pairs
    from src.ga.course_group_pairs import generate_course_group_pairs
    from src.ga.group_hierarchy import analyze_group_hierarchy
    from src.ga.population import get_subsession_durations

    time_system = QuantumTimeSystem()
    individual = []

    # Track assignments
    assigned_times: dict[str, set[int]] = defaultdict(set)
    assigned_rooms: dict[str, set[int]] = defaultdict(set)

    # Generate course-group pairs (ensures consistent structure)
    hierarchy = analyze_group_hierarchy(context.groups)
    pair_tuples = generate_course_group_pairs(
        context.courses, context.groups, hierarchy, silent=True
    )

    # Build list of all sessions to schedule (with subsessions)
    sessions_to_schedule = []
    for course_key, group_ids, _session_type, _num_quanta in pair_tuples:
        course = context.courses.get(course_key)
        if not course:
            continue
        subsession_durations = get_subsession_durations(
            course.quanta_per_week, course.course_type
        )
        for _subsession_idx, subsession_duration in enumerate(subsession_durations):
            sessions_to_schedule.append(
                (course_key, group_ids, course, subsession_duration)
            )

    # Schedule sessions in constraint order
    while sessions_to_schedule:
        # Find most constrained session
        most_constrained = None
        min_options = float("inf")

        for course_key, group_ids, course, subsession_duration in sessions_to_schedule:
            # Count valid time slots for THIS subsession duration
            valid_slots = _count_valid_time_slots(
                context,
                course,
                time_system,
                assigned_times,
                assigned_rooms,
                required_duration=subsession_duration,
            )

            if valid_slots < min_options:
                min_options = valid_slots
                most_constrained = (course_key, group_ids, course, subsession_duration)

        # Remove from pending list
        if most_constrained is None:
            break  # No valid assignments possible
        sessions_to_schedule.remove(most_constrained)
        course_key, group_ids, course, subsession_duration = most_constrained

        # Find best time slot for THIS subsession
        time_quantum = _find_earliest_valid_time(
            context,
            course,
            time_system,
            assigned_times,
            assigned_rooms,
            required_duration=subsession_duration,
        )

        if time_quantum is None:
            valid_starts = [
                q
                for q in context.available_quanta
                if q + subsession_duration <= time_system.total_quanta
            ]
            time_quantum = (
                random.choice(valid_starts)
                if valid_starts
                else context.available_quanta[0]
            )

        # Find room and instructor
        room_id = _find_suitable_room(context, course, time_quantum, assigned_rooms)
        instructor_id = _select_qualified_instructor(
            context, course, time_quantum, assigned_times
        )

        # Create gene with subsession duration and correct group_ids from pair
        course_code, course_type = course_key  # Unpack tuple
        gene = SessionGene(
            course_id=course_code,
            course_type=course_type,
            group_ids=group_ids,  # Use group_ids from pair, not course.enrolled_group_ids
            room_id=room_id,
            instructor_id=instructor_id,
            start_quanta=time_quantum,
            num_quanta=subsession_duration,  # Use subsession duration!
        )

        individual.append(gene)

        # Update assignments for THIS subsession
        for q in range(time_quantum, time_quantum + subsession_duration):
            for group_id in group_ids:  # Use group_ids from pair
                assigned_times[group_id].add(q)
            assigned_times[instructor_id].add(q)
            assigned_rooms[room_id].add(q)

    return individual


# ================
# EARLIEST DEADLINE FIRST (Prioritize by scheduling urgency)
# ================


@construction_heuristic(
    name="earliest_deadline_first",
    description="Schedule courses with more sessions per week first (higher frequency = higher priority)",
    priority=3,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=False,
)
def earliest_deadline_first(context: SchedulingContext) -> list[SessionGene]:
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
    # Import utilities for course-group pairs
    from src.ga.course_group_pairs import generate_course_group_pairs
    from src.ga.group_hierarchy import analyze_group_hierarchy
    from src.ga.population import get_subsession_durations

    time_system = QuantumTimeSystem()
    individual = []

    # Generate course-group pairs (ensures consistent structure)
    hierarchy = analyze_group_hierarchy(context.groups)
    pair_tuples = generate_course_group_pairs(
        context.courses, context.groups, hierarchy, silent=True
    )

    # Calculate urgency scores for pairs
    pair_urgency = _calculate_pair_urgency_scores(pair_tuples, context)

    # Sort by urgency (descending)
    sorted_pairs = sorted(pair_urgency.items(), key=lambda x: x[1], reverse=True)

    # Track assignments
    assigned_times: dict[str, set[int]] = defaultdict(set)
    assigned_rooms: dict[str, set[int]] = defaultdict(set)

    for (course_key, group_ids, _session_type, _num_quanta), _urgency in sorted_pairs:
        course = context.courses.get(course_key)
        if not course:
            continue

        # Unpack tuple - course_key is (course_code, course_type)
        course_code: str
        course_type: str
        if isinstance(course_key, tuple) and len(course_key) == 2:
            course_code, course_type = course_key
        else:
            # Fallback for legacy string keys
            course_code = str(course_key)
            course_type = "theory"

        # Break into subsessions
        subsession_durations = get_subsession_durations(
            course.quanta_per_week, course.course_type
        )

        # Schedule each subsession
        for _subsession_idx, subsession_duration in enumerate(subsession_durations):
            # Find valid time slot for THIS subsession
            time_quantum = _find_earliest_valid_time(
                context,
                course,
                time_system,
                assigned_times,
                assigned_rooms,
                required_duration=subsession_duration,
            )

            if time_quantum is None:
                valid_starts = [
                    q
                    for q in context.available_quanta
                    if q + subsession_duration <= time_system.total_quanta
                ]
                time_quantum = (
                    random.choice(valid_starts)
                    if valid_starts
                    else context.available_quanta[0]
                )

            # Find room and instructor
            room_id = _find_suitable_room(context, course, time_quantum, assigned_rooms)
            instructor_id = _select_qualified_instructor(
                context, course, time_quantum, assigned_times
            )

            # Create gene with subsession duration and correct group_ids from pair
            gene = SessionGene(
                course_id=course_code,
                course_type=course_type,
                group_ids=group_ids,  # Use group_ids from pair, not course.enrolled_group_ids
                room_id=room_id,
                instructor_id=instructor_id,
                start_quanta=time_quantum,
                num_quanta=subsession_duration,  # Use subsession duration!
            )

            individual.append(gene)

            # Update assignments for THIS subsession
            for q in range(time_quantum, time_quantum + subsession_duration):
                for group_id in group_ids:  # Use group_ids from pair
                    assigned_times[group_id].add(q)
                assigned_times[instructor_id].add(q)
                assigned_rooms[room_id].add(q)

    return individual


# ================
# HELPER FUNCTIONS
# ================


def _calculate_conflict_degrees(context: SchedulingContext) -> dict[tuple, int]:
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
            shared_groups = set(course.enrolled_group_ids) & set(
                other_course.enrolled_group_ids
            )
            if shared_groups:
                degree += len(shared_groups) * 2  # Group conflicts more critical

        # Lab courses have higher degree (fewer room options)
        if course.course_type == "practical":
            degree += 5

        degrees[course_id] = degree

    return degrees


def _calculate_urgency_scores(context: SchedulingContext) -> dict[tuple, float]:
    """Calculate urgency score for each course (for earliest deadline first)."""
    urgency = {}

    for course_id, course in context.courses.items():
        score = 0.0

        # More quanta = higher urgency
        score += course.quanta_per_week * 10

        # Lab courses higher urgency (limited room options)
        if course.course_type == "practical":
            score += 5

        # Courses with part-time instructors have higher urgency
        for instructor_id in course.qualified_instructor_ids:
            instructor = context.instructors.get(instructor_id)
            if instructor and not instructor.is_full_time:
                score += 3

        urgency[course_id] = score

    return urgency


def _count_valid_time_slots(
    context: SchedulingContext,
    course: Course,
    time_system: QuantumTimeSystem,
    assigned_times: AssignedTimes,
    assigned_rooms: AssignedRooms,
    required_duration: int | None = None,  # NEW: subsession duration
) -> int:
    """
    Count number of valid time slots for a course session.

    Args:
        required_duration: Duration in quanta for THIS subsession.
                          If None, uses course.quanta_per_week.
    """
    # Use subsession duration if provided, otherwise full course duration
    duration = (
        required_duration if required_duration is not None else course.quanta_per_week
    )

    valid_count = 0

    for time_quantum in context.available_quanta:
        # Check if slot is long enough for the subsession
        if time_quantum + duration > max(context.available_quanta):
            continue

        # Check group conflicts
        time_range = range(time_quantum, time_quantum + duration)
        has_conflict = False

        for group_id in course.enrolled_group_ids:
            if any(q in assigned_times[group_id] for q in time_range):
                has_conflict = True
                break

        if not has_conflict:
            valid_count += 1

    return valid_count


def _find_earliest_valid_time(
    context: SchedulingContext,
    course: Course,
    time_system: QuantumTimeSystem,
    assigned_times: AssignedTimes,
    assigned_rooms: AssignedRooms,
    required_duration: int | None = None,  # NEW: subsession duration
) -> int | None:
    """
    Find earliest valid time slot for a course session.

    Args:
        required_duration: Duration in quanta for THIS subsession (not full course).
                          If None, uses course.quanta_per_week.
    """
    # Use subsession duration if provided, otherwise full course duration
    duration = (
        required_duration if required_duration is not None else course.quanta_per_week
    )

    for time_quantum in context.available_quanta:
        # Check if slot is long enough for the subsession
        if time_quantum + duration > max(context.available_quanta):
            continue

        # Check conflicts
        time_range = range(time_quantum, time_quantum + duration)
        has_conflict = False

        # Check group conflicts
        for group_id in course.enrolled_group_ids:
            if any(q in assigned_times[group_id] for q in time_range):
                has_conflict = True
                break

        if not has_conflict:
            return time_quantum

    return None


def _find_suitable_room(
    context: SchedulingContext,
    course: Course,
    time_quantum: int,
    assigned_rooms: AssignedRooms,
) -> str:
    """Find suitable room for course session."""
    time_range = range(time_quantum, time_quantum + course.quanta_per_week)

    # Filter rooms by features (type matching)
    suitable_rooms = [
        room_id
        for room_id, room in context.rooms.items()
        if isinstance(room, Room)
        and room.is_suitable_for_course_type(course.required_room_features)
    ]

    # Find available room
    for room_id in suitable_rooms:
        if not any(q in assigned_rooms[room_id] for q in time_range):
            return room_id

    # Fallback: return first suitable room (will conflict, but repair will fix)
    return suitable_rooms[0] if suitable_rooms else list(context.rooms.keys())[0]


def _select_qualified_instructor(
    context: SchedulingContext,
    course: Course,
    time_quantum: int,
    assigned_times: AssignedTimes,
) -> str:
    """Select qualified instructor for course session."""
    time_range = range(time_quantum, time_quantum + course.quanta_per_week)

    # Find available qualified instructors
    available_instructors = []

    from src.encoder.quantum_time_system import QuantumTimeSystem

    time_system = QuantumTimeSystem()

    for instructor_id in course.qualified_instructor_ids:
        instructor: Instructor | None = context.instructors.get(instructor_id)
        if not instructor:
            continue

        # Check availability for all quanta in the range
        is_available = all(
            instructor.is_available_at_quanta(q, time_system) for q in time_range
        )

        # Check time conflicts
        has_conflict = any(q in assigned_times[instructor_id] for q in time_range)

        if is_available and not has_conflict:
            available_instructors.append(instructor_id)

    if available_instructors:
        return str(random.choice(available_instructors))

    # Fallback: return first qualified instructor
    return str(
        course.qualified_instructor_ids[0]
        if course.qualified_instructor_ids
        else list(context.instructors.keys())[0]
    )


# ================
# PAIR-BASED HELPER FUNCTIONS (Use course-group pairs)
# ================


def _calculate_pair_conflict_degrees(
    pair_tuples: list[tuple], context: SchedulingContext
) -> dict[tuple, int]:
    """
    Calculate conflict degree for each course-group pair.

    Args:
        pair_tuples: List of (course_key, group_ids, session_type, num_quanta)
        context: SchedulingContext

    Returns:
        Dictionary mapping pair tuple to conflict degree
    """
    degrees = {}

    for pair in pair_tuples:
        course_key, group_ids, _session_type, _num_quanta = pair
        course = context.courses.get(course_key)
        if not course:
            degrees[pair] = 0
            continue

        degree = 0

        # Count instructor conflicts with other pairs
        for other_pair in pair_tuples:
            if other_pair == pair:
                continue

            other_key, other_groups, _, _ = other_pair
            other_course = context.courses.get(other_key)
            if not other_course:
                continue

            # Check for shared qualified instructors
            shared_instructors = set(course.qualified_instructor_ids) & set(
                other_course.qualified_instructor_ids
            )
            if shared_instructors:
                degree += len(shared_instructors)

            # Check for shared groups
            shared_groups = set(group_ids) & set(other_groups)
            if shared_groups:
                degree += len(shared_groups) * 2  # Group conflicts more critical

        # Lab courses have higher degree (fewer room options)
        if course.course_type == "practical":
            degree += 5

        degrees[pair] = degree

    return degrees


def _calculate_pair_urgency_scores(
    pair_tuples: list[tuple], context: SchedulingContext
) -> dict[tuple, float]:
    """
    Calculate urgency score for each course-group pair.

    Args:
        pair_tuples: List of (course_key, group_ids, session_type, num_quanta)
        context: SchedulingContext

    Returns:
        Dictionary mapping pair tuple to urgency score
    """
    urgency = {}

    for pair in pair_tuples:
        course_key, _group_ids, _session_type, num_quanta = pair
        course = context.courses.get(course_key)
        if not course:
            urgency[pair] = 0.0
            continue

        score = 0.0

        # More quanta = higher urgency
        score += num_quanta * 10

        # Lab courses higher urgency (limited room options)
        if course.course_type == "practical":
            score += 5

        # Courses with part-time instructors have higher urgency
        for instructor_id in course.qualified_instructor_ids:
            instructor = context.instructors.get(instructor_id)
            if instructor and not instructor.is_full_time:
                score += 3

        urgency[pair] = score

    return urgency
