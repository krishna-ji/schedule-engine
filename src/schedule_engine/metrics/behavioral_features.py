"""
Behavioral feature extraction for schedule individuals.

ENHANCEMENT #6: Extract phenotypic features that characterize solution behavior.

Behavioral characterization captures HOW a solution achieves its fitness,
not just WHAT fitness it achieves. This enables novelty search and diversity
preservation beyond fitness-based selection.

Features extracted:
1. Time distribution (sessions per day)
2. Room utilization (sessions per room type)
3. Instructor workload (sessions per instructor)
4. Course clustering (sessions per course)
5. Constraint profile (violation pattern)

Mathematical Definition:
    φ: S → R^d
    Where φ is the behavioral characterization function mapping
    solutions S to d-dimensional behavior space.
"""

import numpy as np
from numpy.typing import NDArray

from schedule_engine.domain.types import Individual, SchedulingContext
from schedule_engine.io.decoder import decode_individual


def extract_behavioral_features(
    individual: Individual,
    context: SchedulingContext,
) -> NDArray[np.float64]:
    """
    Extract behavioral features from a schedule individual.

    These features characterize HOW the solution is structured,
    enabling diversity beyond fitness-only metrics.

    Args:
        individual: Schedule individual (list of SessionGenes)
        context: Scheduling context with courses, rooms, instructors, groups

    Returns:
        Feature vector of shape (d,) where d ≈ 20-30 dimensions

    Feature Categories:
    1. Time distribution (7 features):
       - Sessions per day (Sun-Sat)

    2. Room utilization (6 features):
       - Mean/std room utilization
       - Room type distribution (lecture halls, labs, seminar rooms)
       - Room capacity utilization ratio

    3. Instructor workload (4 features):
       - Mean/std sessions per instructor
       - Max instructor load
       - Instructor idle time

    4. Course distribution (4 features):
       - Mean/std sessions per course
       - Course clustering (temporal proximity)
       - Course spread (temporal diversity)

    5. Constraint profile (5 features):
       - Hard constraint violation pattern
       - Soft constraint violation pattern
       - Constraint diversity

    Example:
        >>> features = extract_behavioral_features(individual, context)
        >>> print(f"Behavior vector: {features.shape}")
        >>> print(f"Time distribution: {features[:7]}")
    """
    # Decode individual to sessions
    sessions = decode_individual(
        individual,
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    features = []

    # 1. Time distribution (7 features: sessions per day)
    time_dist = _compute_sessions_per_day(sessions)
    features.extend(time_dist)

    # 2. Room utilization (6 features)
    room_features = _compute_room_utilization(sessions, context)
    features.extend(room_features)

    # 3. Instructor workload (4 features)
    instructor_features = _compute_instructor_workload(sessions, context)
    features.extend(instructor_features)

    # 4. Course distribution (4 features)
    course_features = _compute_course_distribution(sessions, context)
    features.extend(course_features)

    # 5. Constraint profile (5 features)
    constraint_features = _compute_constraint_profile(individual)
    features.extend(constraint_features)

    return np.array(features, dtype=np.float64)


def _compute_sessions_per_day(sessions: list) -> list[float]:
    """
    Compute number of sessions per day of week.

    Returns 7 values (Sun-Sat) normalized by total sessions.
    """
    from schedule_engine.io.time_system import QuantumTimeSystem

    qts = QuantumTimeSystem()
    sessions_per_day = [0.0] * 7  # Sun-Sat
    day_to_index = {
        "sunday": 0,
        "monday": 1,
        "tuesday": 2,
        "wednesday": 3,
        "thursday": 4,
        "friday": 5,
        "saturday": 6,
    }

    for session in sessions:
        for quantum in session.session_quanta:
            day_name, _ = qts.quanta_to_time(quantum)
            day_index = day_to_index.get(day_name.lower(), 0)
            sessions_per_day[day_index] += 1.0

    # Normalize by total sessions
    total = sum(sessions_per_day)
    if total > 0:
        sessions_per_day = [count / total for count in sessions_per_day]

    return sessions_per_day


def _compute_room_utilization(
    sessions: list, context: SchedulingContext
) -> list[float]:
    """
    Compute room utilization metrics.

    Returns 6 features:
    - Mean sessions per room
    - Std sessions per room
    - Room capacity utilization (mean)
    - Room capacity utilization (std)
    - Lecture hall usage ratio
    - Lab usage ratio
    """
    room_usage = dict.fromkeys(context.rooms.keys(), 0)
    room_capacities = []
    group_sizes = []

    for session in sessions:
        room_usage[session.room_id] += 1

        # Calculate capacity utilization
        room = context.rooms.get(session.room_id)
        if room:
            room_capacity = room.capacity
            group_size = sum(
                [
                    context.groups[gid].student_count
                    for gid in session.group_ids
                    if gid in context.groups
                ]
            )
            if room_capacity > 0:
                utilization = group_size / room_capacity
                room_capacities.append(utilization)
                group_sizes.append(group_size)

    usage_values = list(room_usage.values())
    mean_usage = float(np.mean(usage_values)) if usage_values else 0.0
    std_usage = float(np.std(usage_values)) if usage_values else 0.0

    mean_capacity = float(np.mean(room_capacities)) if room_capacities else 0.0
    std_capacity = float(np.std(room_capacities)) if room_capacities else 0.0

    # Room type ratios (simplified: assume room names indicate type)
    lecture_halls = sum(
        1 for rid, room in context.rooms.items() if "lecture" in room.name.lower()
    )
    labs = sum(1 for rid, room in context.rooms.items() if "lab" in room.name.lower())
    total_rooms = len(context.rooms)

    lecture_ratio = lecture_halls / total_rooms if total_rooms > 0 else 0.0
    lab_ratio = labs / total_rooms if total_rooms > 0 else 0.0

    return [
        mean_usage,
        std_usage,
        mean_capacity,
        std_capacity,
        lecture_ratio,
        lab_ratio,
    ]


def _compute_instructor_workload(
    sessions: list, context: SchedulingContext
) -> list[float]:
    """
    Compute instructor workload metrics.

    Returns 4 features:
    - Mean sessions per instructor
    - Std sessions per instructor
    - Max instructor load
    - Instructor idle time (proportion of instructors not assigned)
    """
    instructor_loads = dict.fromkeys(context.instructors.keys(), 0)

    for session in sessions:
        if session.instructor_id in instructor_loads:
            instructor_loads[session.instructor_id] += 1

    loads = list(instructor_loads.values())

    mean_load = float(np.mean(loads)) if loads else 0.0
    std_load = float(np.std(loads)) if loads else 0.0
    max_load = float(np.max(loads)) if loads else 0.0

    # Idle instructors (zero load)
    idle_instructors = sum(1 for load in loads if load == 0)
    idle_ratio = idle_instructors / len(loads) if loads else 0.0

    return [mean_load, std_load, max_load, idle_ratio]


def _compute_course_distribution(
    sessions: list, context: SchedulingContext
) -> list[float]:
    """
    Compute course distribution metrics.

    Returns 4 features:
    - Mean sessions per course
    - Std sessions per course
    - Course temporal clustering (mean gap between sessions of same course)
    - Course temporal spread (std of session times for same course)
    """
    course_sessions = {}
    course_times: dict[str, list[int]] = {}

    for session in sessions:
        course_id = session.course_id
        if course_id not in course_sessions:
            course_sessions[course_id] = 0
            course_times[course_id] = []

        course_sessions[course_id] += 1
        course_times[course_id].extend(session.session_quanta)

    sessions_per_course = list(course_sessions.values())
    mean_sessions = float(np.mean(sessions_per_course)) if sessions_per_course else 0.0
    std_sessions = float(np.std(sessions_per_course)) if sessions_per_course else 0.0

    # Temporal clustering: mean gap between consecutive sessions
    gaps = []
    for times in course_times.values():
        if len(times) > 1:
            sorted_times = sorted(times)
            course_gaps = [
                sorted_times[i + 1] - sorted_times[i]
                for i in range(len(sorted_times) - 1)
            ]
            gaps.extend(course_gaps)

    mean_gap = float(np.mean(gaps)) if gaps else 0.0

    # Temporal spread: std of session times
    spreads = []
    for times in course_times.values():
        if len(times) > 1:
            spreads.append(float(np.std(times)))

    mean_spread = float(np.mean(spreads)) if spreads else 0.0

    return [mean_sessions, std_sessions, mean_gap, mean_spread]


def _compute_constraint_profile(individual: Individual) -> list[float]:
    """
    Compute constraint violation profile.

    Returns 5 features:
    - Total hard violations
    - Total soft violations
    - Hard/soft ratio
    - Constraint diversity (how many different constraints violated)
    - Violation intensity (average violations per constraint)
    """
    if not hasattr(individual, "fitness") or not individual.fitness.valid:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    hard_violations = individual.fitness.values[0]
    soft_violations = individual.fitness.values[1]

    # Hard/soft ratio
    if soft_violations > 0:
        ratio = hard_violations / soft_violations
    else:
        ratio = hard_violations if hard_violations > 0 else 0.0

    # Constraint diversity (placeholder - would need per-constraint breakdown)
    # For now, use a simple heuristic based on violation magnitude
    constraint_diversity = min(hard_violations + soft_violations, 10.0) / 10.0

    # Violation intensity
    total_violations = hard_violations + soft_violations
    intensity = np.tanh(total_violations / 100.0)  # Normalize to [0, 1]

    return [
        float(hard_violations),
        float(soft_violations),
        float(ratio),
        float(constraint_diversity),
        float(intensity),
    ]


def compute_behavioral_distance(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
    metric: str = "euclidean",
) -> float:
    """
    Compute distance between two behavioral feature vectors.

    Args:
        features1: First feature vector
        features2: Second feature vector
        metric: Distance metric ("euclidean", "manhattan", "cosine")

    Returns:
        Distance value
    """
    if metric == "euclidean":
        return float(np.linalg.norm(features1 - features2))
    elif metric == "manhattan":
        return float(np.sum(np.abs(features1 - features2)))
    elif metric == "cosine":
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        return float(1.0 - (dot_product / (norm1 * norm2)))
    else:
        raise ValueError(f"Unknown metric: {metric}")
