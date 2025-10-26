"""
Soft constraint penalty functions for UCTP.
Each function returns an integer penalty representing violations of a quality rule.
These do not impact feasibility, but aim to improve real-world schedule quality.

IMPORTANT: Uses CONTINUOUS quantum system. All time conversions must go through
QuantumTimeSystem. Never use QUANTA_PER_DAY or day = q // QUANTA_PER_DAY.
"""

from typing import List
from collections import defaultdict
from src.entities.decoded_session import CourseSession
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.utils.time_helpers import (
    get_midday_break_quanta,
    quantum_to_day_and_within_day,
)
from config import get_config

# Global QuantumTimeSystem instance (initialized once)
_QTS = QuantumTimeSystem()


# 1. Group Compactness: penalize gaps in daily group schedule
def group_gaps_penalty(sessions: List[CourseSession]) -> int:
    """Calculate penalty for gaps in daily group schedules.

    Penalizes idle time slots between the first and last session of each group
    on each day to encourage compact schedules.

    IMPORTANT: Does NOT penalize gaps that occur during midday break time.
    This allows students to have proper lunch breaks without penalty.

    Args:
        sessions: List of course sessions to evaluate.

    Returns:
        Total penalty points for group schedule gaps (excluding break time gaps).
    """
    penalty = 0

    # Get midday break quanta for each day
    break_quanta_by_day = get_midday_break_quanta(_QTS)

    group_day_quanta = defaultdict(
        lambda: defaultdict(set)
    )  # group_id -> day_name -> set of within-day quanta

    for session in sessions:
        for group_id in session.group_ids:
            for q in session.session_quanta:
                day, within_day = quantum_to_day_and_within_day(q, _QTS)
                group_day_quanta[group_id][day].add(within_day)

    # Analyze gaps for each group on each day
    for days in group_day_quanta.values():
        for day_name, quanta in days.items():
            if not quanta or len(quanta) < 2:
                continue  # No gaps possible with 0 or 1 session

            sorted_quanta = sorted(quanta)
            min_q, max_q = sorted_quanta[0], sorted_quanta[-1]

            # Get break quanta for this specific day
            break_quanta = break_quanta_by_day.get(day_name, set())

            # Find all gaps (missing quanta between min and max)
            for q in range(min_q, max_q + 1):
                if q not in sorted_quanta:
                    # This is a gap - but check if it's during break time
                    if q in break_quanta:
                        # Gap during break time - NO PENALTY (legitimate lunch break)
                        continue
                    else:
                        # Gap during non-break time - PENALIZE (idle/wasted time)
                        penalty += 1

    return penalty


# 2. Instructor Compactness
def instructor_gaps_penalty(sessions: List[CourseSession]) -> int:
    """Calculate penalty for gaps in daily instructor schedules.

    Penalizes idle time slots between the first and last session of each
    instructor on each day to encourage compact teaching schedules.

    IMPORTANT: Does NOT penalize gaps that occur during midday break time.
    This allows instructors to have proper lunch breaks without penalty.

    Args:
        sessions: List of course sessions to evaluate.

    Returns:
        Total penalty points for instructor schedule gaps (excluding break time gaps).
    """
    penalty = 0

    # Get midday break quanta for each day
    break_quanta_by_day = get_midday_break_quanta(_QTS)

    instructor_day_quanta = defaultdict(lambda: defaultdict(set))

    for session in sessions:
        iid = session.instructor_id
        for q in session.session_quanta:
            day, within_day = quantum_to_day_and_within_day(q, _QTS)
            instructor_day_quanta[iid][day].add(within_day)

    # Analyze gaps for each instructor on each day
    for days in instructor_day_quanta.values():
        for day_name, quanta in days.items():
            if not quanta or len(quanta) < 2:
                continue  # No gaps possible with 0 or 1 session

            sorted_quanta = sorted(quanta)
            min_q, max_q = sorted_quanta[0], sorted_quanta[-1]

            # Get break quanta for this specific day
            break_quanta = break_quanta_by_day.get(day_name, set())

            # Find all gaps (missing quanta between min and max)
            for q in range(min_q, max_q + 1):
                if q not in sorted_quanta:
                    # This is a gap - but check if it's during break time
                    if q in break_quanta:
                        # Gap during break time - NO PENALTY (legitimate lunch break)
                        continue
                    else:
                        # Gap during non-break time - PENALIZE (idle/wasted time)
                        penalty += 1

    return penalty


# 3. Group Midday Break Violation
def group_midday_break_violation(sessions: List[CourseSession]) -> int:
    """
    Penalizes groups that do not have a break during the midday break period.

    For each group per day, if no session falls in the break window,
    penalize by the minimum distance from any scheduled quantum to the break block.

    Args:
        sessions (List[CourseSession]): List of CourseSession objects.

    Returns:
        int: Total break violation penalty across all groups and days.
    """
    penalty = 0
    # Get break quanta for each day (day_name -> set of within-day quanta)
    break_quanta_by_day = get_midday_break_quanta(_QTS)

    group_day_quanta = defaultdict(lambda: defaultdict(set))

    for session in sessions:
        for gid in session.group_ids:
            for q in session.session_quanta:
                day, within_day = quantum_to_day_and_within_day(q, _QTS)
                group_day_quanta[gid][day].add(within_day)

    for days in group_day_quanta.values():
        for day_name, quanta in days.items():
            # Get break quanta for this specific day
            if day_name not in break_quanta_by_day:
                continue  # No break defined for this day

            break_quanta = break_quanta_by_day[day_name]

            if break_quanta & quanta:
                continue  # No penalty if group is free during break
            # Compute min distance to break window
            nearest_dist = min(abs(q - bq) for q in quanta for bq in break_quanta)
            penalty += nearest_dist

    return penalty


# 4. Session Block Size Preference
def session_block_clustering_penalty(sessions: List[CourseSession]) -> int:
    """
    Penalizes course sessions that are fragmented into undesirable block sizes.

    Encourages sessions to be clustered into blocks of 2-3 consecutive quanta
    rather than isolated single quanta or overly large blocks.

    Penalty logic:
    - Block size 1 (isolated): Heavy penalty (configurable)
    - Block size 2-3: No penalty (preferred)
    - Block size 4+: Moderate penalty for each quantum beyond 3

    Example:
    - 6 quanta as [3,3] → 0 penalty (ideal)
    - 6 quanta as [2,2,2] → 0 penalty (acceptable)
    - 6 quanta as [1,2,3] → penalty for the isolated 1
    - 6 quanta as [6] → penalty for oversized block (6-3=3 penalty)

    Args:
        sessions: List of course sessions to evaluate.

    Returns:
        Total penalty for non-preferred block sizes.
    """
    cfg = get_config().time

    penalty = 0

    # Group sessions by (course_id, course_type, day) to find blocks
    course_day_quanta = defaultdict(lambda: defaultdict(list))

    for session in sessions:
        # Use course_id + course_type as unique identifier
        course_key = (session.course_id, session.course_type)

        for q in session.session_quanta:
            day, within_day = quantum_to_day_and_within_day(q, _QTS)
            course_day_quanta[course_key][day].append(within_day)

    # Analyze block sizes for each course on each day
    for course_days in course_day_quanta.values():
        for day_quanta in course_days.values():
            # Sort quanta to identify consecutive blocks
            sorted_quanta = sorted(day_quanta)

            # Find consecutive blocks
            blocks = []
            if sorted_quanta:
                current_block = [sorted_quanta[0]]

                for i in range(1, len(sorted_quanta)):
                    if sorted_quanta[i] == sorted_quanta[i - 1] + 1:
                        # Consecutive - add to current block
                        current_block.append(sorted_quanta[i])
                    else:
                        # Gap - start new block
                        blocks.append(len(current_block))
                        current_block = [sorted_quanta[i]]

                # Don't forget the last block
                blocks.append(len(current_block))

            # Penalize based on block sizes
            for block_size in blocks:
                if block_size == 1:
                    # Isolated single quantum - heavy penalty
                    penalty += cfg.isolated_session_penalty
                elif block_size < cfg.preferred_block_size_min:
                    # Below minimum preferred size (shouldn't happen if min=2 and we penalize 1)
                    penalty += cfg.preferred_block_size_min - block_size
                elif block_size > cfg.preferred_block_size_max:
                    # Oversized block - penalize excess quanta
                    excess = block_size - cfg.preferred_block_size_max
                    penalty += excess * cfg.oversized_block_penalty_per_quantum
                # else: block_size is 2 or 3 → no penalty (preferred)

    return penalty


# ---------------------------
# Soft Constraint Registry
# ---------------------------
def get_all_soft_constraints():
    """
    Returns a dictionary of all available soft constraint functions.

    Soft constraints (4 total):
    1. group_gaps_penalty - Penalize gaps in group schedules
    2. instructor_gaps_penalty - Penalize gaps in instructor schedules
    3. group_midday_break_violation - Penalize sessions during midday break
    4. session_block_clustering_penalty - Encourage 2-3 quantum session blocks

    Returns:
        Dict[str, callable]: Mapping of constraint names to their functions.
    """
    return {
        "group_gaps_penalty": group_gaps_penalty,
        "instructor_gaps_penalty": instructor_gaps_penalty,
        "group_midday_break_violation": group_midday_break_violation,
        "session_block_clustering_penalty": session_block_clustering_penalty,
    }


def get_enabled_soft_constraints():
    """
    Returns only the enabled soft constraints based on config.

    Returns:
        Dict[str, dict]: Mapping of enabled constraint names to their config (function, weight).
    """
    all_constraints = get_all_soft_constraints()
    enabled = {}

    cfg = get_config().soft_constraints
    for name, func in all_constraints.items():
        constraint_cfg = getattr(cfg, name, None)
        if constraint_cfg and constraint_cfg.enabled:
            enabled[name] = {
                "function": func,
                "weight": constraint_cfg.weight,
            }

    return enabled
