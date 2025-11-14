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
from src.config import get_config
from src.constraints.registry import soft_constraint

# Global QuantumTimeSystem instance (initialized once)
_QTS = QuantumTimeSystem()


@soft_constraint(
    name="student_schedule_compactness",
    description="Minimizes gaps in student schedules",
    default_weight=1.5,
    needs_courses=False,
)
def student_schedule_compactness(sessions: List[CourseSession]) -> int:
    """
    Encourages compact student schedules by minimizing idle time gaps.

    Penalizes gaps between the first and last session of each group on each day.
    Does NOT penalize gaps during midday break time (allows proper lunch breaks).

    Args:
        sessions: List of course sessions to evaluate.

    Returns:
        Total penalty points for schedule gaps (excluding break time gaps).
    """
    cfg = get_config()
    gap_penalty = (
        cfg.soft_constraints.student_schedule_compactness.gap_penalty_per_quantum or 1
    )
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
                        penalty += gap_penalty

    return penalty


# 2. Instructor Compactness
@soft_constraint(
    name="instructor_schedule_compactness",
    description="Minimizes gaps in instructor schedules",
    default_weight=1.0,
    needs_courses=False,
)
def instructor_schedule_compactness(sessions: List[CourseSession]) -> int:
    """
    Encourages compact instructor schedules by minimizing idle time gaps.

    Penalizes gaps between the first and last session of each instructor on each day.
    Does NOT penalize gaps during midday break time (allows proper lunch breaks).

    Args:
        sessions: List of course sessions to evaluate.

    Returns:
        Total penalty points for schedule gaps (excluding break time gaps).
    """
    cfg = get_config()
    gap_penalty = (
        cfg.soft_constraints.instructor_schedule_compactness.gap_penalty_per_quantum
        or 1
    )
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
                        penalty += gap_penalty

    return penalty


# 3. Group Midday Break Violation
@soft_constraint(
    name="student_lunch_break",
    description="Encourages students to have midday break time",
    default_weight=1.2,
    needs_courses=False,
)
def student_lunch_break(sessions: List[CourseSession]) -> int:
    """
    Encourages students to have free time during the midday break period.

    Penalizes groups scheduled during lunch hours, with penalty based on
    distance from the break window if no break is available.

    Args:
        sessions: List of CourseSession objects.

    Returns:
        Total lunch break violation penalty across all groups and days.
    """
    cfg = get_config()
    distance_penalty = (
        cfg.soft_constraints.student_lunch_break.distance_penalty_per_quantum or 1
    )
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
            penalty += nearest_dist * distance_penalty

    return penalty


@soft_constraint(
    name="session_continuity",
    description="Encourages sessions to be in appropriate continuous blocks",
    default_weight=2.0,
    needs_courses=False,
)
def session_continuity(sessions: List[CourseSession]) -> int:
    """
    Encourages sessions to be scheduled in continuous, appropriately-sized blocks.

    Theory courses:
    - Preferred block sizes: 2-3 consecutive quanta
    - Penalizes oversized blocks (>3) and isolated single slots (except first one)

    Practical courses:
    - Must be in a single continuous block (no fragmentation)
    - Heavy penalty for splitting practical sessions

    Example (Theory - 6 quanta):
    - [3,3] → 0 penalty (ideal)
    - [2,2,2] → 0 penalty (acceptable)
    - [1,2,3] → 0 penalty (first isolated slot excused)
    - [1,1,4] → 3 penalty (second isolated slot + oversized block)

    Example (Practical - 3 quanta):
    - [3] → 0 penalty (ideal - single continuous block)
    - [2,1] → 20 penalty (fragmented practical)

    Args:
        sessions: List of course sessions to evaluate.

    Returns:
        Total penalty for non-preferred block configurations.
    """
    cfg = get_config().time

    penalty = 0

    # Group sessions by (course_id, course_type, day) to find blocks
    course_day_quanta = defaultdict(lambda: defaultdict(list))
    course_type_map = {}  # Track course types

    for session in sessions:
        # Use course_id + course_type as unique identifier
        course_key = (session.course_id, session.course_type)
        course_type_map[course_key] = session.course_type

        for q in session.session_quanta:
            day, within_day = quantum_to_day_and_within_day(q, _QTS)
            course_day_quanta[course_key][day].append(within_day)

    # Analyze block sizes for each course on each day
    for course_key, course_days in course_day_quanta.items():
        course_type = course_type_map[course_key]

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

            # Apply penalties based on course type
            if course_type.lower() == "practical":
                # Practical courses: must be in a single block
                if len(blocks) > 1:
                    # Heavy penalty for fragmentation
                    penalty += cfg.practical_fragmentation_penalty * (len(blocks) - 1)
            else:
                # Theory courses: apply refined penalty logic
                isolated_count = 0

                for block_size in blocks:
                    if block_size == 1:
                        # Isolated single quantum
                        isolated_count += 1
                        if isolated_count > cfg.theory_max_excused_isolated:
                            # Excused slots exceeded, penalize subsequent ones
                            penalty += cfg.theory_isolated_penalty
                    elif block_size > cfg.preferred_block_size_max:
                        # Oversized block - penalty per quantum beyond max
                        excess = block_size - cfg.preferred_block_size_max
                        penalty += excess * cfg.theory_oversized_penalty_per_quantum
                    # Block sizes within preferred range have no penalty

    return penalty
