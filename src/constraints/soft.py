"""
Soft constraint penalty functions for UCTP.
Each function returns an integer penalty representing violations of a quality rule.
These do not impact feasibility, but aim to improve real-world schedule quality.

IMPORTANT: Uses CONTINUOUS quantum system. All time conversions must go through
QuantumTimeSystem. Never use QUANTA_PER_DAY or day = q // QUANTA_PER_DAY.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from src.config import get_config
from src.constraints.registry import soft_constraint
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.entities.decoded_session import CourseSession
from src.utils.time_helpers import (
    get_midday_break_quanta,
    quantum_to_day_and_within_day,
)

# Global QuantumTimeSystem instance (initialized once)
_QTS = QuantumTimeSystem()


@soft_constraint(
    name="student_schedule_compactness",
    description="Minimizes gaps in student schedules",
    default_weight=1.5,
    needs_courses=False,
)
def student_schedule_compactness(sessions: list[CourseSession]) -> int:
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

    group_day_quanta: dict[str, dict[str, set[int]]] = defaultdict(
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
def instructor_schedule_compactness(sessions: list[CourseSession]) -> int:
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

    instructor_day_quanta: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )

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
def student_lunch_break(sessions: list[CourseSession]) -> int:
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

    group_day_quanta: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )

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
            # Compute min distance to break window (vectorized for 5-20x speedup)
            if quanta and break_quanta:
                quanta_arr = np.array(sorted(quanta))
                break_arr = np.array(sorted(break_quanta))
                # Broadcasting: compute all pairwise differences efficiently
                diffs = np.abs(quanta_arr[:, np.newaxis] - break_arr)
                nearest_dist = np.min(diffs)
                penalty += nearest_dist * distance_penalty

    return penalty


@soft_constraint(
    name="session_continuity",
    description="Encourages sessions to be in appropriate continuous blocks",
    default_weight=1.0,
    needs_courses=False,
)
def session_continuity(sessions: list[CourseSession]) -> int:
    """
    Encourages sessions to be scheduled in continuous, appropriately-sized blocks.

    Theory/Lecture/Tutorial courses (L+T combined):
    - Preferred block sizes: 2-3 consecutive quanta per day
    - Penalizes isolated single slots (without contiguity)
    - First isolated single slot is excused per course
    - Theory and practical are evaluated SEPARATELY (no cross-type coalescence)

    Practical courses:
    - Always contiguous (enforced by SessionGene structure)
    - No penalty needed - fragmentation is structurally impossible
    - Practical is "one-shot scheduled" with multiple quanta (3, 4, etc.)

    Example (Theory - L+T=5 quanta):
    - [2,2,1] across different times → 0 penalty (first isolated slot excused)
    - [2,1,1,1] → 10 penalty (3 isolated slots, only 1 excused)
    - [3,2] → 0 penalty (acceptable block sizes)

    Example (Practical - P=3 quanta):
    - [3] continuous → 0 penalty (always contiguous by design)

    Note: Theory and practical are separate course types and are NOT evaluated
    together for coalescence. Each type has independent clustering rules.

    Args:
        sessions: List of course sessions to evaluate.

    Returns:
        Total penalty for non-preferred block configurations.
    """
    cfg = get_config().time

    penalty = 0

    # Group sessions by (course_id, course_type, day) to find blocks
    course_day_quanta: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    course_type_map: dict[tuple[str, str], str] = {}  # Track course types

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

        for _day, day_quanta in course_days.items():
            # Sort quanta to identify consecutive blocks
            sorted_quanta = sorted(day_quanta)

            # Find consecutive blocks with their actual quantum values
            blocks_with_quanta = []
            if sorted_quanta:
                current_block = [sorted_quanta[0]]

                for i in range(1, len(sorted_quanta)):
                    if sorted_quanta[i] == sorted_quanta[i - 1] + 1:
                        # Consecutive - add to current block
                        current_block.append(sorted_quanta[i])
                    else:
                        # Gap - start new block
                        blocks_with_quanta.append(current_block)
                        current_block = [sorted_quanta[i]]

                # Don't forget the last block
                blocks_with_quanta.append(current_block)

            # Apply penalties based on course type
            if course_type.lower() == "practical":
                # Practical courses are always contiguous (enforced by SessionGene)
                # No fragmentation penalty needed - structurally impossible
                # Practical is "one-shot scheduled" automatically
                pass
            else:
                # Theory/Tutorial courses (L+T combined): apply clustering penalty
                # Theory and practical are evaluated SEPARATELY - no cross-type coalescence
                isolated_count = 0

                for block in blocks_with_quanta:
                    block_size = len(block)

                    if block_size == 1:
                        # Isolated single quantum - penalize for lack of clustering
                        isolated_count += 1
                        if isolated_count > cfg.theory_max_excused_isolated:
                            # Excused slots exceeded, penalize subsequent ones
                            penalty += cfg.theory_isolated_penalty
                    elif block_size > cfg.preferred_block_size_max:
                        # Oversized block - penalty per quantum beyond max
                        excess = block_size - cfg.preferred_block_size_max
                        penalty += excess * cfg.theory_oversized_penalty_per_quantum
                    # Block sizes within preferred range (2-3) have no penalty

    return penalty


@soft_constraint(
    name="paired_cohort_practical_alignment",
    description=(
        "Encourages parallel practical sessions for paired cohorts that share practical courses"
    ),
    default_weight=1.0,
    needs_courses=True,
)
def paired_cohort_practical_alignment(
    sessions: list[CourseSession],
    course_map: dict[tuple[str, ...], object],
) -> int:
    """Penalize misaligned practical sessions for paired cohorts.

    For each configured cohort pair (e.g., bei1a/bei1b) and each shared
    *practical* course, this constraint compares the sets of quanta where
    each group attends that course's practical sessions. The penalty is the
    size of the symmetric difference between these two sets, summed over all
    pairs and courses.

    A value of 0 means that for every practical course they share, both
    cohorts attend practicals in perfectly parallel time windows.
    """

    cfg = get_config()
    soft_cfg = getattr(cfg.soft_constraints, "paired_cohort_practical_alignment", None)
    if soft_cfg is not None and getattr(soft_cfg, "enabled", True) is False:
        return 0

    # Cohort pairs are configured on the time or higher-level config; fall back to []
    cohort_pairs: Iterable[tuple[str, str]] = getattr(
        getattr(cfg, "time", cfg), "cohort_pairs", []
    )

    penalty = 0

    # Index quanta per (course_id, course_type, group_id)
    course_group_quanta: dict[tuple[str, str, str], set[int]] = defaultdict(set)

    for session in sessions:
        if session.course_type.lower() != "practical":
            continue

        course_id = session.course_id
        course_type = session.course_type

        for group_id in session.group_ids:
            key = (course_id, course_type, group_id)
            course_group_quanta[key].update(session.session_quanta)

    # For each cohort pair, measure misalignment on shared practical courses
    for left_id, right_id in cohort_pairs:
        # Find practical courses present for at least one side
        practical_courses: set[tuple[str, str]] = set()

        for course_id, course_type, group_id in course_group_quanta:
            if course_type.lower() != "practical":
                continue
            if group_id in (left_id, right_id):
                practical_courses.add((course_id, course_type))

        for course_id, course_type in practical_courses:
            # Check that both sides actually attend this course
            key_left = (course_id, course_type, left_id)
            key_right = (course_id, course_type, right_id)

            if (
                key_left not in course_group_quanta
                or key_right not in course_group_quanta
            ):
                continue

            quanta_left = course_group_quanta[key_left]
            quanta_right = course_group_quanta[key_right]

            if not quanta_left and not quanta_right:
                continue

            # Symmetric difference size: quanta where exactly one cohort has the course
            diff = quanta_left.symmetric_difference(quanta_right)
            penalty += len(diff)

    return penalty


# ==========================================
# BREAK PLACEMENT COMPLIANCE
# ==========================================


def _get_break_window_quanta(
    qts: QuantumTimeSystem,
) -> dict[str, set[int]]:
    """
    Convert break window times to quantum indices per day.

    Returns:
        Dict mapping day_name -> set of within-day quanta for break window
    """
    cfg = get_config()
    break_windows: dict[str, set[int]] = {}

    for day in qts.DAY_NAMES:
        if not qts.is_operational(day):
            continue

        try:
            start_q = qts.time_to_quanta(day, cfg.time.break_window_start)
            end_q = qts.time_to_quanta(day, cfg.time.break_window_end)

            day_offset = qts.day_quanta_offset[day]
            if day_offset is None:
                continue

            # Convert to within-day quanta
            within_day_start = start_q - day_offset
            within_day_end = end_q - day_offset

            break_windows[day] = set(range(within_day_start, within_day_end))
        except ValueError:
            continue

    return break_windows


def _build_group_day_schedules(
    sessions: list[CourseSession],
    qts: QuantumTimeSystem,
) -> dict[tuple[str, str], set[int]]:
    """
    Build occupied quanta per group per day.

    Returns:
        Dict mapping (group_id, day_name) -> set of within-day occupied quanta
    """
    group_day_map: dict[tuple[str, str], set[int]] = defaultdict(set)

    for session in sessions:
        for group_id in session.group_ids:
            for quantum in session.session_quanta:
                day_name, within_day_q = quantum_to_day_and_within_day(quantum, qts)
                group_day_map[(group_id, day_name)].add(within_day_q)

    return dict(group_day_map)


@soft_constraint(
    name="break_placement_compliance",
    description="Ensures groups have proper break time during designated windows",
    default_weight=1.0,
    needs_courses=False,
)
def break_placement_compliance(sessions: list[CourseSession]) -> int:
    """
    Penalizes schedules where groups don't have breaks during designated windows.

    For each group and each operational day:
    - Identifies sessions within the break window (e.g., 12:00-14:00)
    - Counts free quanta in the break window
    - Penalizes if free quanta < break_min_quanta

    Args:
        sessions: List of decoded course sessions

    Returns:
        Total penalty for break placement violations
    """
    cfg = get_config()

    if not cfg.time.enforce_break_placement:
        return 0  # Constraint disabled

    penalty = 0
    break_penalty = cfg.time.break_violation_penalty
    min_free = cfg.time.break_min_quanta

    # Step 1: Get break window quanta for each day
    break_windows = _get_break_window_quanta(_QTS)

    # Step 2: Build group schedules per day
    group_schedules = _build_group_day_schedules(sessions, _QTS)

    # Step 3: Check each group on each day
    for (_group_id, day_name), occupied_quanta in group_schedules.items():
        if day_name not in break_windows:
            continue

        break_quanta = break_windows[day_name]

        # Count occupied quanta during break window
        occupied_in_break = occupied_quanta & break_quanta
        free_in_break = len(break_quanta) - len(occupied_in_break)

        # Penalize if insufficient free quanta
        if free_in_break < min_free:
            shortage = min_free - free_in_break
            penalty += shortage * break_penalty

    return penalty
