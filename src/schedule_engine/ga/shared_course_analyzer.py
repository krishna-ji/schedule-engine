"""
Shared Course Analyzer

Pre-computes "hotspot" information for courses shared across multiple departments.
These courses require extra care in scheduling since they create contention for
instructor and room resources.

Example: ENCT 101 (Computer Programming) is taken by BAM, BCE, BCT, BEI, BIE, BME
This means 6× instructor sessions and 12× practical room slots needed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from schedule_engine.domain.course import Course
from schedule_engine.domain.group import Group


@dataclass
class SharedCourseInfo:
    """Information about a course shared across groups/departments."""

    course_key: tuple[str, str]  # (course_code, course_type)
    course_title: str
    group_ids: list[str]  # All groups taking this course
    department_count: int  # Number of distinct departments
    total_sessions: int  # Total sessions needed (theory + practicals)
    quanta_per_week: int
    is_hotspot: bool  # True if high contention (>3 departments)

    # Pre-computed scheduling constraints
    min_timeslot_spread: int = 1  # Minimum quanta apart for different sessions
    requires_multiple_instructors: bool = False
    requires_multiple_rooms: bool = False


@dataclass
class SharedCourseAnalysis:
    """Complete analysis of shared courses in the problem."""

    shared_courses: dict[tuple[str, str], SharedCourseInfo] = field(
        default_factory=dict
    )
    hotspot_courses: list[tuple[str, str]] = field(default_factory=list)

    # Course -> [group_ids] for quick lookup
    course_to_groups: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    # Group -> [course_keys] for quick lookup
    group_to_shared_courses: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict
    )

    # Statistics
    total_shared_courses: int = 0
    total_hotspot_courses: int = 0
    max_department_overlap: int = 0


def analyze_shared_courses(
    courses: dict[tuple[str, str], Course],
    groups: dict[str, Group],
) -> SharedCourseAnalysis:
    """
    Analyze courses to identify those shared across multiple groups/departments.

    A "hotspot" course is one taken by 4+ departments, creating scheduling contention.

    Args:
        courses: Dict of (course_code, course_type) -> Course
        groups: Dict of group_id -> Group

    Returns:
        SharedCourseAnalysis with pre-computed hotspot information
    """
    analysis = SharedCourseAnalysis()

    # Build course -> groups mapping from group enrollments
    course_to_groups: dict[tuple[str, str], list[str]] = defaultdict(list)

    for group_id, group in groups.items():
        for course_code in group.enrolled_courses:
            # Check both theory and practical variants
            theory_key = (course_code, "theory")
            practical_key = (course_code, "practical")

            if theory_key in courses:
                course_to_groups[theory_key].append(group_id)
            if practical_key in courses:
                course_to_groups[practical_key].append(group_id)

    # Analyze each course
    for course_key, group_ids in course_to_groups.items():
        course = courses.get(course_key)
        if not course:
            continue

        # Extract departments from group IDs (first 3 chars typically: BCE, BME, etc.)
        departments = set()
        for gid in group_ids:
            # Department is typically first 3 characters (BCE1A -> BCE)
            dept = gid[:3] if len(gid) >= 3 else gid
            departments.add(dept)

        dept_count = len(departments)

        # Calculate total sessions needed
        if course.course_type == "practical":
            # Each group gets separate practical
            total_sessions = len(group_ids)
        else:
            # Theory: depends on parent-subgroup structure
            # Count unique parent groups (BME1A, BME1B -> 1 session for BME1)
            parent_groups = set()
            for gid in group_ids:
                # Remove trailing letter if it's a subgroup
                if len(gid) > 1 and gid[-1].isalpha() and gid[-2].isdigit():
                    parent_groups.add(gid[:-1])
                else:
                    parent_groups.add(gid)
            total_sessions = len(parent_groups)

        is_hotspot = dept_count >= 4 or len(group_ids) >= 8

        info = SharedCourseInfo(
            course_key=course_key,
            course_title=course.course_name,
            group_ids=list(group_ids),
            department_count=dept_count,
            total_sessions=total_sessions,
            quanta_per_week=course.quanta_per_week,
            is_hotspot=is_hotspot,
            requires_multiple_instructors=dept_count >= 3,
            requires_multiple_rooms=total_sessions >= 4,
        )

        analysis.shared_courses[course_key] = info
        analysis.course_to_groups[course_key] = list(group_ids)

        for gid in group_ids:
            if gid not in analysis.group_to_shared_courses:
                analysis.group_to_shared_courses[gid] = []
            analysis.group_to_shared_courses[gid].append(course_key)

        if is_hotspot:
            analysis.hotspot_courses.append(course_key)
            analysis.total_hotspot_courses += 1

        if dept_count > analysis.max_department_overlap:
            analysis.max_department_overlap = dept_count

    analysis.total_shared_courses = len(analysis.shared_courses)

    return analysis


def get_course_scheduling_priority(
    course_key: tuple[str, str],
    analysis: SharedCourseAnalysis,
) -> int:
    """
    Get scheduling priority for a course (lower = schedule first).

    Hotspot courses should be scheduled first to ensure they get
    good timeslots before contention increases.

    Returns:
        Priority value (1-10, where 1 is highest priority)
    """
    info = analysis.shared_courses.get(course_key)
    if not info:
        return 5  # Default priority

    if info.is_hotspot:
        return 1  # Highest priority
    elif info.department_count >= 3:
        return 2
    elif info.total_sessions >= 4:
        return 3
    else:
        return 5


def sort_course_group_pairs_by_priority(
    pairs: list[tuple[tuple[str, str], list[str], str, int]],
    analysis: SharedCourseAnalysis,
) -> list[tuple[tuple[str, str], list[str], str, int]]:
    """
    Sort course-group pairs by scheduling priority.

    Hotspot courses first, then by number of groups, then by session type
    (theory before practical to ensure shared sessions are placed first).

    This helps the initialization place hard-to-schedule courses early
    when more timeslots are available.
    """

    def priority_key(
        pair: tuple[tuple[str, str], list[str], str, int],
    ) -> tuple[int, int, int]:
        course_key, group_ids, session_type, _ = pair

        # Primary: scheduling priority (lower = first)
        sched_priority = get_course_scheduling_priority(course_key, analysis)

        # Secondary: number of groups (more groups = harder, schedule first)
        num_groups = -len(group_ids)  # Negative for descending

        # Tertiary: theory before practical
        type_priority = 0 if session_type == "theory" else 1

        return (sched_priority, num_groups, type_priority)

    return sorted(pairs, key=priority_key)


def find_instructor_availability_for_course(
    course_key: tuple[str, str],
    instructors: dict,
    analysis: SharedCourseAnalysis,
) -> dict[str, int]:
    """
    Find instructors qualified for a course and their availability scores.

    For hotspot courses, we need multiple instructors, so this returns
    availability-weighted options.

    Returns:
        Dict mapping instructor_id -> availability_score (higher = more available)
    """
    info = analysis.shared_courses.get(course_key)
    sessions_needed = info.total_sessions if info else 1

    instructor_scores: dict[str, int] = {}

    for inst_id, instructor in instructors.items():
        if course_key in getattr(instructor, "qualified_courses", []):
            # Score based on availability (more available quanta = higher score)
            available = len(getattr(instructor, "available_quanta", set()))

            # Penalize if instructor doesn't have enough time for all sessions
            quanta_needed = (info.quanta_per_week if info else 2) * sessions_needed
            if available < quanta_needed:
                score = available // 2  # Reduced score
            else:
                score = available

            instructor_scores[inst_id] = score

    return instructor_scores
