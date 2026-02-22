"""Input Data JSON Encoder Module.

This module provides functions to load and encode input data from JSON files
into structured entities such as Course, Group, Instructor, and Room.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.domain.course import Course
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room

if TYPE_CHECKING:
    from src.io.time_system import QuantumTimeSystem

__all__ = [
    "derive_cohort_pairs_from_groups",
    "encode_availability",
    "link_courses_and_groups",
    "link_courses_and_instructors",
    "load_courses",
    "load_groups",
    "load_instructors",
    "load_rooms",
]


def encode_availability(
    availability_dict: dict[str, Any], qts: QuantumTimeSystem
) -> set[int]:
    """
    Converts human-readable availability into a set of quantum indices.
    Automatically clips availability periods to operating hours.

    Start times are rounded **up** (ceiling) to the next quantum boundary
    so the instructor is only marked available for quanta they fully cover.
    End times are rounded **down** (floor) for the same reason.

    Args:
        availability_dict: Availability per weekday, e.g.
            ``{"Monday": [{"start": "08:00", "end": "10:00"}]}``.
        qts: An instance for time conversion.

    Returns:
        set: Set of integer quantum indices available.
    """
    import math

    quanta: set[int] = set()
    for day, periods in availability_dict.items():
        day_cap = day.capitalize()

        # Skip if day is not operational
        if not qts.is_operational(day_cap):
            continue

        # Skip if periods is None
        if periods is None:
            continue

        # Get operating hours for this day
        operating_hours = qts.operating_hours[day_cap]
        if operating_hours is None:
            continue
        op_start_str, op_end_str = operating_hours
        op_start_minutes = int(op_start_str.split(":")[0]) * 60 + int(
            op_start_str.split(":")[1]
        )
        op_end_minutes = int(op_end_str.split(":")[0]) * 60 + int(
            op_end_str.split(":")[1]
        )

        day_offset = qts.day_quanta_offset[day_cap]
        if day_offset is None:
            continue

        for period in periods:
            # Parse availability period
            avail_start = period["start"]
            avail_end = period["end"]
            avail_start_minutes = int(avail_start.split(":")[0]) * 60 + int(
                avail_start.split(":")[1]
            )
            avail_end_minutes = int(avail_end.split(":")[0]) * 60 + int(
                avail_end.split(":")[1]
            )

            # Clip to operating hours
            clipped_start_minutes = max(avail_start_minutes, op_start_minutes)
            clipped_end_minutes = min(avail_end_minutes, op_end_minutes)

            # Skip if no overlap with operating hours
            if clipped_start_minutes >= clipped_end_minutes:
                continue

            # ── Quantum-safe rounding ──────────────────────────────
            # Start: round UP (ceil) — instructor must be available for
            # the ENTIRE quantum, not just the tail end.
            start_from_day = clipped_start_minutes - op_start_minutes
            start_q = day_offset + math.ceil(start_from_day / qts.QUANTUM_MINUTES)

            # End: round DOWN (floor) — same logic for the last quantum.
            # For end time, if it equals operating hours end, use the
            # last quantum boundary instead of trying to convert the
            # boundary time.
            operating_hours = qts.operating_hours[day_cap]
            if operating_hours is None:
                raise ValueError(f"Day {day_cap} has no operating hours")

            op_end_hour, op_end_minute = map(int, operating_hours[1].split(":"))
            op_end_minutes_check = op_end_hour * 60 + op_end_minute

            if clipped_end_minutes >= op_end_minutes_check:
                day_count = qts.day_quanta_count[day_cap]
                if day_count is None:
                    raise ValueError(f"Day {day_cap} has incomplete configuration")
                end_q = day_offset + day_count  # Exclusive end
            else:
                end_from_day = clipped_end_minutes - op_start_minutes
                end_q = day_offset + end_from_day // qts.QUANTUM_MINUTES

            # Only add if at least one full quantum fits
            if start_q < end_q:
                quanta.update(range(start_q, end_q))

    return quanta


def load_instructors(path: str, qts: QuantumTimeSystem) -> dict[str, Instructor]:
    """
    Loads instructor data from JSON file and encodes their availability.

    New format supports courses as list of objects with 'coursecode' and 'coursetype'.
    Old format supported courses as flat list of strings.

    Args:
        path (str): Path to JSON file.
        qts (QuantumTimeSystem): Time conversion system.

    Returns:
        Dict[str, Instructor]: Dictionary mapping instructor IDs to Instructor instances.
    """
    with Path(path).open() as f:
        data = json.load(f)
    instructors = {}
    for item in data:
        availability = item.get("availability", {})
        available_quanta = (
            encode_availability(availability, qts) if availability else set()
        )

        # Determine full-time status:
        # - No availability specified = full-time (available all operating hours)
        # - Availability specified but encodes to empty = treat as full-time
        #   (happens when specified hours are outside operating hours, e.g., 7:00-8:30 vs 10:00-17:00)
        # - Availability specified and has valid quanta = part-time
        is_full_time = not bool(availability) or not available_quanta
        if not available_quanta:
            # Full-time instructors are available during all operating hours
            available_quanta = (
                set()
            )  # Will be filled by caller or treated as "always available"

        # Parse courses - support both old flat list and new object format
        courses_data = item.get("courses", [])
        course_qualifications = []

        for course_entry in courses_data:
            if isinstance(course_entry, dict):
                # New format: {"coursecode": "ENSH 151", "coursetype": "Theory"}
                course_qualifications.append(course_entry)
            # Old format: "ENSH 151" or "ENSH 151-PR"
            # Convert to new format for backward compatibility
            elif course_entry.endswith("-PR"):
                course_qualifications.append(
                    {"coursecode": course_entry[:-3], "coursetype": "Practical"}
                )
            else:
                course_qualifications.append(
                    {"coursecode": course_entry, "coursetype": "Theory"}
                )

        instructors[item["id"]] = Instructor(
            instructor_id=item["id"],
            name=item["name"],
            qualified_courses=course_qualifications,
            is_full_time=is_full_time,
            available_quanta=available_quanta,
        )
    return instructors


def load_courses(
    path: str,
) -> tuple[dict[tuple[str, str], Course], dict[str, dict[str, int]]]:
    """
    Loads courses from FullSyllabusAll format and creates separate theory/practical course objects.

    Clean architecture: NO suffix overhead!
    - course_id = course_code (plain, e.g., "ENME 103")
    - course_type attribute = "theory" or "practical"
    - Dict keyed by (course_code, course_type) tuple for uniqueness

    Also returns a dict of courses that were skipped because they are
    non-schedulable (zero credits or zero L/T/P hours).

    Args:
        path (str): Path to the course JSON file.

    Returns:
        Tuple of:
          - Dict[tuple, Course]: Courses keyed by (course_code, course_type).
          - Dict[str, dict]: Skipped non-schedulable courses keyed by course_code,
            values are {"L": int, "T": int, "P": int, "credits": int}.
    """
    with Path(path).open() as f:
        data = json.load(f)
    courses = {}
    skipped_courses: dict[str, dict[str, int]] = {}

    for item in data:
        course_code = item["CourseCode"].strip()
        name = item["CourseTitle"].strip()
        dept_list = [d.strip() for d in item.get("Dept", "GENERAL").split(",")]
        department = dept_list[0]
        semester = item.get("Semester", 1)
        credit_hours = item.get("Credits", 3)

        lec = item.get("L", 0)
        tut = item.get("T", 0)
        prac = item.get("P", 0)

        # Skip non-schedulable courses (zero L/T/P hours → no classroom sessions)
        # Rationale: If a course has no lecture, tutorial, or practical hours,
        # there is nothing to schedule (e.g. Survey Camp, Industrial Attachment).
        # Courses with Credits=0 but valid L/T/P hours ARE still scheduled.
        if lec == 0 and tut == 0 and prac == 0:
            skipped_courses[course_code] = {
                "L": int(lec),
                "T": int(tut),
                "P": int(prac),
                "credits": int(credit_hours),
            }
            continue

        practical_features = item.get("PracticalRoomFeatures", "").strip()
        practical_features = [
            f.strip().lower() for f in practical_features.split(",") if f.strip()
        ]

        # Create theory course object if lec + tut > 0
        if lec + tut > 0:
            course = Course(
                course_id=course_code,  # Plain course code, no suffix!
                name=f"{name} (Theory)",
                quanta_per_week=int(lec + tut),
                required_room_features="lecture",  # Simple string, not list
                enrolled_group_ids=[],
                qualified_instructor_ids=[],
                course_type="theory",
                L=int(lec),  # Store lecture hours
                T=int(tut),  # Store tutorial hours
                P=0,
                course_code=course_code,
                department=department,
                semester=semester,
                credits=credit_hours,
                lecture_hours=lec + tut,
                practical_hours=0,
                specific_lab_features=[],  # Theory courses don't need lab features
            )
            # Key by (course_code, course_type) for uniqueness
            courses[(course_code, "theory")] = course

        # Create practical course object if prac > 0
        if prac > 0:
            course = Course(
                course_id=course_code,  # Same course_id as theory!
                name=f"{name} (Practical)",
                quanta_per_week=int(prac),
                required_room_features="practical",  # Simple string, not list
                enrolled_group_ids=[],
                qualified_instructor_ids=[],
                course_type="practical",
                L=0,
                T=0,
                P=int(prac),  # Store practical hours
                course_code=course_code,
                department=department,
                semester=semester,
                credits=credit_hours,
                lecture_hours=0,
                practical_hours=prac,
                specific_lab_features=practical_features,  # e.g. ["networking lab", "general programming lab"]
            )
            # Key by (course_code, course_type) for uniqueness
            courses[(course_code, "practical")] = course

    return courses, skipped_courses


def load_groups(path: str, qts: QuantumTimeSystem) -> dict[str, Group]:
    """
    Loads student group data and encodes availability.

    NEW ARCHITECTURE: No parent groups!
    - Only creates subgroup entities if subgroups exist
    - If no subgroups, creates the group itself
    - All groups are independent, no parent-child relationship

    Args:
        path (str): Path to group JSON file.
        qts (QuantumTimeSystem): Time system to encode availability.

    Returns:
        Dict[str, Group]: Dictionary of group IDs to Group instances.
    """
    with Path(path).open() as f:
        data = json.load(f)
    groups = {}

    for item in data:
        group_availability = item.get("availability", {})
        available_quanta = (
            encode_availability(group_availability, qts)
            if group_availability
            else qts.get_all_operating_quanta()
        )

        # Check if subgroups exist
        subgroups_data = item.get("subgroups", [])

        if subgroups_data:
            # If subgroups exist, ONLY create subgroups (no parent)
            for subgroup in subgroups_data:
                # Handle both old format (string) and new format (dict with id and student_count)
                if isinstance(subgroup, dict):
                    subgroup_id = subgroup["id"]
                    subgroup_count = subgroup.get(
                        "student_count", item["student_count"] // len(subgroups_data)
                    )
                else:
                    # Old format: subgroup is just a string ID
                    subgroup_id = subgroup
                    subgroup_count = item["student_count"] // len(subgroups_data)

                # Create subgroup with inherited courses and availability
                groups[subgroup_id] = Group(
                    group_id=subgroup_id,
                    name=f"{item['name']} - {subgroup_id}",
                    student_count=subgroup_count,
                    enrolled_courses=item.get("courses", []),
                    available_quanta=available_quanta,
                )
        else:
            # No subgroups, create the group itself
            group_id = item["group_id"]
            groups[group_id] = Group(
                group_id=group_id,
                name=item["name"],
                student_count=item["student_count"],
                enrolled_courses=item.get("courses", []),
                available_quanta=available_quanta,
            )

    return groups


def derive_cohort_pairs_from_groups(path: str) -> list[tuple[str, str]]:
    """Auto-derive cohort pairings from the group hierarchy JSON.

    The helper scans every parent entry in ``Groups.json`` and pairs the listed
    subgroups so SC5 can reason about practical alignment without manual config
    tweaks. The first subgroup acts as the anchor and every remaining subgroup
    in that parent is paired against it (A with B, A with C, ...). This keeps the
    pair count compact even if a cohort splits into 3+ lab sections.

    Args:
        path: Absolute path to ``Groups.json``.

    Returns:
        Deterministic list of subgroup ID tuples ready for ``SchedulingContext``.
    """

    with Path(path).open() as file_handle:
        raw_data = json.load(file_handle)

    derived_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for item in raw_data:
        subgroups = item.get("subgroups")
        if not subgroups or len(subgroups) < 2:
            continue

        subgroup_ids = _extract_subgroup_ids(subgroups)
        if len(subgroup_ids) < 2:
            continue

        anchor_id = subgroup_ids[0]
        for peer_id in subgroup_ids[1:]:
            candidate = (anchor_id, peer_id)
            canonical_parts: list[str] = sorted((anchor_id.lower(), peer_id.lower()))
            canonical_key = (canonical_parts[0], canonical_parts[1])
            if canonical_key in seen_pairs:
                continue
            seen_pairs.add(canonical_key)
            derived_pairs.append(candidate)

    return derived_pairs


def _extract_subgroup_ids(subgroups: list[Any]) -> list[str]:
    """Normalize subgroup entries to clean string identifiers."""

    normalized: list[str] = []
    seen: set[str] = set()

    for raw_entry in subgroups:
        subgroup_id: str | None
        if isinstance(raw_entry, dict):
            subgroup_id = raw_entry.get("id")
        else:
            subgroup_id = str(raw_entry)

        if subgroup_id is None:
            continue

        clean_id = subgroup_id.strip()
        if not clean_id:
            continue

        canonical = clean_id.lower()
        if canonical in seen:
            continue

        seen.add(canonical)
        normalized.append(clean_id)

    return normalized


def load_rooms(path: str, qts: QuantumTimeSystem) -> dict[str, Room]:
    """
    Loads room data from JSON and encodes availability.

    Uses 'type' field for room_features (e.g., "Practical", "Lecture") to match
    course requirements. The 'features' array contains specific capabilities but
    isn't used for general room type matching.

    Args:
        path (str): Path to room JSON file.
        qts (QuantumTimeSystem): Time conversion system.

    Returns:
        Dict[str, Room]: Dictionary of room IDs to Room objects.
    """
    with Path(path).open() as f:
        data = json.load(f)
    rooms = {}
    for item in data:
        room_id = item["room_id"]
        if room_id in rooms:
            raise ValueError(f"Duplicate room ID found: {room_id}")
        availability = item.get("availability", {})
        available_quanta = (
            encode_availability(availability, qts)
            if availability
            else qts.get_all_operating_quanta()
        )

        # Use 'type' field for room_features (normalized to lowercase)
        # This matches course.required_room_features format
        # "Practical" -> "practical", "Lecture" -> "lecture"
        room_type = item.get("type", "Lecture").strip().lower()

        # Parse specific features array (e.g. ["Networking Lab", "General Programming Lab"])
        specific_features = [
            f.strip().lower() for f in item.get("features", []) if f.strip()
        ]

        rooms[room_id] = Room(
            room_id=room_id,
            name=item.get("name", room_id),
            capacity=item["capacity"],
            room_features=room_type,  # Use type field, not features array
            available_quanta=available_quanta,
            specific_features=specific_features,  # e.g. ["networking lab", "drawing hall"]
        )
    return rooms


def link_courses_and_groups(
    courses: dict[tuple[str, str], Course],
    groups: dict[str, Group],
    skipped_courses: dict[str, dict[str, int]] | None = None,
) -> None:
    """
    Links courses and groups based on group enrollment.

    Args:
        courses: Courses dict keyed by (course_code, course_type).
        groups: Groups with enrolled course codes.
        skipped_courses: Optional dict of non-schedulable courses from load_courses().
            When provided, only non-schedulable courses are displayed (with L+T / P reason).
            Courses not found in Course.json at all are silently ignored.
    """
    for course in courses.values():
        course.enrolled_group_ids = []

    # Collect non-schedulable courses for batch display
    # (silently ignore courses that are completely missing from Course.json)
    non_schedulable_entries: list[tuple[str, str]] = []  # (course_code, group_id)

    # Link groups to ALL courses with matching course_code (theory AND practical)
    for group_id, group in groups.items():
        valid_courses: list[str] = []
        for course_code in group.enrolled_courses:
            # Check for both theory and practical versions
            theory_key = (course_code, "theory")
            practical_key = (course_code, "practical")

            found_any = False
            if theory_key in courses:
                if group_id not in courses[theory_key].enrolled_group_ids:
                    courses[theory_key].enrolled_group_ids.append(group_id)
                found_any = True

            if practical_key in courses:
                if group_id not in courses[practical_key].enrolled_group_ids:
                    courses[practical_key].enrolled_group_ids.append(group_id)
                found_any = True

            if not found_any:
                # Only track if it was explicitly skipped as non-schedulable
                if skipped_courses and course_code in skipped_courses:
                    non_schedulable_entries.append((course_code, group_id))
                # else: truly missing from Course.json — silently ignore
            else:
                valid_courses.append(course_code)

        # Remove non-schedulable/missing courses from group enrollments
        group.enrolled_courses = valid_courses

    # Display non-schedulable courses if any found
    if non_schedulable_entries and skipped_courses:
        from collections import defaultdict

        from rich.console import Console

        console = Console()

        console.print()
        console.print("[yellow]Non-schedulable courses skipped[/yellow]")

        # Group by course code for compact display
        courses_by_code: dict[str, list[str]] = defaultdict(list)
        for course_code, group_id in non_schedulable_entries:
            courses_by_code[course_code].append(group_id)

        for course_code, group_ids in sorted(courses_by_code.items()):
            groups_str = ", ".join(sorted(group_ids))
            info = skipped_courses.get(course_code, {})
            lt = info.get("L", 0) + info.get("T", 0)
            p = info.get("P", 0)
            console.print(
                f"  [dim]{course_code}:[/dim] {groups_str} [dim](L+T={lt} P={p})[/dim]"
            )

        console.print(
            f"  [dim]{len(non_schedulable_entries)} enrollments skipped[/dim]"
        )
        console.print()

    # Note: We no longer warn about unassigned courses here since filtering
    # happens in load_input_data() before this function is called


def link_courses_and_instructors(
    courses: dict[tuple[str, str], Course], instructors: dict[str, Instructor]
) -> None:
    """
    Links instructors to the courses they are qualified to teach.

    Instructor qualified_courses contains dicts with 'coursecode' and 'coursetype'.
    Maps instructors to specific course types based on coursetype field.

    Args:
        courses (Dict[tuple, Course]): Course dict keyed by (course_code, course_type).
        instructors (Dict[str, Instructor]): Instructor dictionary.

    Note:
        After linking, instructor.qualified_courses contains (course_code, course_type) tuples.
        Original qualifications are stored in instructor_original_courses dict for validation.
    """
    # Store original qualified courses BEFORE clearing
    instructor_original_courses: dict[str, list[Any]] = {}
    for instructor_id, instructor in instructors.items():
        instructor_original_courses[instructor_id] = instructor.qualified_courses[:]
        instructor.qualified_courses = []

    for course in courses.values():
        course.qualified_instructor_ids = []

    # Link instructors to courses based on coursecode and coursetype
    for instructor_id, instructor in instructors.items():
        for qual_entry in instructor_original_courses[instructor_id]:
            course_code = qual_entry["coursecode"]
            course_type = qual_entry["coursetype"].lower()  # "theory" or "practical"

            # Direct lookup using tuple key
            course_key = (course_code, course_type)

            if course_key in courses:
                course = courses[course_key]
                if instructor_id not in course.qualified_instructor_ids:
                    course.qualified_instructor_ids.append(instructor_id)
                if course_key not in instructor.qualified_courses:
                    instructor.qualified_courses.append(course_key)
