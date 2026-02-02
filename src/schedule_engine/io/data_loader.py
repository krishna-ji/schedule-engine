"""Input Data JSON Encoder Module.

This module provides functions to load and encode input data from JSON files
into structured entities such as Course, Group, Instructor, and Room.
"""

from __future__ import annotations

import json
from typing import Any

from schedule_engine.domain.course import Course
from schedule_engine.domain.group import Group
from schedule_engine.domain.instructor import Instructor
from schedule_engine.domain.room import Room
from schedule_engine.io.time_system import QuantumTimeSystem

__all__ = [
    "encode_availability",
    "load_instructors",
    "load_courses",
    "load_groups",
    "load_rooms",
    "link_courses_and_groups",
    "link_courses_and_instructors",
    "derive_cohort_pairs_from_groups",
]


def encode_availability(
    availability_dict: dict[str, Any], qts: QuantumTimeSystem
) -> set[int]:
    """
    Converts human-readable availability into a set of quantum indices.
    Automatically clips availability periods to operating hours.

    Args:
        availability_dict (Dict): Availability per weekday in format {"Monday": [{"start": "08:00", "end": "10:00"}, ...]}.
        qts (QuantumTimeSystem): An instance for time conversion.

    Returns:
        set: Set of integer quantum indices available.
    """
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

            # Convert back to HH:MM format
            clipped_start = (
                f"{clipped_start_minutes // 60:02d}:{clipped_start_minutes % 60:02d}"
            )

            # For end time, if it equals operating hours end, use the last quantum
            # instead of trying to convert the boundary time
            operating_hours = qts.operating_hours[day_cap]
            if operating_hours is None:
                raise ValueError(f"Day {day_cap} has no operating hours")

            op_end_hour, op_end_minute = map(int, operating_hours[1].split(":"))
            op_end_minutes_check = op_end_hour * 60 + op_end_minute

            if clipped_end_minutes >= op_end_minutes_check:
                # Available until end of operating hours - use total day quanta
                day_offset = qts.day_quanta_offset[day_cap]
                day_count = qts.day_quanta_count[day_cap]

                if day_offset is None or day_count is None:
                    raise ValueError(f"Day {day_cap} has incomplete configuration")

                start_q = qts.time_to_quanta(day_cap, clipped_start)
                end_q = day_offset + day_count  # Exclusive end
            else:
                clipped_end = (
                    f"{clipped_end_minutes // 60:02d}:{clipped_end_minutes % 60:02d}"
                )
                start_q = qts.time_to_quanta(day_cap, clipped_start)
                end_q = qts.time_to_quanta(day_cap, clipped_end)

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
    with open(path) as f:
        data = json.load(f)
    instructors = {}
    for item in data:
        availability = item.get("availability", {})
        is_full_time = not bool(availability)
        available_quanta = (
            encode_availability(availability, qts) if availability else set()
        )

        # Parse courses - support both old flat list and new object format
        courses_data = item.get("courses", [])
        course_qualifications = []

        for course_entry in courses_data:
            if isinstance(course_entry, dict):
                # New format: {"coursecode": "ENSH 151", "coursetype": "Theory"}
                course_qualifications.append(course_entry)
            else:
                # Old format: "ENSH 151" or "ENSH 151-PR"
                # Convert to new format for backward compatibility
                if course_entry.endswith("-PR"):
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


def load_courses(path: str) -> dict[tuple[str, str], Course]:
    """
    Loads courses from FullSyllabusAll format and creates separate theory/practical course objects.

    Clean architecture: NO suffix overhead!
    - course_id = course_code (plain, e.g., "ENME 103")
    - course_type attribute = "theory" or "practical"
    - Dict keyed by (course_code, course_type) tuple for uniqueness

    Args:
        path (str): Path to the course JSON file.

    Returns:
        Dict[tuple, Course]: Dictionary keyed by (course_code, course_type) tuples.
    """
    with open(path) as f:
        data = json.load(f)
    courses = {}

    for item in data:
        course_code = item["CourseCode"].strip()
        name = item["CourseTitle"].strip()
        dept_list = [d.strip() for d in item.get("Dept", "GENERAL").split(",")]
        department = dept_list[0]
        semester = item.get("Semester", 1)
        credits = item.get("Credits", 3)

        lec = item.get("L", 0)
        tut = item.get("T", 0)
        prac = item.get("P", 0)

        # Skip non-schedulable courses (zero credits OR zero L/T/P)
        # Examples: Survey Camp, Industrial Attachment, Group Work
        if credits == 0 or (lec == 0 and tut == 0 and prac == 0):
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
                credits=credits,
                lecture_hours=lec + tut,
                practical_hours=0,
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
                credits=credits,
                lecture_hours=0,
                practical_hours=prac,
            )
            # Key by (course_code, course_type) for uniqueness
            courses[(course_code, "practical")] = course

    return courses


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
    with open(path) as f:
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

    with open(path) as file_handle:
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
    with open(path) as f:
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

        rooms[room_id] = Room(
            room_id=room_id,
            name=item.get("name", room_id),
            capacity=item["capacity"],
            room_features=room_type,  # Use type field, not features array
            available_quanta=available_quanta,
        )
    return rooms


def link_courses_and_groups(
    courses: dict[tuple[str, str], Course], groups: dict[str, Group]
) -> None:
    """
    Links courses and groups based on group enrollment.

    Args:
        courses (Dict[tuple, Course]): Courses dict keyed by (course_code, course_type).
        groups (Dict[str, Group]): Groups with enrolled course codes.
    """
    for course in courses.values():
        course.enrolled_group_ids = []

    # Collect missing courses for batch display
    missing_courses = []

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
                missing_courses.append((course_code, group_id))
            else:
                valid_courses.append(course_code)

        # Remove non-schedulable/missing courses from group enrollments
        group.enrolled_courses = valid_courses

    # Display missing courses in a table if any found
    if missing_courses:
        from rich.console import Console

        console = Console()

        console.print()
        console.print("[yellow]️  Non-schedulable or missing courses skipped[/yellow]")

        # Group by course code for compact display
        from collections import defaultdict

        courses_by_code = defaultdict(list)
        for course_code, group_id in missing_courses:
            courses_by_code[course_code].append(group_id)

        for course_code, group_ids in sorted(courses_by_code.items()):
            groups_str = ", ".join(sorted(group_ids))
            console.print(
                f"  [dim]{course_code}:[/dim] {groups_str} [dim](non-schedulable)[/dim]"
            )

        console.print(
            f"  [dim]{len(missing_courses)} enrollments skipped (Survey Camp, Industrial Attachment, etc.)[/dim]"
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
