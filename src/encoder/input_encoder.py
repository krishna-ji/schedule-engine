"""
Input Data JSON Encoder Module

This module provides functions to load and encode input data from JSON files
into structured entities such as Course, Group, Instructor, and Room.

Functions:
- encode_availability
- load_instructors
- load_courses
- load_groups
- load_rooms
- link_courses_and_groups
- link_courses_and_instructors
"""

import json
from typing import Dict

from src.entities.course import Course
from src.entities.group import Group
from src.entities.instructor import Instructor
from src.entities.room import Room
from src.encoder.quantum_time_system import QuantumTimeSystem


def encode_availability(availability_dict: Dict, qts: QuantumTimeSystem) -> set:
    """
    Converts human-readable availability into a set of quantum indices.
    Automatically clips availability periods to operating hours.

    Args:
        availability_dict (Dict): Availability per weekday in format {"Monday": [{"start": "08:00", "end": "10:00"}, ...]}.
        qts (QuantumTimeSystem): An instance for time conversion.

    Returns:
        set: Set of integer quantum indices available.
    """
    quanta = set()
    for day, periods in availability_dict.items():
        day_cap = day.capitalize()

        # Skip if day is not operational
        if not qts.is_operational(day_cap):
            continue

        # Get operating hours for this day
        operating_hours = qts.operating_hours[day_cap]
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
            op_end_hour, op_end_minute = map(
                int, qts.operating_hours[day_cap][1].split(":")
            )
            op_end_minutes_check = op_end_hour * 60 + op_end_minute

            if clipped_end_minutes >= op_end_minutes_check:
                # Available until end of operating hours - use total day quanta
                day_offset = qts.day_quanta_offset[day_cap]
                day_count = qts.day_quanta_count[day_cap]
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


def load_instructors(path: str, qts: QuantumTimeSystem) -> Dict[str, Instructor]:
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


def load_courses(path: str) -> Dict[tuple, Course]:
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

        L = item.get("L", 0)
        T = item.get("T", 0)
        P = item.get("P", 0)

        practical_features = item.get("PracticalRoomFeatures", "").strip()
        practical_features = [
            f.strip().lower() for f in practical_features.split(",") if f.strip()
        ]

        # Create theory course object if L + T > 0
        if L + T > 0:
            course = Course(
                course_id=course_code,  # Plain course code, no suffix!
                name=f"{name} (Theory)",
                quanta_per_week=int(L + T),
                required_room_features="lecture",  # Simple string, not list
                enrolled_group_ids=[],
                qualified_instructor_ids=[],
                course_type="theory",
            )
            course.course_code = course_code
            course.department = department
            course.semester = semester
            course.credits = credits
            course.lecture_hours = L + T
            course.practical_hours = 0
            # Key by (course_code, course_type) for uniqueness
            courses[(course_code, "theory")] = course

        # Create practical course object if P > 0
        if P > 0:
            course = Course(
                course_id=course_code,  # Same course_id as theory!
                name=f"{name} (Practical)",
                quanta_per_week=int(P),
                required_room_features="practical",  # Simple string, not list
                enrolled_group_ids=[],
                qualified_instructor_ids=[],
                course_type="practical",
            )
            course.course_code = course_code
            course.department = department
            course.semester = semester
            course.credits = credits
            course.lecture_hours = 0
            course.practical_hours = P
            # Key by (course_code, course_type) for uniqueness
            courses[(course_code, "practical")] = course

    return courses


def load_groups(path: str, qts: QuantumTimeSystem) -> Dict[str, Group]:
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


def load_rooms(path: str, qts: QuantumTimeSystem) -> Dict[str, Room]:
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
    courses: Dict[tuple, Course], groups: Dict[str, Group]
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

    # Display missing courses in a table if any found
    if missing_courses:
        from rich.console import Console

        console = Console()

        console.print()
        console.print("[yellow][!warn] groups enrolled but courses missing[/yellow]")

        # Group by course code for compact display
        from collections import defaultdict

        courses_by_code = defaultdict(list)
        for course_code, group_id in missing_courses:
            courses_by_code[course_code].append(group_id)

        for course_code, group_ids in sorted(courses_by_code.items()):
            groups_str = ", ".join(sorted(group_ids))
            console.print(
                f"  [dim]{course_code}:[/dim] {groups_str} [dim](ltp null)[/dim]"
            )

        console.print(f"  [dim]{len(missing_courses)} course enrollments skipped[/dim]")
        console.print()

    # Note: We no longer warn about unassigned courses here since filtering
    # happens in load_input_data() before this function is called


def link_courses_and_instructors(
    courses: Dict[tuple, Course], instructors: Dict[str, Instructor]
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
        instructor.original_qualified_courses preserves the raw JSON data for validation.
    """
    # Store original qualified courses BEFORE clearing
    instructor_original_courses = {}
    for instructor_id, instructor in instructors.items():
        instructor_original_courses[instructor_id] = instructor.qualified_courses[:]
        if not hasattr(instructor, "original_qualified_courses"):
            instructor.original_qualified_courses = instructor.qualified_courses[:]
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
