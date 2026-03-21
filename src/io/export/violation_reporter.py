"""
Constraint Violation Report Generator

Generates detailed human-readable reports of all constraint violations
in a schedule. Outputs to log_violations.log in the output directory.
"""

import logging
from collections import defaultdict

from src.domain.course import Course
from src.domain.session import CourseSession
from src.io.time_system import QuantumTimeSystem


def _course_display(
    course_id: str, course_type: str, course_map: dict[tuple[str, str], Course]
) -> tuple[str, str]:
    """Return a human-readable name plus the underlying course code."""

    course = course_map.get((course_id, course_type))
    base_name = course.name if course else course_id
    course_code = course.course_code if course and course.course_code else course_id
    suffix = "PR" if course_type == "practical" else "TH"
    return f"{base_name} ({suffix})", course_code


def generate_violation_report(
    sessions: list[CourseSession],
    course_map: dict[tuple[str, str], Course],
    qts: QuantumTimeSystem,
    output_path: str,
) -> None:
    """
    Generate a comprehensive violation report and save to file.

    Args:
        sessions: List of decoded course sessions
        course_map: Dictionary of courses
        qts: QuantumTimeSystem for time conversion
        output_path: Directory path where report will be saved
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CONSTRAINT VIOLATION REPORT".center(80))
    report_lines.append("=" * 80)
    report_lines.append("")

    # Generate each section
    group_violations = _check_group_overlaps(sessions, qts, course_map)
    instructor_violations = _check_instructor_conflicts(sessions, qts, course_map)
    room_violations = _check_room_conflicts(sessions, qts, course_map)
    qualification_violations = _check_instructor_qualifications(sessions, course_map)
    room_type_violations = _check_room_type_mismatches(sessions, course_map)
    availability_violations = _check_availability_violations(sessions, qts, course_map)
    schedule_violations = _check_incomplete_schedules(sessions, course_map)

    # Count totals
    total_violations = (
        len(group_violations)
        + len(instructor_violations)
        + len(room_violations)
        + len(qualification_violations)
        + len(room_type_violations)
        + len(availability_violations)
        + len(schedule_violations)
    )

    report_lines.append(f"Total Constraint Violations: {total_violations}")
    report_lines.append("")

    # Add each section to report
    if group_violations:
        report_lines.extend(_format_group_violations(group_violations))
    else:
        report_lines.append("[[!ok]]No Group Overlap Violations")
        report_lines.append("")

    if instructor_violations:
        report_lines.extend(_format_instructor_violations(instructor_violations))
    else:
        report_lines.append("[[!ok]]No Instructor Conflict Violations")
        report_lines.append("")

    if room_violations:
        report_lines.extend(_format_room_violations(room_violations))
    else:
        report_lines.append("[[!ok]]No Room Conflict Violations")
        report_lines.append("")

    if qualification_violations:
        report_lines.extend(_format_qualification_violations(qualification_violations))
    else:
        report_lines.append("[[!ok]]No Instructor Qualification Violations")
        report_lines.append("")

    if room_type_violations:
        report_lines.extend(_format_room_type_violations(room_type_violations))
    else:
        report_lines.append("[[!ok]]No Room Type Mismatch Violations")
        report_lines.append("")

    if availability_violations:
        report_lines.extend(_format_availability_violations(availability_violations))
    else:
        report_lines.append("[[!ok]]No Availability Violations")
        report_lines.append("")

    if schedule_violations:
        report_lines.extend(_format_schedule_violations(schedule_violations))
    else:
        report_lines.append("[[!ok]]No Schedule Completeness Violations")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT".center(80))
    report_lines.append("=" * 80)

    # Write to file
    from pathlib import Path

    report_file = Path(output_path) / "log_violations.log"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logging.getLogger(__name__).info("Violation report saved: %s", report_file)


def _check_group_overlaps(
    sessions: list[CourseSession],
    qts: QuantumTimeSystem,
    course_map: dict[tuple[str, str], Course],
) -> list[dict]:
    """Check for groups scheduled at the same time."""
    violations = []
    group_time_map = defaultdict(list)

    for session in sessions:
        for group_id in session.group_ids:
            for q in session.session_quanta:
                key = (group_id, q)
                group_time_map[key].append(session)

    # Find conflicts (more than one session per group-time)
    for (group_id, quantum), session_list in group_time_map.items():
        if len(session_list) > 1:
            day, time = qts.quanta_to_time(quantum)
            time_str = f"{day} {time}"
            for session in session_list:
                course_display, course_code = _course_display(
                    session.course_id, session.course_type, course_map
                )
                violations.append(
                    {
                        "group": group_id,
                        "course": course_display,
                        "course_code": course_code,
                        "room": session.room.name if session.room else session.room_id,
                        "time": time_str,
                        "instructor": (
                            session.instructor.name
                            if session.instructor
                            else session.instructor_id
                        ),
                    }
                )

    return violations


def _check_instructor_conflicts(
    sessions: list[CourseSession],
    qts: QuantumTimeSystem,
    course_map: dict[tuple[str, str], Course],
) -> list[dict]:
    """Check for instructors scheduled at the same time."""
    violations = []
    instructor_time_map = defaultdict(list)

    for session in sessions:
        for q in session.session_quanta:
            key = (session.instructor_id, q)
            instructor_time_map[key].append(session)
            # Co-instructors are also occupied
            for co_id in getattr(session, "co_instructor_ids", []):
                if co_id != session.instructor_id:
                    instructor_time_map[(co_id, q)].append(session)

    # Find conflicts
    for (instructor_id, quantum), session_list in instructor_time_map.items():
        if len(session_list) > 1:
            day, time = qts.quanta_to_time(quantum)
            time_str = f"{day} {time}"
            for session in session_list:
                course_display, course_code = _course_display(
                    session.course_id, session.course_type, course_map
                )
                violations.append(
                    {
                        "instructor": (
                            session.instructor.name
                            if session.instructor
                            else instructor_id
                        ),
                        "course": course_display,
                        "course_code": course_code,
                        "groups": ", ".join(session.group_ids),
                        "room": session.room.name if session.room else session.room_id,
                        "time": time_str,
                    }
                )

    return violations


def _check_room_conflicts(
    sessions: list[CourseSession],
    qts: QuantumTimeSystem,
    course_map: dict[tuple[str, str], Course],
) -> list[dict]:
    """Check for rooms scheduled at the same time."""
    violations = []
    room_time_map = defaultdict(list)

    for session in sessions:
        for q in session.session_quanta:
            key = (session.room_id, q)
            room_time_map[key].append(session)

    # Find conflicts
    for (room_id, quantum), session_list in room_time_map.items():
        if len(session_list) > 1:
            day, time = qts.quanta_to_time(quantum)
            time_str = f"{day} {time}"
            for session in session_list:
                course_display, course_code = _course_display(
                    session.course_id, session.course_type, course_map
                )
                violations.append(
                    {
                        "room": session.room.name if session.room else room_id,
                        "course": course_display,
                        "course_code": course_code,
                        "groups": ", ".join(session.group_ids),
                        "instructor": (
                            session.instructor.name
                            if session.instructor
                            else session.instructor_id
                        ),
                        "time": time_str,
                    }
                )

    return violations


def _check_instructor_qualifications(
    sessions: list[CourseSession], course_map: dict[tuple[str, str], Course]
) -> list[dict]:
    """Check for unqualified instructors."""
    violations = []

    for session in sessions:
        course_key = (session.course_id, session.course_type)
        if course_key not in course_map:
            continue

        course = course_map[course_key]
        if session.instructor_id not in course.qualified_instructor_ids:
            course_display, course_code = _course_display(
                session.course_id, session.course_type, course_map
            )
            violations.append(
                {
                    "course": course_display,
                    "course_code": course_code,
                    "course_type": session.course_type,
                    "instructor": (
                        session.instructor.name
                        if session.instructor
                        else session.instructor_id
                    ),
                    "groups": ", ".join(session.group_ids),
                    "room": session.room.name if session.room else session.room_id,
                }
            )
        # Check co-instructor qualifications
        for co_id in getattr(session, "co_instructor_ids", []):
            if co_id not in course.qualified_instructor_ids:
                course_display, course_code = _course_display(
                    session.course_id, session.course_type, course_map
                )
                violations.append(
                    {
                        "course": course_display,
                        "course_code": course_code,
                        "course_type": session.course_type,
                        "instructor": f"{co_id} (co-instructor)",
                        "groups": ", ".join(session.group_ids),
                        "room": session.room.name if session.room else session.room_id,
                    }
                )

    return violations


def _check_room_type_mismatches(
    sessions: list[CourseSession],
    course_map: dict[tuple[str, str], Course],
) -> list[dict]:
    """Check for room type mismatches."""
    violations = []

    for session in sessions:
        # Normalize required room features to a set[str]
        if session.required_room_features is None:
            required_features: set[str] = set()
        elif isinstance(session.required_room_features, list):
            required_features = set(session.required_room_features)
        else:
            required_features = {session.required_room_features}
        if not session.room:
            continue
        # Normalize room features to a set[str]
        if isinstance(session.room.room_features, list):
            room_features = set(session.room.room_features)
        else:
            room_features = {session.room.room_features}

        if not required_features.issubset(room_features):
            missing = required_features - room_features
            course_display, course_code = _course_display(
                session.course_id, session.course_type, course_map
            )
            violations.append(
                {
                    "course": course_display,
                    "course_code": course_code,
                    "groups": ", ".join(session.group_ids),
                    "room": session.room.name if session.room else session.room_id,
                    "required_features": ", ".join(required_features),
                    "room_features": ", ".join(room_features),
                    "missing_features": ", ".join(missing),
                }
            )

    return violations


def _check_availability_violations(
    sessions: list[CourseSession],
    qts: QuantumTimeSystem,
    course_map: dict[tuple[str, str], Course],
) -> list[dict]:
    """Check for availability violations."""
    violations = []

    for session in sessions:
        for q in session.session_quanta:
            day, time = qts.quanta_to_time(q)
            time_str = f"{day} {time}"

            course_display, course_code = _course_display(
                session.course_id, session.course_type, course_map
            )

            # Check instructor availability (full-time instructors are always available)
            if (
                session.instructor
                and not session.instructor.is_full_time
                and q not in session.instructor.available_quanta
            ):
                violations.append(
                    {
                        "type": "Instructor Unavailable",
                        "entity": (
                            session.instructor.name
                            if session.instructor
                            else session.instructor_id
                        ),
                        "course": course_display,
                        "course_code": course_code,
                        "groups": ", ".join(session.group_ids),
                        "room": session.room.name if session.room else session.room_id,
                        "time": time_str,
                    }
                )

            # Check room availability
            if session.room and q not in session.room.available_quanta:
                violations.append(
                    {
                        "type": "Room Unavailable",
                        "entity": (
                            session.room.name if session.room else session.room_id
                        ),
                        "course": course_display,
                        "course_code": course_code,
                        "groups": ", ".join(session.group_ids),
                        "instructor": (
                            session.instructor.name
                            if session.instructor
                            else session.instructor_id
                        ),
                        "time": time_str,
                    }
                )

            # Check group availability
            if session.group and q not in session.group.available_quanta:
                violations.append(
                    {
                        "type": "Group Unavailable",
                        "entity": session.group.group_id,
                        "course": course_display,
                        "course_code": course_code,
                        "room": session.room.name if session.room else session.room_id,
                        "instructor": (
                            session.instructor.name
                            if session.instructor
                            else session.instructor_id
                        ),
                        "time": time_str,
                    }
                )

    return violations


def _check_incomplete_schedules(
    sessions: list[CourseSession], course_map: dict[tuple[str, str], Course]
) -> list[dict]:
    """Check for incomplete or over-scheduled courses."""
    violations = []
    # Key is ((course_id, course_type), group_id) to match course_map structure
    course_group_quanta: dict[tuple[tuple[str, str], str], int] = defaultdict(int)

    # Count quanta per (course_id, group_id)
    for session in sessions:
        for group_id in session.group_ids:
            key = ((session.course_id, session.course_type), group_id)
            course_group_quanta[key] += len(session.session_quanta)

    # Check each course's enrolled groups
    for course_id, course in course_map.items():
        expected_quanta = course.quanta_per_week

        for group_id in course.enrolled_group_ids:
            key = (course_id, group_id)  # course_id is tuple[str, str]
            actual_quanta = course_group_quanta.get(key, 0)

            if actual_quanta != expected_quanta:
                status = (
                    "Under-scheduled"
                    if actual_quanta < expected_quanta
                    else "Over-scheduled"
                )
                course_display = (
                    f"{course.name} ({'PR' if course.course_type == 'practical' else 'TH'})"
                    if course
                    else f"{course_id[0]} ({course_id[1]})"
                )
                violations.append(
                    {
                        "course": course_display,
                        "course_code": course.course_code if course else course_id[0],
                        "group": group_id,
                        "expected": expected_quanta,
                        "actual": actual_quanta,
                        "status": status,
                    }
                )

    return violations


# Formatting functions
def _format_group_violations(violations: list[dict]) -> list[str]:
    """Format group overlap violations."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"GROUP OVERLAP VIOLATIONS: {len(violations)} found")
    lines.append("-" * 80)

    # Group by (group, time) to show all conflicts together
    conflict_groups = defaultdict(list)
    for v in violations:
        key = (v["group"], v["time"])
        conflict_groups[key].append(v)

    for (group, time), conflicts in conflict_groups.items():
        lines.append(
            f"\n[!]  Group {group} has {len(conflicts)} overlapping sessions at {time}:"
        )
        lines.extend(
            f"    - {conflict['course']} @ {conflict['room']} with {conflict['instructor']}"
            for conflict in conflicts
        )

    lines.append("")
    return lines


def _format_instructor_violations(violations: list[dict]) -> list[str]:
    """Format instructor conflict violations."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"INSTRUCTOR CONFLICT VIOLATIONS: {len(violations)} found")
    lines.append("-" * 80)

    # Group by (instructor, time)
    conflict_groups = defaultdict(list)
    for v in violations:
        key = (v["instructor"], v["time"])
        conflict_groups[key].append(v)

    for (instructor, time), conflicts in conflict_groups.items():
        lines.append(
            f"\n[!]  Instructor {instructor} has {len(conflicts)} overlapping sessions at {time}:"
        )
        lines.extend(
            f"    - {conflict['course']} with {conflict['groups']} @ {conflict['room']}"
            for conflict in conflicts
        )

    lines.append("")
    return lines


def _format_room_violations(violations: list[dict]) -> list[str]:
    """Format room conflict violations."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"ROOM CONFLICT VIOLATIONS: {len(violations)} found")
    lines.append("-" * 80)

    # Group by (room, time)
    conflict_groups = defaultdict(list)
    for v in violations:
        key = (v["room"], v["time"])
        conflict_groups[key].append(v)

    for (room, time), conflicts in conflict_groups.items():
        lines.append(
            f"\n[!]  Room {room} has {len(conflicts)} overlapping sessions at {time}:"
        )
        lines.extend(
            f"    - {conflict['course']} with {conflict['groups']} by {conflict['instructor']}"
            for conflict in conflicts
        )

    lines.append("")
    return lines


def _format_qualification_violations(violations: list[dict]) -> list[str]:
    """Format instructor qualification violations."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"INSTRUCTOR QUALIFICATION VIOLATIONS: {len(violations)} found")
    lines.append("-" * 80)

    for v in violations:
        lines.append(
            f"\n[!]  Instructor {v['instructor']} is NOT qualified for {v['course']} ({v['course_type']})"
        )
        lines.append(f"    Groups: {v['groups']}")
        lines.append(f"    Room: {v['room']}")

    lines.append("")
    return lines


def _format_room_type_violations(violations: list[dict]) -> list[str]:
    """Format room type mismatch violations."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"ROOM TYPE MISMATCH VIOLATIONS: {len(violations)} found")
    lines.append("-" * 80)

    for v in violations:
        lines.append(
            f"\n[!]  Course {v['course']} requires features not in room {v['room']}"
        )
        lines.append(f"    Groups: {v['groups']}")
        lines.append(f"    Required: {v['required_features']}")
        lines.append(f"    Room has: {v['room_features']}")
        lines.append(f"    Missing: {v['missing_features']}")

    lines.append("")
    return lines


def _format_availability_violations(violations: list[dict]) -> list[str]:
    """Format availability violations."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"AVAILABILITY VIOLATIONS: {len(violations)} found")
    lines.append("-" * 80)

    # Group by type
    by_type = defaultdict(list)
    for v in violations:
        by_type[v["type"]].append(v)

    for viol_type, viols in by_type.items():
        lines.append(f"\n{viol_type}: {len(viols)} violations")
        for v in viols:
            lines.append(f"  [!]  {v['entity']} unavailable at {v['time']}")
            lines.append(f"      Course: {v['course']}")
            if "groups" in v:
                lines.append(f"      Groups: {v['groups']}")
            if "room" in v:
                lines.append(f"      Room: {v['room']}")
            if "instructor" in v:
                lines.append(f"      Instructor: {v['instructor']}")

    lines.append("")
    return lines


def _format_schedule_violations(violations: list[dict]) -> list[str]:
    """Format schedule completeness violations."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"SCHEDULE COMPLETENESS VIOLATIONS: {len(violations)} found")
    lines.append("-" * 80)

    # Separate under/over scheduled
    under = [v for v in violations if v["status"] == "Under-scheduled"]
    over = [v for v in violations if v["status"] == "Over-scheduled"]

    if under:
        lines.append(f"\nUnder-scheduled Courses: {len(under)}")
        lines.extend(
            f"  [!]  {v['course']} for group {v['group']}: "
            f"Expected {v['expected']} quanta, got {v['actual']}"
            for v in under
        )

    if over:
        lines.append(f"\nOver-scheduled Courses: {len(over)}")
        lines.extend(
            f"  [!]  {v['course']} for group {v['group']}: "
            f"Expected {v['expected']} quanta, got {v['actual']}"
            for v in over
        )

    lines.append("")
    return lines
