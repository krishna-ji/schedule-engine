"""
Constraint Violation Report Generator

Generates detailed human-readable reports of all constraint violations
in a schedule. Outputs to violation_report.txt in the output directory.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
from src.entities.decoded_session import CourseSession
from src.entities.course import Course
from src.encoder.quantum_time_system import QuantumTimeSystem


def generate_violation_report(
    sessions: List[CourseSession],
    course_map: Dict[str, Course],
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
    group_violations = _check_group_overlaps(sessions, qts)
    instructor_violations = _check_instructor_conflicts(sessions, qts)
    room_violations = _check_room_conflicts(sessions, qts)
    qualification_violations = _check_instructor_qualifications(sessions, course_map)
    room_type_violations = _check_room_type_mismatches(sessions)
    availability_violations = _check_availability_violations(sessions, qts)
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
        report_lines.append("[OK}No Group Overlap Violations")
        report_lines.append("")

    if instructor_violations:
        report_lines.extend(_format_instructor_violations(instructor_violations))
    else:
        report_lines.append("[OK}No Instructor Conflict Violations")
        report_lines.append("")

    if room_violations:
        report_lines.extend(_format_room_violations(room_violations))
    else:
        report_lines.append("[OK}No Room Conflict Violations")
        report_lines.append("")

    if qualification_violations:
        report_lines.extend(_format_qualification_violations(qualification_violations))
    else:
        report_lines.append("[OK}No Instructor Qualification Violations")
        report_lines.append("")

    if room_type_violations:
        report_lines.extend(_format_room_type_violations(room_type_violations))
    else:
        report_lines.append("[OK}No Room Type Mismatch Violations")
        report_lines.append("")

    if availability_violations:
        report_lines.extend(_format_availability_violations(availability_violations))
    else:
        report_lines.append("[OK}No Availability Violations")
        report_lines.append("")

    if schedule_violations:
        report_lines.extend(_format_schedule_violations(schedule_violations))
    else:
        report_lines.append("[OK}No Schedule Completeness Violations")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT".center(80))
    report_lines.append("=" * 80)

    # Write to file
    import os

    report_file = os.path.join(output_path, "violation_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Violation report saved: {report_file}")


def _check_group_overlaps(
    sessions: List[CourseSession], qts: QuantumTimeSystem
) -> List[Dict]:
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
                violations.append(
                    {
                        "group": group_id,
                        "course": session.course_id,
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
    sessions: List[CourseSession], qts: QuantumTimeSystem
) -> List[Dict]:
    """Check for instructors scheduled at the same time."""
    violations = []
    instructor_time_map = defaultdict(list)

    for session in sessions:
        for q in session.session_quanta:
            key = (session.instructor_id, q)
            instructor_time_map[key].append(session)

    # Find conflicts
    for (instructor_id, quantum), session_list in instructor_time_map.items():
        if len(session_list) > 1:
            day, time = qts.quanta_to_time(quantum)
            time_str = f"{day} {time}"
            for session in session_list:
                violations.append(
                    {
                        "instructor": (
                            session.instructor.name
                            if session.instructor
                            else instructor_id
                        ),
                        "course": session.course_id,
                        "groups": ", ".join(session.group_ids),
                        "room": session.room.name if session.room else session.room_id,
                        "time": time_str,
                    }
                )

    return violations


def _check_room_conflicts(
    sessions: List[CourseSession], qts: QuantumTimeSystem
) -> List[Dict]:
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
                violations.append(
                    {
                        "room": session.room.name if session.room else room_id,
                        "course": session.course_id,
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
    sessions: List[CourseSession], course_map: Dict[tuple, Course]
) -> List[Dict]:
    """Check for unqualified instructors."""
    violations = []

    for session in sessions:
        course_key = (session.course_id, session.course_type)
        if course_key not in course_map:
            continue

        course = course_map[course_key]
        if session.instructor_id not in course.qualified_instructor_ids:
            violations.append(
                {
                    "course": session.course_id,
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

    return violations


def _check_room_type_mismatches(sessions: List[CourseSession]) -> List[Dict]:
    """Check for room type mismatches."""
    violations = []

    for session in sessions:
        required_features = (
            set(session.required_room_features)
            if isinstance(session.required_room_features, list)
            else {session.required_room_features}
        )
        room_features = (
            set(session.room.room_features)
            if isinstance(session.room.room_features, list)
            else {session.room.room_features}
        )

        if not required_features.issubset(room_features):
            missing = required_features - room_features
            violations.append(
                {
                    "course": session.course_id,
                    "groups": ", ".join(session.group_ids),
                    "room": session.room.name if session.room else session.room_id,
                    "required_features": ", ".join(required_features),
                    "room_features": ", ".join(room_features),
                    "missing_features": ", ".join(missing),
                }
            )

    return violations


def _check_availability_violations(
    sessions: List[CourseSession], qts: QuantumTimeSystem
) -> List[Dict]:
    """Check for availability violations."""
    violations = []

    for session in sessions:
        for q in session.session_quanta:
            day, time = qts.quanta_to_time(q)
            time_str = f"{day} {time}"

            # Check instructor availability
            if q not in session.instructor.available_quanta:
                violations.append(
                    {
                        "type": "Instructor Unavailable",
                        "entity": (
                            session.instructor.name
                            if session.instructor
                            else session.instructor_id
                        ),
                        "course": session.course_id,
                        "groups": ", ".join(session.group_ids),
                        "room": session.room.name if session.room else session.room_id,
                        "time": time_str,
                    }
                )

            # Check room availability
            if q not in session.room.available_quanta:
                violations.append(
                    {
                        "type": "Room Unavailable",
                        "entity": (
                            session.room.name if session.room else session.room_id
                        ),
                        "course": session.course_id,
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
                        "course": session.course_id,
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
    sessions: List[CourseSession], course_map: Dict[str, Course]
) -> List[Dict]:
    """Check for incomplete or over-scheduled courses."""
    violations = []
    course_group_quanta = defaultdict(int)

    # Count quanta per (course_id, group_id)
    for session in sessions:
        for group_id in session.group_ids:
            key = (session.course_id, group_id)
            course_group_quanta[key] += len(session.session_quanta)

    # Check each course's enrolled groups
    for course_id, course in course_map.items():
        expected_quanta = course.quanta_per_week

        for group_id in course.enrolled_group_ids:
            key = (course_id, group_id)
            actual_quanta = course_group_quanta.get(key, 0)

            if actual_quanta != expected_quanta:
                status = (
                    "Under-scheduled"
                    if actual_quanta < expected_quanta
                    else "Over-scheduled"
                )
                violations.append(
                    {
                        "course": course_id,
                        "group": group_id,
                        "expected": expected_quanta,
                        "actual": actual_quanta,
                        "status": status,
                    }
                )

    return violations


# Formatting functions
def _format_group_violations(violations: List[Dict]) -> List[str]:
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
        for conflict in conflicts:
            lines.append(
                f"    - {conflict['course']} @ {conflict['room']} with {conflict['instructor']}"
            )

    lines.append("")
    return lines


def _format_instructor_violations(violations: List[Dict]) -> List[str]:
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
        for conflict in conflicts:
            lines.append(
                f"    - {conflict['course']} with {conflict['groups']} @ {conflict['room']}"
            )

    lines.append("")
    return lines


def _format_room_violations(violations: List[Dict]) -> List[str]:
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
        for conflict in conflicts:
            lines.append(
                f"    - {conflict['course']} with {conflict['groups']} by {conflict['instructor']}"
            )

    lines.append("")
    return lines


def _format_qualification_violations(violations: List[Dict]) -> List[str]:
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


def _format_room_type_violations(violations: List[Dict]) -> List[str]:
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


def _format_availability_violations(violations: List[Dict]) -> List[str]:
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


def _format_schedule_violations(violations: List[Dict]) -> List[str]:
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
        for v in under:
            lines.append(
                f"  [!]  {v['course']} for group {v['group']}: "
                f"Expected {v['expected']} quanta, got {v['actual']}"
            )

    if over:
        lines.append(f"\nOver-scheduled Courses: {len(over)}")
        for v in over:
            lines.append(
                f"  [!]  {v['course']} for group {v['group']}: "
                f"Expected {v['expected']} quanta, got {v['actual']}"
            )

    lines.append("")
    return lines
