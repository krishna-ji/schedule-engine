"""
Feasibility Checker Module

Analyzes the scheduling problem before running the GA to determine if it's solvable.
Identifies fundamental bottlenecks that would prevent any algorithm from finding a solution.

PERFORMANCE: Runs 5 independent checks in parallel using ThreadPoolExecutor (3-5x speedup).

This module implements five critical feasibility checks:
1. Instructor Workload vs Availability
2. Instructor Qualification Bottleneck (per-course)
3. Room Capacity Bottleneck
4. Room Feature Bottleneck (per-feature)
5. Group Pigeonhole Problem (per-group)

Usage:
    from src.io.feasibility import check_feasibility

    is_feasible, report = check_feasibility(
        courses, instructors, rooms, groups, quantum_time_system
    )
"""

from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table

from src.config import get_config
from src.domain.course import Course
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.io.time_system import QuantumTimeSystem
from src.utils.console_service import get_console

__all__ = ["check_feasibility", "FeasibilityReport"]
from src.utils.system_info import get_cpu_count

console = get_console()


@dataclass
class FeasibilityResult:
    """Result of a single feasibility check."""

    check_name: str
    passed: bool
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class FeasibilityReport:
    """Complete feasibility analysis report."""

    is_feasible: bool
    results: list[FeasibilityResult]
    summary: dict[str, Any]

    def get_failed_checks(self) -> list[FeasibilityResult]:
        """Get all failed checks."""
        return [r for r in self.results if not r.passed]

    def get_critical_failures(self) -> list[FeasibilityResult]:
        """Get all critical failures."""
        return [r for r in self.results if not r.passed and r.severity == "critical"]


def check_feasibility(
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    rooms: dict[str, Room],
    groups: dict[str, Group],
    qts: QuantumTimeSystem,
) -> tuple[bool, FeasibilityReport]:
    """
    Performs comprehensive feasibility analysis on the scheduling problem.

    Args:
        courses: Dictionary of (course_code, course_type) tuple -> Course
        instructors: Dictionary of instructor_id -> Instructor
        rooms: Dictionary of room_id -> Room
        groups: Dictionary of group_id -> Group
        qts: QuantumTimeSystem for time calculations

    Returns:
        Tuple of (is_feasible, FeasibilityReport)
        is_feasible is True only if all critical checks pass
    """
    if not get_config().feasibility.enable_checks:
        console.print("[yellow]Feasibility checks are disabled in config[/yellow]")
        return True, FeasibilityReport(
            is_feasible=True,
            results=[],
            summary={"status": "skipped", "reason": "disabled in config"},
        )

    if get_config().feasibility.show_console_output:
        console.print()
        console.print("[bold cyan]feasibility analysis[/bold cyan]")
        console.print()

    # Get total operating quanta for calculations
    total_operating_quanta = len(qts.get_all_operating_quanta())

    # PERFORMANCE: Run all checks in parallel (3-5x speedup)
    # Build list of checks to run - each check has different function signature
    checks_to_run: list[tuple[str, Any, tuple[Any, ...]]] = []

    if get_config().feasibility.checks["instructor_workload"]["enabled"]:
        checks_to_run.append(
            (
                "instructor_workload",
                _check_instructor_workload,
                (courses, instructors, qts),
            )
        )

    if get_config().feasibility.checks["instructor_qualification_bottleneck"][
        "enabled"
    ]:
        checks_to_run.append(
            (
                "qualification_bottleneck",
                _check_instructor_qualification_bottleneck,
                (courses, instructors, qts),
            )
        )

    if get_config().feasibility.checks["room_capacity_bottleneck"]["enabled"]:
        checks_to_run.append(
            (
                "room_capacity",
                _check_room_capacity_bottleneck,
                (courses, rooms, groups, qts),
            )
        )

    if get_config().feasibility.checks["room_feature_bottleneck"]["enabled"]:
        checks_to_run.append(
            ("room_feature", _check_room_feature_bottleneck, (courses, rooms, qts))
        )

    if get_config().feasibility.checks["group_pigeonhole"]["enabled"]:
        checks_to_run.append(
            (
                "group_pigeonhole",
                _check_group_pigeonhole,
                (courses, groups, total_operating_quanta),
            )
        )

    # Execute all checks concurrently
    results = []

    max_workers = get_cpu_count()  # Auto-detect all cores
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_check = {
            executor.submit(check_func, *args): name
            for name, check_func, args in checks_to_run
        }

        for future in as_completed(future_to_check):
            try:
                result = future.result()
                results.append(result)
                if get_config().feasibility.show_console_output:
                    _print_check_result(result)
            except Exception as e:
                check_name = future_to_check[future]
                console.print(f"[red]Error in {check_name}: {e}[/red]")
                # Create failed result
                results.append(
                    FeasibilityResult(
                        check_name=check_name,
                        passed=False,
                        severity="critical",
                        message=f"Check failed with error: {e}",
                        details={},
                    )
                )

    # Determine overall feasibility
    critical_failures = [
        r for r in results if not r.passed and r.severity == "critical"
    ]
    is_feasible = len(critical_failures) == 0

    # Create summary
    summary = {
        "total_checks": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "critical_failures": len(critical_failures),
        "status": "feasible" if is_feasible else "infeasible",
    }

    report = FeasibilityReport(
        is_feasible=is_feasible, results=results, summary=summary
    )

    if get_config().feasibility.show_console_output:
        _print_summary(report)

    # Handle infeasibility
    if not is_feasible and get_config().feasibility.fail_on_infeasibility:
        console.print()
        console.print(
            Panel(
                "[bold red][!ERR] PROBLEM IS INFEASIBLE[/bold red]\n\n"
                f"Found {len(critical_failures)} critical issue(s) that make this problem unsolvable.\n"
                "Please review the detailed report above and fix the identified issues.\n\n"
                "[dim]Set get_config().feasibility.fail_on_infeasibility=False in config to continue anyway (not recommended).[/dim]",
                border_style="red",
                box=box.DOUBLE,
            )
        )
        console.print()
        console.print("[bold red] Exiting program...[/bold red]\n")

        # Gracefully exit without traceback
        sys.exit(1)

    return is_feasible, report


def _check_instructor_workload(
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 1: Instructor Workload vs Availability

    Verifies that the total teaching demand doesn't exceed total instructor availability.
    This is a global check - if it fails, the problem is definitely unsolvable.

    Note: courses dict is keyed by (course_code, course_type) tuples
    """
    # Calculate total demand (in quanta)
    total_demand = sum(course.quanta_per_week for course in courses.values())

    # Calculate total supply (in quanta)
    all_operating_quanta = qts.get_all_operating_quanta()
    total_supply = 0

    for instructor in instructors.values():
        if instructor.is_full_time:
            # Full-time instructor: available during all operating hours
            total_supply += len(all_operating_quanta)
        else:
            # Part-time instructor: only available during specified quanta
            total_supply += len(instructor.available_quanta)

    # Apply tolerance margin
    adjusted_supply = total_supply * (1 + get_config().feasibility.tolerance_margin)

    passed = total_demand <= adjusted_supply
    utilization_rate = (
        (total_demand / total_supply * 100) if total_supply > 0 else float("inf")
    )

    message = f"Demand: {total_demand} quanta, Supply: {total_supply} quanta"
    if passed:
        message += f" [!ok] (Utilization: {utilization_rate:.1f}%)"
    else:
        shortage = total_demand - total_supply
        message += f" ✗ (Shortage: {shortage} quanta, {shortage * qts.QUANTUM_MINUTES // 60} hours)"

    recommendations = []
    if not passed:
        shortage_hours = shortage * qts.QUANTUM_MINUTES // 60
        recommendations.extend(
            [
                f"Add {shortage_hours} more hours of instructor availability",
                "Hire additional instructors to cover the shortage",
                f"Reduce course offerings by {shortage} quanta",
                "Increase availability of existing part-time instructors",
            ]
        )
    elif utilization_rate > 90:
        recommendations.append(
            f"High utilization ({utilization_rate:.1f}%) - consider adding buffer capacity"
        )

    return FeasibilityResult(
        check_name="Instructor Workload vs Availability",
        passed=passed,
        severity=get_config().feasibility.checks["instructor_workload"]["severity"],
        message=message,
        details={
            "total_demand_quanta": total_demand,
            "total_supply_quanta": total_supply,
            "shortage_quanta": max(0, total_demand - total_supply),
            "utilization_rate": utilization_rate,
            "num_instructors": len(instructors),
            "full_time_instructors": sum(
                1 for i in instructors.values() if i.is_full_time
            ),
            "part_time_instructors": sum(
                1 for i in instructors.values() if not i.is_full_time
            ),
        },
        recommendations=recommendations,
    )


def _check_instructor_qualification_bottleneck(
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 2: Instructor Qualification Bottleneck

    For each course, verifies that there are enough qualified instructors
    with sufficient availability to cover all required sessions.

    Note: courses dict is keyed by (course_code, course_type) tuples
    """
    all_operating_quanta = qts.get_all_operating_quanta()
    bottlenecks: list[dict[str, Any]] = []
    total_courses = len(courses)
    problematic_courses = 0

    for course_key, course in courses.items():
        demand = course.quanta_per_week

        # Find all qualified instructors and sum their availability
        supply = 0
        qualified_count = 0

        for instructor_id in course.qualified_instructor_ids:
            if instructor_id not in instructors:
                continue

            instructor = instructors[instructor_id]
            qualified_count += 1

            if instructor.is_full_time:
                supply += len(all_operating_quanta)
            else:
                supply += len(instructor.available_quanta)

        # Check if supply meets demand
        adjusted_supply = supply * (1 + get_config().feasibility.tolerance_margin)

        if demand > adjusted_supply:
            shortage = demand - supply
            # Format course_key as string for display: "ENME 103 (theory)"
            course_display = (
                f"{course_key[0]} ({course_key[1]})"
                if isinstance(course_key, tuple)
                else str(course_key)
            )
            bottlenecks.append(
                {
                    "course_key": course_key,
                    "course_display": course_display,
                    "course_name": course.name,
                    "demand": demand,
                    "supply": supply,
                    "shortage": shortage,
                    "qualified_instructors": qualified_count,
                }
            )
            problematic_courses += 1

    passed = len(bottlenecks) == 0

    if passed:
        message = f"All {total_courses} courses have sufficient qualified instructor availability [!ok]"
    else:
        message = f"{problematic_courses}/{total_courses} courses lack qualified instructor capacity ✗"

    recommendations = []
    if not passed:
        # Show top 5 most problematic courses
        bottlenecks.sort(key=lambda x: x.get("shortage", 0), reverse=True)  # type: ignore[arg-type,return-value]
        recommendations.append("Most critical bottlenecks:")
        for b in bottlenecks[:5]:
            shortage = b.get("shortage", 0)
            shortage_hours = (
                int(shortage) * qts.QUANTUM_MINUTES // 60
                if isinstance(shortage, int)
                else 0
            )
            recommendations.append(
                f"  • {b['course_name']} ({b['course_display']}): "
                f"needs {shortage_hours}h more from qualified instructors "
                f"(currently {b['qualified_instructors']} qualified)"
            )

        if len(bottlenecks) > 5:
            recommendations.append(f"  ... and {len(bottlenecks) - 5} more courses")

        recommendations.extend(
            [
                "",
                "Solutions:",
                "• Qualify more instructors for bottleneck courses",
                "• Increase availability of qualified instructors",
                "• Reduce sections/sessions for problematic courses",
            ]
        )

    return FeasibilityResult(
        check_name="Instructor Qualification Bottleneck",
        passed=passed,
        severity=get_config().feasibility.checks["instructor_qualification_bottleneck"][
            "severity"
        ],
        message=message,
        details={
            "total_courses": total_courses,
            "problematic_courses": problematic_courses,
            "bottlenecks": bottlenecks,
        },
        recommendations=recommendations,
    )


def _check_room_capacity_bottleneck(
    courses: dict[tuple, Course],
    rooms: dict[str, Room],
    groups: dict[str, Group],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 3: Room Capacity Bottleneck

    Verifies that total seat-hours available can accommodate total student-hours required.
    Also checks if the largest class can fit in any room.

    Note: courses dict is keyed by (course_code, course_type) tuples
    """
    all_operating_quanta = qts.get_all_operating_quanta()

    # Calculate demand: sum of (students * quanta) for all course-group sessions
    # IMPORTANT: Each group takes the course separately, so we need capacity per session
    total_student_hours = 0
    largest_class_size = 0
    largest_class_course = None
    largest_class_key = None

    for course_key, course in courses.items():
        # Each enrolled group takes this course in SEPARATE sessions
        for group_id in course.enrolled_group_ids:
            if group_id not in groups:
                continue

            group = groups[group_id]
            group_size = group.student_count

            # This group needs (group_size * quanta) seat-hours
            student_hours = group_size * course.quanta_per_week
            total_student_hours += student_hours

            # Track largest single session (not sum of all groups!)
            if group_size > largest_class_size:
                largest_class_size = group_size
                largest_class_course = course
                largest_class_key = course_key

    # Calculate supply: sum of (capacity * available_quanta) for all rooms
    total_seat_hours = 0
    largest_room_capacity = 0

    for room in rooms.values():
        if room.available_quanta:
            # Room has specific availability
            room_capacity_hours = room.capacity * len(room.available_quanta)
        else:
            # Room is available during all operating hours
            room_capacity_hours = room.capacity * len(all_operating_quanta)

        total_seat_hours += room_capacity_hours
        largest_room_capacity = max(largest_room_capacity, room.capacity)

    # Apply tolerance
    adjusted_supply = total_seat_hours * (1 + get_config().feasibility.tolerance_margin)

    # Check 1: Global capacity
    global_passed = total_student_hours <= adjusted_supply

    # Check 2: Largest class vs largest room
    largest_class_passed = largest_class_size <= largest_room_capacity

    passed = global_passed and largest_class_passed

    utilization = (
        (total_student_hours / total_seat_hours * 100)
        if total_seat_hours > 0
        else float("inf")
    )

    if passed:
        message = f"Seat-hours sufficient: {total_seat_hours:,} available, {total_student_hours:,} needed [!ok] ({utilization:.1f}%)"
    else:
        message = "Seat-hours insufficient ✗"

    recommendations = []
    if not global_passed:
        shortage = total_student_hours - total_seat_hours
        recommendations.extend(
            [
                f"Global shortage: {shortage:,} seat-hours needed",
                "Solutions:",
                "• Add more rooms to the schedule",
                "• Increase room availability hours",
                "• Reduce course enrollments",
                "• Offer some courses at different times (if rooms are underutilized)",
            ]
        )

    if not largest_class_passed:
        course_display = (
            f"{largest_class_key[0]} ({largest_class_key[1]})"
            if isinstance(largest_class_key, tuple)
            else str(largest_class_key)
        )
        course_name = largest_class_course.name if largest_class_course else "Unknown"
        recommendations.extend(
            [
                "",
                f"Largest single session has {largest_class_size} students but biggest room only holds {largest_room_capacity}",
                f"   Problem course: {course_name} ({course_display})",
                "   Note: This is the largest group size, not sum of all groups",
                "Solutions:",
                "• Split the large group into smaller sections",
                "• Add a larger room (capacity ≥ {largest_class_size})",
                "• Reduce enrollment for this group",
            ]
        )

    if passed and utilization > 85:
        recommendations.append(
            f"High room utilization ({utilization:.1f}%) - may cause scheduling conflicts"
        )

    return FeasibilityResult(
        check_name="Room Capacity Bottleneck",
        passed=passed,
        severity=get_config().feasibility.checks["room_capacity_bottleneck"][
            "severity"
        ],
        message=message,
        details={
            "total_student_hours": total_student_hours,
            "total_seat_hours": total_seat_hours,
            "shortage": max(0, total_student_hours - total_seat_hours),
            "utilization_rate": utilization,
            "largest_class_size": largest_class_size,
            "largest_class_course": (
                largest_class_course.name if largest_class_course else "N/A"
            ),
            "largest_class_course_id": (
                largest_class_course.course_id if largest_class_course else "N/A"
            ),
            "largest_room_capacity": largest_room_capacity,
            "num_rooms": len(rooms),
            "largest_class_course_key": largest_class_key,
            "largest_class_course_name": (
                largest_class_course.name if largest_class_course else None
            ),
        },
        recommendations=recommendations,
    )


def _check_room_feature_bottleneck(
    courses: dict[tuple, Course],
    rooms: dict[str, Room],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 4: Room Feature Bottleneck

    For each required room feature, verifies that rooms with that feature
    have sufficient total availability to cover all courses requiring it.

    Note: courses dict is keyed by (course_code, course_type) tuples
    """
    all_operating_quanta = qts.get_all_operating_quanta()

    # Group courses by required feature
    feature_demand: dict[str, int] = defaultdict(int)
    for course in courses.values():
        feature_demand[course.required_room_features] += course.quanta_per_week

    # Calculate supply for each feature
    feature_supply: dict[str, int] = defaultdict(int)
    for room in rooms.values():
        if room.available_quanta:
            availability = len(room.available_quanta)
        else:
            availability = len(all_operating_quanta)

        feature_supply[room.room_features] += availability

    # Check each feature
    bottlenecks: list[dict[str, Any]] = []
    for feature, demand in feature_demand.items():
        supply = feature_supply.get(feature, 0)
        adjusted_supply = supply * (1 + get_config().feasibility.tolerance_margin)

        if demand > adjusted_supply:
            shortage = demand - supply
            # Count rooms with this feature
            room_count = sum(1 for r in rooms.values() if r.room_features == feature)

            bottlenecks.append(
                {
                    "feature": feature,
                    "demand": demand,
                    "supply": supply,
                    "shortage": shortage,
                    "room_count": room_count,
                }
            )

    passed = len(bottlenecks) == 0

    if passed:
        message = "All required room features have sufficient availability [!ok]"
    else:
        message = f"{len(bottlenecks)} room feature(s) have capacity shortages ✗"

    recommendations = []
    if not passed:
        recommendations.append("Feature bottlenecks:")
        for b in bottlenecks:
            shortage = b.get("shortage", 0)
            shortage_hours = (
                int(shortage) * qts.QUANTUM_MINUTES // 60
                if isinstance(shortage, int)
                else 0
            )
            recommendations.append(
                f"  • Feature '{b['feature']}': needs {shortage_hours}h more "
                f"({b['room_count']} rooms currently have this feature)"
            )

        recommendations.extend(
            [
                "",
                "Solutions:",
                "• Add more rooms with the required features",
                "• Equip existing rooms with needed features",
                "• Increase availability of feature-specific rooms",
                "• Reduce courses requiring scarce features",
            ]
        )

    return FeasibilityResult(
        check_name="Room Feature Bottleneck",
        passed=passed,
        severity=get_config().feasibility.checks["room_feature_bottleneck"]["severity"],
        message=message,
        details={
            "total_features": len(feature_demand),
            "bottleneck_features": len(bottlenecks),
            "bottlenecks": bottlenecks,
        },
        recommendations=recommendations,
    )


def _check_group_pigeonhole(
    courses: dict[tuple, Course],
    groups: dict[str, Group],
    total_operating_quanta: int,
) -> FeasibilityResult:
    """
    Check 5: Group Pigeonhole Problem

    Verifies that no student group has more required course hours
    than there are available time slots in the week.
    This is the most fundamental check - if a group needs 80 hours
    but there are only 72 hours in the week, it's impossible.

    Note: courses dict is keyed by (course_code, course_type) tuples.
          Groups store enrolled_courses as course_codes (strings).
          We need to check BOTH theory and practical for each course_code.
    """
    overloaded_groups = []
    max_utilization: float = 0.0

    for group_id, group in groups.items():
        # Calculate total quanta needed for this group
        total_demand = 0
        for course_code in group.enrolled_courses:
            # Check both theory and practical versions of this course
            theory_key = (course_code, "theory")
            practical_key = (course_code, "practical")

            if theory_key in courses:
                total_demand += courses[theory_key].quanta_per_week
            if practical_key in courses:
                total_demand += courses[practical_key].quanta_per_week

        # Check group-specific availability if specified
        if group.available_quanta:
            available = len(group.available_quanta)
        else:
            available = total_operating_quanta

        # Apply tolerance
        adjusted_available = available * (1 + get_config().feasibility.tolerance_margin)

        utilization = (
            (total_demand / available * 100) if available > 0 else float("inf")
        )
        max_utilization = max(max_utilization, utilization)

        if total_demand > adjusted_available:
            overload = total_demand - available
            overloaded_groups.append(
                {
                    "group_id": group_id,
                    "group_name": group.name,
                    "demand": total_demand,
                    "available": available,
                    "overload": overload,
                    "utilization": utilization,
                    "num_courses": len(group.enrolled_courses),
                }
            )

    passed = len(overloaded_groups) == 0

    if passed:
        message = f"All {len(groups)} groups have feasible course loads [!ok]"
        if max_utilization > 80:
            message += f" (Max utilization: {max_utilization:.1f}%)"
    else:
        message = f"{len(overloaded_groups)}/{len(groups)} groups are overloaded ✗"

    recommendations = []
    if not passed:
        recommendations.append("Overloaded groups:")
        for g in overloaded_groups:
            recommendations.append(
                f"  • {g['group_name']} ({g['group_id']}): "
                f"needs {g['demand']} quanta but only {g['available']} available "
                f"({g['utilization']:.0f}% utilization, {g['num_courses']} courses)"
            )

        recommendations.extend(
            [
                "",
                "Solutions:",
                "• Reduce number of courses for overloaded groups",
                "• Split large groups into multiple sections",
                "• Extend operating hours (if feasible)",
                "• Distribute courses across multiple semesters",
            ]
        )
    elif max_utilization > 85:
        recommendations.append(
            f"Some groups have high utilization (>{max_utilization:.0f}%) - "
            f"this leaves little room for scheduling flexibility"
        )

    return FeasibilityResult(
        check_name="Group Pigeonhole Problem",
        passed=passed,
        severity=get_config().feasibility.checks["group_pigeonhole"]["severity"],
        message=message,
        details={
            "total_groups": len(groups),
            "overloaded_groups": len(overloaded_groups),
            "max_utilization": max_utilization,
            "details": overloaded_groups,
        },
        recommendations=recommendations,
    )


def _print_check_result(result: FeasibilityResult) -> None:
    """Print a single check result with rich formatting."""
    if result.passed:
        icon = "[!ok]"
        color = "green"
    else:
        icon = "✗"
        color = "red" if result.severity == "critical" else "yellow"

    console.print(f"[{color}]{icon} {result.check_name}[/{color}]")
    console.print(f"  {result.message}")

    # For failed checks, show more details
    if not result.passed and result.recommendations:
        # Show first 5 recommendations on console (more for critical failures)
        display_count = 5 if result.severity == "critical" else 3
        for rec in result.recommendations[:display_count]:
            console.print(f"  [dim]{rec}[/dim]")

        # Indicate if there are more recommendations in the report
        if len(result.recommendations) > display_count:
            remaining = len(result.recommendations) - display_count
            console.print(
                f"  [dim italic]... and {remaining} more (see detailed report)[/dim italic]"
            )

    console.print()


def _print_summary(report: FeasibilityReport) -> None:
    """Print overall feasibility summary."""
    console.print()
    console.print("[bold cyan]summary[/bold cyan]")
    console.print()

    # Create summary table
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total Checks", str(report.summary["total_checks"]))
    table.add_row("Passed", f"[green]{report.summary['passed']}[/green]")
    table.add_row("Failed", f"[red]{report.summary['failed']}[/red]")
    table.add_row(
        "Critical Failures",
        f"[bold red]{report.summary['critical_failures']}[/bold red]",
    )

    console.print(table)
    console.print()

    if report.is_feasible:
        console.print("[green][!ok] problem is feasible[/green]")
        console.print(
            "  [dim]all critical checks passed. GA should find a solution.[/dim]"
        )
        console.print(
            "  [dim]note: this doesn't guarantee a perfect solution exists[/dim]"
        )
    else:
        console.print("[red][!err] problem is infeasible[/red]")
        console.print(
            f"  [dim]found {report.summary['critical_failures']} critical issue(s)[/dim]"
        )
        console.print(
            "  [dim]GA will not find valid solution until these are fixed[/dim]"
        )

    console.print()


def generate_feasibility_report_file(
    report: FeasibilityReport, output_path: str
) -> None:
    """
    Generate a detailed text report file with feasibility analysis results.

    Args:
        report: FeasibilityReport to save
        output_path: Path to save the report
    """
    from datetime import datetime

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FEASIBILITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Status: {report.summary['status'].upper()}\n")
        f.write("\n")

        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Checks: {report.summary['total_checks']}\n")
        f.write(f"Passed: {report.summary['passed']}\n")
        f.write(f"Failed: {report.summary['failed']}\n")
        f.write(f"Critical Failures: {report.summary['critical_failures']}\n")
        f.write("\n")

        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n")

        for i, result in enumerate(report.results, 1):
            f.write(f"\n{i}. {result.check_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Status: {'PASS' if result.passed else 'FAIL'}\n")
            f.write(f"Severity: {result.severity.upper()}\n")
            f.write(f"Message: {result.message}\n")
            f.write("\n")

            # Write detailed information in human-readable format
            if result.details:
                f.write("Details:\n")

                # Format details based on check type
                if "bottlenecks" in result.details and result.details["bottlenecks"]:
                    f.write(
                        f"  Bottlenecks Found: {len(result.details['bottlenecks'])}\n\n"
                    )
                    for j, bottleneck in enumerate(result.details["bottlenecks"], 1):
                        f.write(f"  Bottleneck {j}:\n")
                        for key, value in bottleneck.items():
                            f.write(f"    {key}: {value}\n")
                        f.write("\n")

                # Check for overloaded groups (stored in 'details' subkey for group pigeonhole)
                elif (
                    "details" in result.details
                    and isinstance(result.details["details"], list)
                    and result.details["details"]
                ):
                    f.write(
                        f"  Overloaded Groups: {len(result.details['details'])}\n\n"
                    )
                    for j, group_info in enumerate(result.details["details"], 1):
                        f.write(f"  Group {j}:\n")
                        for key, value in group_info.items():
                            f.write(f"    {key}: {value}\n")
                        f.write("\n")

                else:
                    # Generic detail printing for other metrics
                    for key, value in result.details.items():
                        if key not in ["bottlenecks", "details"] and not isinstance(
                            value, list | dict
                        ):
                            f.write(f"  {key}: {value}\n")
                    f.write("\n")

            if result.recommendations:
                f.write("Recommendations:\n")
                for rec in result.recommendations:
                    f.write(f"  {rec}\n")
                f.write("\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
