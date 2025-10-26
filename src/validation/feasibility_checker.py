"""
Feasibility Checker Module

Analyzes the scheduling problem before running the GA to determine if it's solvable.
Identifies fundamental bottlenecks that would prevent any algorithm from finding a solution.

This module implements five critical feasibility checks:
1. Instructor Workload vs Availability
2. Instructor Qualification Bottleneck (per-course)
3. Room Capacity Bottleneck
4. Room Feature Bottleneck (per-feature)
5. Group Pigeonhole Problem (per-group)

Usage:
    from src.validation.feasibility_checker import check_feasibility

    is_feasible, report = check_feasibility(
        courses, instructors, rooms, groups, quantum_time_system
    )
"""

from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.entities.course import Course
from src.entities.instructor import Instructor
from src.entities.room import Room
from src.entities.group import Group
from src.encoder.quantum_time_system import QuantumTimeSystem
from config.feasibility_config import (
    ENABLE_FEASIBILITY_CHECKS,
    FAIL_ON_INFEASIBILITY,
    FEASIBILITY_CHECKS,
    SHOW_CONSOLE_OUTPUT,
    TOLERANCE_MARGIN,
)

console = Console()


@dataclass
class FeasibilityResult:
    """Result of a single feasibility check."""

    check_name: str
    passed: bool
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FeasibilityReport:
    """Complete feasibility analysis report."""

    is_feasible: bool
    results: List[FeasibilityResult]
    summary: Dict[str, Any]

    def get_failed_checks(self) -> List[FeasibilityResult]:
        """Get all failed checks."""
        return [r for r in self.results if not r.passed]

    def get_critical_failures(self) -> List[FeasibilityResult]:
        """Get all critical failures."""
        return [r for r in self.results if not r.passed and r.severity == "critical"]


def check_feasibility(
    courses: Dict[str, Course],
    instructors: Dict[str, Instructor],
    rooms: Dict[str, Room],
    groups: Dict[str, Group],
    qts: QuantumTimeSystem,
) -> Tuple[bool, FeasibilityReport]:
    """
    Performs comprehensive feasibility analysis on the scheduling problem.

    Args:
        courses: Dictionary of course_id -> Course
        instructors: Dictionary of instructor_id -> Instructor
        rooms: Dictionary of room_id -> Room
        groups: Dictionary of group_id -> Group
        qts: QuantumTimeSystem for time calculations

    Returns:
        Tuple of (is_feasible, FeasibilityReport)
        is_feasible is True only if all critical checks pass
    """
    if not ENABLE_FEASIBILITY_CHECKS:
        console.print("[yellow]⚠ Feasibility checks are disabled in config[/yellow]")
        return True, FeasibilityReport(
            is_feasible=True,
            results=[],
            summary={"status": "skipped", "reason": "disabled in config"},
        )

    if SHOW_CONSOLE_OUTPUT:
        console.print()
        console.rule("[bold cyan]FEASIBILITY ANALYSIS[/bold cyan]", style="cyan")
        console.print()

    results = []

    # Get total operating quanta for calculations
    total_operating_quanta = len(qts.get_all_operating_quanta())

    # Run enabled checks
    if FEASIBILITY_CHECKS["instructor_workload"]["enabled"]:
        result = _check_instructor_workload(courses, instructors, qts)
        results.append(result)
        if SHOW_CONSOLE_OUTPUT:
            _print_check_result(result)

    if FEASIBILITY_CHECKS["instructor_qualification_bottleneck"]["enabled"]:
        result = _check_instructor_qualification_bottleneck(courses, instructors, qts)
        results.append(result)
        if SHOW_CONSOLE_OUTPUT:
            _print_check_result(result)

    if FEASIBILITY_CHECKS["room_capacity_bottleneck"]["enabled"]:
        result = _check_room_capacity_bottleneck(courses, rooms, groups, qts)
        results.append(result)
        if SHOW_CONSOLE_OUTPUT:
            _print_check_result(result)

    if FEASIBILITY_CHECKS["room_feature_bottleneck"]["enabled"]:
        result = _check_room_feature_bottleneck(courses, rooms, qts)
        results.append(result)
        if SHOW_CONSOLE_OUTPUT:
            _print_check_result(result)

    if FEASIBILITY_CHECKS["group_pigeonhole"]["enabled"]:
        result = _check_group_pigeonhole(courses, groups, total_operating_quanta)
        results.append(result)
        if SHOW_CONSOLE_OUTPUT:
            _print_check_result(result)

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

    if SHOW_CONSOLE_OUTPUT:
        _print_summary(report)

    # Handle infeasibility
    if not is_feasible and FAIL_ON_INFEASIBILITY:
        console.print()
        console.print(
            Panel(
                "[bold red]❌ PROBLEM IS INFEASIBLE[/bold red]\n\n"
                f"Found {len(critical_failures)} critical issue(s) that make this problem unsolvable.\n"
                "Please review the detailed report above and fix the identified issues.\n\n"
                "[dim]Set FAIL_ON_INFEASIBILITY=False in config to continue anyway (not recommended).[/dim]",
                border_style="red",
                box=box.DOUBLE,
            )
        )
        console.print()

    return is_feasible, report


def _check_instructor_workload(
    courses: Dict[str, Course],
    instructors: Dict[str, Instructor],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 1: Instructor Workload vs Availability

    Verifies that the total teaching demand doesn't exceed total instructor availability.
    This is a global check - if it fails, the problem is definitely unsolvable.
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
    adjusted_supply = total_supply * (1 + TOLERANCE_MARGIN)

    passed = total_demand <= adjusted_supply
    utilization_rate = (
        (total_demand / total_supply * 100) if total_supply > 0 else float("inf")
    )

    message = f"Demand: {total_demand} quanta, Supply: {total_supply} quanta"
    if passed:
        message += f" ✓ (Utilization: {utilization_rate:.1f}%)"
    else:
        shortage = total_demand - total_supply
        message += f" ✗ (Shortage: {shortage} quanta, {shortage * qts.QUANTUM_MINUTES // 60} hours)"

    recommendations = []
    if not passed:
        shortage_hours = shortage * qts.QUANTUM_MINUTES // 60
        recommendations.extend(
            [
                f"Add {shortage_hours} more hours of instructor availability",
                f"Hire additional instructors to cover the shortage",
                f"Reduce course offerings by {shortage} quanta",
                f"Increase availability of existing part-time instructors",
            ]
        )
    elif utilization_rate > 90:
        recommendations.append(
            f"⚠ High utilization ({utilization_rate:.1f}%) - consider adding buffer capacity"
        )

    return FeasibilityResult(
        check_name="Instructor Workload vs Availability",
        passed=passed,
        severity=FEASIBILITY_CHECKS["instructor_workload"]["severity"],
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
    courses: Dict[str, Course],
    instructors: Dict[str, Instructor],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 2: Instructor Qualification Bottleneck

    For each course, verifies that there are enough qualified instructors
    with sufficient availability to cover all required sessions.
    """
    all_operating_quanta = qts.get_all_operating_quanta()
    bottlenecks = []
    total_courses = len(courses)
    problematic_courses = 0

    for course_id, course in courses.items():
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
        adjusted_supply = supply * (1 + TOLERANCE_MARGIN)

        if demand > adjusted_supply:
            shortage = demand - supply
            bottlenecks.append(
                {
                    "course_id": course_id,
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
        message = f"All {total_courses} courses have sufficient qualified instructor availability ✓"
    else:
        message = f"{problematic_courses}/{total_courses} courses lack qualified instructor capacity ✗"

    recommendations = []
    if not passed:
        # Show top 5 most problematic courses
        bottlenecks.sort(key=lambda x: x["shortage"], reverse=True)
        recommendations.append("Most critical bottlenecks:")
        for b in bottlenecks[:5]:
            shortage_hours = b["shortage"] * qts.QUANTUM_MINUTES // 60
            recommendations.append(
                f"  • {b['course_name']} ({b['course_id']}): "
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
        severity=FEASIBILITY_CHECKS["instructor_qualification_bottleneck"]["severity"],
        message=message,
        details={
            "total_courses": total_courses,
            "problematic_courses": problematic_courses,
            "bottlenecks": bottlenecks,
        },
        recommendations=recommendations,
    )


def _check_room_capacity_bottleneck(
    courses: Dict[str, Course],
    rooms: Dict[str, Room],
    groups: Dict[str, Group],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 3: Room Capacity Bottleneck

    Verifies that total seat-hours available can accommodate total student-hours required.
    Also checks if the largest class can fit in any room.
    """
    all_operating_quanta = qts.get_all_operating_quanta()

    # Calculate demand: sum of (students × quanta) for all courses
    total_student_hours = 0
    largest_class_size = 0
    largest_class_course = None

    for course in courses.values():
        # Count students enrolled (from groups)
        enrolled_students = 0
        for group_id in course.enrolled_group_ids:
            if group_id in groups:
                enrolled_students += groups[group_id].student_count

        student_hours = enrolled_students * course.quanta_per_week
        total_student_hours += student_hours

        if enrolled_students > largest_class_size:
            largest_class_size = enrolled_students
            largest_class_course = course

    # Calculate supply: sum of (capacity × available_quanta) for all rooms
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
    adjusted_supply = total_seat_hours * (1 + TOLERANCE_MARGIN)

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
        message = f"Seat-hours sufficient: {total_seat_hours:,} available, {total_student_hours:,} needed ✓ ({utilization:.1f}%)"
    else:
        message = f"Seat-hours insufficient ✗"

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
        recommendations.extend(
            [
                f"",
                f"⚠ Largest class has {largest_class_size} students but biggest room only holds {largest_room_capacity}",
                f"   Problem course: {largest_class_course.name} ({largest_class_course.course_id})",
                "Solutions:",
                "• Split the large course into multiple sections",
                "• Add a larger room",
                "• Reduce enrollment for this course",
            ]
        )

    if passed and utilization > 85:
        recommendations.append(
            f"⚠ High room utilization ({utilization:.1f}%) - may cause scheduling conflicts"
        )

    return FeasibilityResult(
        check_name="Room Capacity Bottleneck",
        passed=passed,
        severity=FEASIBILITY_CHECKS["room_capacity_bottleneck"]["severity"],
        message=message,
        details={
            "total_student_hours": total_student_hours,
            "total_seat_hours": total_seat_hours,
            "shortage": max(0, total_student_hours - total_seat_hours),
            "utilization_rate": utilization,
            "largest_class_size": largest_class_size,
            "largest_room_capacity": largest_room_capacity,
            "num_rooms": len(rooms),
        },
        recommendations=recommendations,
    )


def _check_room_feature_bottleneck(
    courses: Dict[str, Course],
    rooms: Dict[str, Room],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 4: Room Feature Bottleneck

    For each required room feature, verifies that rooms with that feature
    have sufficient total availability to cover all courses requiring it.
    """
    all_operating_quanta = qts.get_all_operating_quanta()

    # Group courses by required feature
    feature_demand = defaultdict(int)
    for course in courses.values():
        feature_demand[course.required_room_features] += course.quanta_per_week

    # Calculate supply for each feature
    feature_supply = defaultdict(int)
    for room in rooms.values():
        if room.available_quanta:
            availability = len(room.available_quanta)
        else:
            availability = len(all_operating_quanta)

        feature_supply[room.room_features] += availability

    # Check each feature
    bottlenecks = []
    for feature, demand in feature_demand.items():
        supply = feature_supply.get(feature, 0)
        adjusted_supply = supply * (1 + TOLERANCE_MARGIN)

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
        message = f"All required room features have sufficient availability ✓"
    else:
        message = f"{len(bottlenecks)} room feature(s) have capacity shortages ✗"

    recommendations = []
    if not passed:
        recommendations.append("Feature bottlenecks:")
        for b in bottlenecks:
            shortage_hours = b["shortage"] * qts.QUANTUM_MINUTES // 60
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
        severity=FEASIBILITY_CHECKS["room_feature_bottleneck"]["severity"],
        message=message,
        details={
            "total_features": len(feature_demand),
            "bottleneck_features": len(bottlenecks),
            "bottlenecks": bottlenecks,
        },
        recommendations=recommendations,
    )


def _check_group_pigeonhole(
    courses: Dict[str, Course],
    groups: Dict[str, Group],
    total_operating_quanta: int,
) -> FeasibilityResult:
    """
    Check 5: Group Pigeonhole Problem

    Verifies that no student group has more required course hours
    than there are available time slots in the week.
    This is the most fundamental check - if a group needs 80 hours
    but there are only 72 hours in the week, it's impossible.
    """
    overloaded_groups = []
    max_utilization = 0

    for group_id, group in groups.items():
        # Calculate total quanta needed for this group
        total_demand = 0
        for course_id in group.enrolled_courses:
            if course_id in courses:
                total_demand += courses[course_id].quanta_per_week

        # Check group-specific availability if specified
        if group.available_quanta:
            available = len(group.available_quanta)
        else:
            available = total_operating_quanta

        # Apply tolerance
        adjusted_available = available * (1 + TOLERANCE_MARGIN)

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
        message = f"All {len(groups)} groups have feasible course loads ✓"
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
            f"⚠ Some groups have high utilization (>{max_utilization:.0f}%) - "
            f"this leaves little room for scheduling flexibility"
        )

    return FeasibilityResult(
        check_name="Group Pigeonhole Problem",
        passed=passed,
        severity=FEASIBILITY_CHECKS["group_pigeonhole"]["severity"],
        message=message,
        details={
            "total_groups": len(groups),
            "overloaded_groups": len(overloaded_groups),
            "max_utilization": max_utilization,
            "details": overloaded_groups,
        },
        recommendations=recommendations,
    )


def _print_check_result(result: FeasibilityResult):
    """Print a single check result with rich formatting."""
    if result.passed:
        icon = "✓"
        color = "green"
    else:
        icon = "✗"
        color = "red" if result.severity == "critical" else "yellow"

    console.print(f"[{color}]{icon} {result.check_name}[/{color}]")
    console.print(f"  {result.message}")

    if result.recommendations:
        for rec in result.recommendations[:3]:  # Show first 3 recommendations
            console.print(f"  [dim]{rec}[/dim]")

    console.print()


def _print_summary(report: FeasibilityReport):
    """Print overall feasibility summary."""
    console.rule("[bold]Summary[/bold]")
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
        console.print(
            Panel(
                "[bold green]✓ PROBLEM IS FEASIBLE[/bold green]\n\n"
                "All critical checks passed. The GA should be able to find a solution.\n"
                "[dim]Note: This doesn't guarantee a perfect solution exists, but fundamental constraints are satisfied.[/dim]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]✗ PROBLEM IS INFEASIBLE[/bold red]\n\n"
                f"Found {report.summary['critical_failures']} critical issue(s) that make this problem unsolvable.\n"
                "The GA will not be able to find a valid solution until these are fixed.",
                border_style="red",
            )
        )

    console.print()
    console.rule(style="cyan")
    console.print()


def generate_feasibility_report_file(report: FeasibilityReport, output_path: str):
    """
    Generate a detailed text report file with feasibility analysis results.

    Args:
        report: FeasibilityReport to save
        output_path: Path to save the report
    """
    import json
    from datetime import datetime

    with open(output_path, "w") as f:
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

            if result.details:
                f.write("\nDetails:\n")
                f.write(json.dumps(result.details, indent=2))
                f.write("\n")

            if result.recommendations:
                f.write("\nRecommendations:\n")
                for rec in result.recommendations:
                    f.write(f"  {rec}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
