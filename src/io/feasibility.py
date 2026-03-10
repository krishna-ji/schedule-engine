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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich import box
from rich.table import Table

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.group import Group
    from src.domain.instructor import Instructor
    from src.domain.room import Room
    from src.io.time_system import QuantumTimeSystem

# Feasibility defaults (previously on FeasibilityConfig)
_TOLERANCE_MARGIN = 0.02
from src.utils.console_service import get_console

__all__ = ["FeasibilityReport", "InfeasibleProblemError", "check_feasibility"]
from src.utils.system_info import get_cpu_count

console = get_console()


class InfeasibleProblemError(RuntimeError):
    """Raised when pre-scheduling feasibility checks detect an unsolvable problem.

    Attributes:
        report: The full FeasibilityReport with all check results and details.
    """

    def __init__(self, report: FeasibilityReport) -> None:
        critical = report.get_critical_failures()
        names = [r.check_name for r in critical]
        msg = (
            f"Problem is INFEASIBLE: {len(critical)} critical check(s) failed "
            f"({', '.join(names)}). Fix the data before scheduling."
        )
        super().__init__(msg)
        self.report = report


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
    if False:  # All checks always enabled
        console.print("[yellow]Feasibility checks are disabled in config[/yellow]")
        return True, FeasibilityReport(
            is_feasible=True,
            results=[],
            summary={"status": "skipped", "reason": "disabled in config"},
        )

    if True:  # Always show console output
        console.print()
        console.print("[bold cyan]feasibility analysis[/bold cyan]")
        console.print()

    # Get total operating quanta for calculations
    total_operating_quanta = len(qts.get_all_operating_quanta())

    # PERFORMANCE: Run all checks in parallel (3-5x speedup)
    # Build list of checks to run - each check has different function signature
    checks_to_run: list[tuple[str, Any, tuple[Any, ...]]] = []

    if True:  # All checks enabled
        checks_to_run.append(
            (
                "instructor_workload",
                _check_instructor_workload,
                (courses, instructors, qts),
            )
        )

    if True:  # All checks enabled
        checks_to_run.append(
            (
                "qualification_bottleneck",
                _check_instructor_qualification_bottleneck,
                (courses, instructors, qts),
            )
        )

    if True:  # All checks enabled
        checks_to_run.append(
            ("room_feature", _check_room_feature_bottleneck, (courses, rooms, qts))
        )

    if True:  # All checks enabled
        checks_to_run.append(
            (
                "specific_lab_features",
                _check_specific_lab_features,
                (courses, rooms, qts),
            )
        )

    if True:  # All checks enabled
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
                if True:  # Always show console output
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

    if True:  # Always show console output
        _print_summary(report)

    # Note: We do NOT call sys.exit here - caller handles that after saving reports
    # This allows the caller to save the feasibility report before exiting
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
    # Calculate total demand (in quanta) - only for courses with enrolled groups
    total_demand = sum(
        course.quanta_per_week
        for course in courses.values()
        if course.enrolled_group_ids
    )

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
    adjusted_supply = total_supply * (1 + _TOLERANCE_MARGIN)

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
        severity="critical",
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
        # Skip courses with no enrolled groups (not being scheduled)
        if not course.enrolled_group_ids:
            continue

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
        adjusted_supply = supply * (1 + _TOLERANCE_MARGIN)

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
        bottlenecks.sort(key=lambda x: x.get("shortage", 0), reverse=True)
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
        severity="critical",
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
    adjusted_supply = total_seat_hours * (1 + _TOLERANCE_MARGIN)

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
        severity="critical",
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
        # Skip courses with no enrolled groups (not being scheduled)
        if not course.enrolled_group_ids:
            continue
        feature_demand[course.required_room_features] += course.quanta_per_week

    # Calculate supply for each feature
    # "both" type rooms contribute to both "lecture" and "practical" supply
    feature_supply: dict[str, int] = defaultdict(int)
    for room in rooms.values():
        if room.available_quanta:
            availability = len(room.available_quanta)
        else:
            availability = len(all_operating_quanta)

        if room.room_features == "both":
            feature_supply["lecture"] += availability
            feature_supply["practical"] += availability
        feature_supply[room.room_features] += availability

    # Check each feature
    bottlenecks: list[dict[str, Any]] = []
    for feature, demand in feature_demand.items():
        supply = feature_supply.get(feature, 0)
        adjusted_supply = supply * (1 + _TOLERANCE_MARGIN)

        if demand > adjusted_supply:
            shortage = demand - supply
            # Count rooms with this feature (including "both" rooms for lecture/practical)
            room_count = sum(
                1
                for r in rooms.values()
                if r.room_features == feature
                or (r.room_features == "both" and feature in ("lecture", "practical"))
            )

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
        severity="critical",
        message=message,
        details={
            "total_features": len(feature_demand),
            "bottleneck_features": len(bottlenecks),
            "bottlenecks": bottlenecks,
        },
        recommendations=recommendations,
    )


def _check_specific_lab_features(
    courses: dict[tuple, Course],
    rooms: dict[str, Room],
    qts: QuantumTimeSystem,
) -> FeasibilityResult:
    """
    Check 6: Specific Lab Feature Availability

    Three-layer analysis for practical room features:

    Layer 1 — Existence:  Does the feature exist in ANY room?
    Layer 2 — Type match: Does it exist on a practical-capable room
              (type = "practical" or "both")?
    Layer 3 — Capacity:   Is total quanta supply from rooms carrying
              that feature ≥ the quanta demand from courses needing it?

    Theory courses are NOT checked here — they only need the broad
    "lecture" type match (handled by ``_check_room_feature_bottleneck``).
    """
    all_operating_quanta = qts.get_all_operating_quanta()

    # ── 1. Collect demand per specific feature ──────────────────────
    # feature -> list of course keys that need it
    feature_demand: dict[str, list[tuple]] = defaultdict(list)
    feature_quanta: dict[str, int] = defaultdict(int)

    for course_key, course in courses.items():
        if not course.enrolled_group_ids:
            continue
        if not course.specific_lab_features:
            continue
        for feat in course.specific_lab_features:
            feature_demand[feat].append(course_key)
            feature_quanta[feat] += course.quanta_per_week

    if not feature_demand:
        return FeasibilityResult(
            check_name="Specific Lab Feature Availability",
            passed=True,
            severity="critical",
            message="No specific lab features required by any course [!ok]",
            details={"required_features": 0, "available_features": 0},
        )

    # ── 2. Collect supply per specific feature ──────────────────────
    available_features: set[str] = set()
    practical_features: set[str] = set()
    feature_rooms: dict[str, list[str]] = defaultdict(list)
    feature_practical_rooms: dict[str, list[str]] = defaultdict(list)
    # Per-feature quanta supply (only from practical-capable rooms)
    feature_supply_quanta: dict[str, int] = defaultdict(int)

    for room in rooms.values():
        is_practical_capable = room.room_features in ("practical", "both")
        room_avail = (
            len(room.available_quanta)
            if room.available_quanta
            else len(all_operating_quanta)
        )

        for feat in room.specific_features:
            available_features.add(feat)
            feature_rooms[feat].append(room.room_id)
            if is_practical_capable:
                practical_features.add(feat)
                feature_practical_rooms[feat].append(room.room_id)
                feature_supply_quanta[feat] += room_avail

    # ── 3. Layer 1 — Existence check ───────────────────────────────
    required_features = set(feature_demand.keys())
    missing_features = required_features - available_features
    present_features = required_features & available_features

    # ── 4. Layer 2 — Type match check ──────────────────────────────
    type_mismatch_features: set[str] = set()
    for feat in present_features:
        if feat not in practical_features:
            type_mismatch_features.add(feat)

    # ── 5. Layer 3 — Capacity check (per-feature supply vs demand) ─
    capacity_bottlenecks: list[dict[str, Any]] = []
    for feat in sorted(present_features - type_mismatch_features):
        demand = feature_quanta[feat]
        supply = feature_supply_quanta.get(feat, 0)
        adjusted_supply = supply * (1 + _TOLERANCE_MARGIN)
        if demand > adjusted_supply:
            shortage = demand - supply
            course_keys = feature_demand[feat]
            course_names = [
                f"{ck[0]} ({ck[1]})" for ck in course_keys[:5] if ck in courses
            ]
            capacity_bottlenecks.append(
                {
                    "feature": feat,
                    "demand_quanta": demand,
                    "supply_quanta": supply,
                    "shortage_quanta": shortage,
                    "room_count": len(feature_practical_rooms.get(feat, [])),
                    "rooms": feature_practical_rooms.get(feat, []),
                    "required_by_courses": len(course_keys),
                    "sample_courses": course_names,
                }
            )

    # ── 6. Build missing-feature details (Layer 1 failures) ────────
    missing_details: list[dict[str, Any]] = []
    for feat in sorted(missing_features):
        course_keys = feature_demand[feat]
        course_names = [f"{ck[0]} ({ck[1]})" for ck in course_keys[:5] if ck in courses]
        missing_details.append(
            {
                "feature": feat,
                "required_by_courses": len(course_keys),
                "sample_courses": course_names,
                "total_quanta_demand": feature_quanta[feat],
                "reason": "not in any room",
            }
        )

    # ── 7. Build type-mismatch details (Layer 2 warnings) ──────────
    mismatch_details: list[dict[str, Any]] = []
    for feat in sorted(type_mismatch_features):
        course_keys = feature_demand[feat]
        course_names = [f"{ck[0]} ({ck[1]})" for ck in course_keys[:5] if ck in courses]
        mismatch_details.append(
            {
                "feature": feat,
                "required_by_courses": len(course_keys),
                "sample_courses": course_names,
                "total_quanta_demand": feature_quanta[feat],
                "reason": (
                    f"only on lecture room(s) {feature_rooms[feat]}, "
                    "but listed for practical courses"
                ),
            }
        )

    # ── 8. Determine pass/fail ─────────────────────────────────────
    # CRITICAL: missing features OR capacity bottlenecks
    # WARNING:  type mismatches
    passed = len(missing_features) == 0 and len(capacity_bottlenecks) == 0

    usable_count = len(present_features - type_mismatch_features)

    if passed and not type_mismatch_features:
        message = (
            f"All {len(required_features)} required practical room features exist "
            f"with sufficient capacity ({usable_count} matched, "
            f"{len(practical_features)} in practical rooms) [!ok]"
        )
    elif passed:
        message = (
            f"All {len(required_features)} features found; "
            f"{len(type_mismatch_features)} only on lecture rooms (warning) [!ok]"
        )
    else:
        parts = []
        if missing_features:
            parts.append(f"{len(missing_features)} MISSING from all rooms")
        if capacity_bottlenecks:
            parts.append(f"{len(capacity_bottlenecks)} have insufficient room capacity")
        message = (
            f"{' + '.join(parts)} out of {len(required_features)} "
            f"required practical features ✗"
        )
        if type_mismatch_features:
            message += f" (+ {len(type_mismatch_features)} only on lecture rooms)"

    # ── 9. Recommendations ─────────────────────────────────────────
    recommendations = []

    if missing_details:
        recommendations.append("MISSING features (no room provides these):")
        for d in missing_details:
            samples = ", ".join(d["sample_courses"])
            recommendations.append(
                f"  • '{d['feature']}': {d['reason']} — needed by "
                f"{d['required_by_courses']} course(s) "
                f"({d['total_quanta_demand']} quanta) — e.g. {samples}"
            )
        recommendations.extend(
            [
                "",
                "Solutions:",
                "• Add the missing features to practical rooms in Rooms.json",
                "• Re-map course PracticalRoomFeatures in Course.json "
                "to existing room features",
            ]
        )

    if capacity_bottlenecks:
        if missing_details:
            recommendations.append("")
        recommendations.append(
            "CAPACITY bottlenecks (feature exists but not enough room-hours):"
        )
        for b in capacity_bottlenecks:
            samples = ", ".join(b["sample_courses"])
            recommendations.append(
                f"  • '{b['feature']}': demand={b['demand_quanta']}q, "
                f"supply={b['supply_quanta']}q, "
                f"shortage={b['shortage_quanta']}q — "
                f"{b['room_count']} room(s) {b['rooms']} — "
                f"e.g. {samples}"
            )
        recommendations.extend(
            [
                "",
                "Solutions:",
                "• Add the feature to more practical/both rooms in Rooms.json",
                "• Increase room availability hours",
                "• Reduce practical hours for courses using these features",
            ]
        )

    if mismatch_details:
        if missing_details or capacity_bottlenecks:
            recommendations.append("")
        recommendations.append(
            "WARNING: Features only on lecture rooms "
            "(practical courses may need re-mapping):"
        )
        for d in mismatch_details:
            samples = ", ".join(d["sample_courses"])
            recommendations.append(
                f"  • '{d['feature']}': {d['reason']} — e.g. {samples}"
            )

    if passed and not type_mismatch_features:
        recommendations.append(
            f"All {usable_count} required practical features matched "
            f"to rooms with sufficient capacity"
        )

    return FeasibilityResult(
        check_name="Specific Lab Feature Availability",
        passed=passed,
        severity="critical",
        message=message,
        details={
            "required_features": len(required_features),
            "available_features": len(available_features),
            "practical_features": len(practical_features),
            "missing_count": len(missing_features),
            "type_mismatch_count": len(type_mismatch_features),
            "capacity_bottleneck_count": len(capacity_bottlenecks),
            "missing_features": missing_details,
            "type_mismatch_warnings": mismatch_details,
            "capacity_bottlenecks": capacity_bottlenecks,
            "present_count": usable_count,
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
        adjusted_available = available * (1 + _TOLERANCE_MARGIN)

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
        recommendations.extend(
            f"  • {g['group_name']} ({g['group_id']}): "
            f"needs {g['demand']} quanta but only {g['available']} available "
            f"({g['utilization']:.0f}% utilization, {g['num_courses']} courses)"
            for g in overloaded_groups
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
        severity="critical",
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
    Generate a detailed markdown report file with feasibility analysis results.

    Args:
        report: FeasibilityReport to save
        output_path: Path to save the report
    """
    from datetime import datetime
    from pathlib import Path

    with Path(output_path).open("w", encoding="utf-8") as f:
        status = report.summary["status"].upper()
        status_icon = "" if status == "FEASIBLE" else ""

        f.write("# Feasibility Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Status:** {status_icon} {status}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Checks | {report.summary['total_checks']} |\n")
        f.write(f"| Passed | {report.summary['passed']} |\n")
        f.write(f"| Failed | {report.summary['failed']} |\n")
        f.write(f"| Critical Failures | {report.summary['critical_failures']} |\n")
        f.write("\n---\n\n")

        # Detailed results
        f.write("## Detailed Results\n\n")

        for i, result in enumerate(report.results, 1):
            pass_icon = "" if result.passed else ""
            severity_badge = f"`{result.severity.upper()}`"

            f.write(f"### {i}. {result.check_name}\n\n")
            f.write("| | |\n|---|---|\n")
            f.write(
                f"| **Status** | {pass_icon} {'PASS' if result.passed else 'FAIL'} |\n"
            )
            f.write(f"| **Severity** | {severity_badge} |\n")
            f.write(f"| **Message** | {result.message} |\n\n")

            # Write detailed information
            if result.details:
                f.write("#### Details\n\n")

                if result.details.get("bottlenecks"):
                    bottlenecks = result.details["bottlenecks"]
                    f.write(f"**Bottlenecks Found:** {len(bottlenecks)}\n\n")

                    # Render bottlenecks as a table if they have consistent keys
                    if bottlenecks:
                        keys = list(bottlenecks[0].keys())
                        f.write(
                            "| # | "
                            + " | ".join(k.replace("_", " ").title() for k in keys)
                            + " |\n"
                        )
                        f.write("|---" * (len(keys) + 1) + "|\n")
                        for j, bottleneck in enumerate(bottlenecks, 1):
                            vals = [str(bottleneck.get(k, "")) for k in keys]
                            f.write(f"| {j} | " + " | ".join(vals) + " |\n")
                        f.write("\n")

                elif (
                    "details" in result.details
                    and isinstance(result.details["details"], list)
                    and result.details["details"]
                ):
                    groups_list = result.details["details"]
                    f.write(f"**Overloaded Groups:** {len(groups_list)}\n\n")

                    if groups_list:
                        keys = list(groups_list[0].keys())
                        f.write(
                            "| # | "
                            + " | ".join(k.replace("_", " ").title() for k in keys)
                            + " |\n"
                        )
                        f.write("|---" * (len(keys) + 1) + "|\n")
                        for j, group_info in enumerate(groups_list, 1):
                            vals = [str(group_info.get(k, "")) for k in keys]
                            f.write(f"| {j} | " + " | ".join(vals) + " |\n")
                        f.write("\n")

                else:
                    # Generic key-value details
                    has_simple = False
                    for key, value in result.details.items():
                        if key not in (
                            "bottlenecks",
                            "details",
                            "missing_features",
                            "type_mismatch_warnings",
                        ) and not isinstance(value, list | dict):
                            if not has_simple:
                                f.write("| Key | Value |\n|-----|-------|\n")
                                has_simple = True
                            f.write(f"| {key.replace('_', ' ').title()} | {value} |\n")
                    if has_simple:
                        f.write("\n")

                    # Handle missing_features and type_mismatch_warnings lists
                    for list_key in ("missing_features", "type_mismatch_warnings"):
                        if result.details.get(list_key):
                            items = result.details[list_key]
                            header = list_key.replace("_", " ").title()
                            f.write(f"**{header}:**\n\n")
                            for item in items:
                                if isinstance(item, dict):
                                    feat = item.get("feature", "?")
                                    reason = item.get("reason", "")
                                    samples = ", ".join(item.get("sample_courses", []))
                                    f.write(f"- **{feat}**: {reason}")
                                    if samples:
                                        f.write(f" — e.g. {samples}")
                                    f.write("\n")
                            f.write("\n")

            if result.recommendations:
                f.write("#### Recommendations\n\n")
                for rec in result.recommendations:
                    rec_stripped = rec.strip()
                    if not rec_stripped:
                        continue
                    # Detect section headers (lines ending with :)
                    if rec_stripped.endswith(":") and not rec_stripped.startswith("•"):
                        f.write(f"**{rec_stripped}**\n\n")
                    elif rec_stripped.startswith("•"):
                        f.write(f"- {rec_stripped[1:].strip()}\n")
                    elif rec_stripped.startswith("  •"):
                        f.write(f"  - {rec_stripped[3:].strip()}\n")
                    else:
                        f.write(f"- {rec_stripped}\n")
                f.write("\n")

            f.write("---\n\n")

        # Footer
        if report.is_feasible:
            f.write(
                ">  **Problem is feasible** — all critical checks passed. GA should find a solution.  \n"
            )
            f.write("> *Note: this doesn't guarantee a perfect solution exists.*\n")
        else:
            f.write(
                f">  **Problem is INFEASIBLE** — {report.summary['critical_failures']} critical issue(s) found.  \n"
            )
            f.write("> *GA will not find a valid solution until these are fixed.*\n")
