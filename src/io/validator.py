"""
Input Validation Module

Validates loaded data for consistency before GA execution.
Fails fast with clear error messages to prevent cryptic runtime failures.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, cast

from src.exceptions import DataValidationError
from src.utils.console_service import get_console

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.types import SchedulingContext

__all__ = ["InputValidator", "ValidationError", "validate_input"]

console = get_console()


class ValidationError(DataValidationError):
    """Represents a single validation error.

    Note: This inherits from DataValidationError (Exception subclass)
    for proper exception handling while maintaining backward compatibility
    with existing error collection code.
    """

    def __init__(self, category: str, message: str, severity: str = "ERROR"):
        """
        Args:
            category: Error category (e.g., "COURSE", "GROUP", "INSTRUCTOR")
            message: Detailed error message
            severity: "ERROR", "WARNING", or "INFO"
        """
        self.category = category
        self.message = message
        self.severity = severity
        super().__init__(f"[{severity}] {category}: {message}")

    def __str__(self) -> str:
        return f"[{self.severity}] {self.category}: {self.message}"

    def __repr__(self) -> str:
        return self.__str__()


class InputValidator:
    """
    Validates scheduling input data for consistency and completeness.

    Checks:
    - Required relationships exist
    - No orphaned entities
    - Data types are correct
    - Reasonable constraints (e.g., room capacity > 0)
    """

    def __init__(self, context: SchedulingContext):
        """
        Initialize validator with scheduling context.

        Args:
            context: SchedulingContext to validate
        """
        self.context = context
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

    def _find_courses_for_identifier(
        self, course_identifier: str
    ) -> list[tuple[Any, Course]]:
        """Return all course entries matching provided identifier."""
        courses = cast("dict[Any, Course]", self.context.courses)
        matches: list[tuple[Any, Course]] = []
        for key, course in courses.items():
            if key == course_identifier:
                matches.append((key, course))
                continue
            if isinstance(key, tuple) and key and key[0] == course_identifier:
                matches.append((key, course))
                continue
            course_code_attr = getattr(course, "course_code", None)
            if course_code_attr == course_identifier:
                matches.append((key, course))
        return matches

    def validate(self, parallel: bool = True) -> list[ValidationError]:
        """
        Run all validation checks.

        Args:
            parallel: If True, run independent validation checks concurrently (default: True)

        Returns:
            List of validation errors (empty if valid)
        """
        self.errors = []
        self.warnings = []

        if not parallel:
            # Sequential validation (for debugging)
            self._validate_courses()
            self._validate_groups()
            self._validate_instructors()
            self._validate_rooms()
            self._validate_relationships()
            self._validate_enrolled_courses_without_instructors()
            self._validate_availability()
            self._validate_room_features_for_enrolled_courses()
        else:
            # Parallel validation (3-4x faster)
            # Phase 1: Independent entity validations (can run fully in parallel)
            independent_checks = [
                self._validate_courses,
                self._validate_groups,
                self._validate_instructors,
                self._validate_rooms,
            ]

            from src.utils.system_info import get_cpu_count

            max_workers = get_cpu_count()  # Auto-detect all cores
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(check): check.__name__
                    for check in independent_checks
                }
                for future in as_completed(futures):
                    check_name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        console.print(f"[red]Error in {check_name}: {e}[/red]")

            # Phase 2: Relationship validations (depend on Phase 1 completion)
            relationship_checks = [
                self._validate_relationships,
                self._validate_enrolled_courses_without_instructors,
                self._validate_availability,
                self._validate_room_features_for_enrolled_courses,
            ]

            max_workers = get_cpu_count()  # Reuse from earlier
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(check): check.__name__
                    for check in relationship_checks
                }
                for future in as_completed(futures):
                    check_name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        console.print(f"[red]Error in {check_name}: {e}[/red]")

        return self.errors + self.warnings

    def _validate_courses(self) -> None:
        """Validate course data."""
        if not self.context.courses:
            self.errors.append(
                ValidationError(
                    "COURSE", "No courses loaded! Cannot generate schedule."
                )
            )
            return

        for course_id, course in self.context.courses.items():
            # Check required fields
            if not course.name:
                self.warnings.append(
                    ValidationError(
                        "COURSE", f"Course {course_id} has no name", "WARNING"
                    )
                )

            if course.quanta_per_week <= 0:
                self.errors.append(
                    ValidationError(
                        "COURSE",
                        f"Course {course_id} has invalid quanta_per_week: {course.quanta_per_week}",
                    )
                )

            # Check for qualified instructors
            if not course.qualified_instructor_ids:
                self.warnings.append(
                    ValidationError(
                        "COURSE",
                        f"Course {course_id} has no qualified instructors",
                        "WARNING",
                    )
                )

            # Check for practical courses
            if course.course_type == "practical" and not course.required_room_features:
                self.warnings.append(
                    ValidationError(
                        "COURSE",
                        f"Practical course {course_id} has no required room features",
                        "WARNING",
                    )
                )

    def _validate_groups(self) -> None:
        """Validate group data."""
        if not self.context.groups:
            self.errors.append(
                ValidationError("GROUP", "No groups loaded! Cannot generate schedule.")
            )
            return

        for group_id, group in self.context.groups.items():
            # Check enrolled courses
            if not group.enrolled_courses:
                self.warnings.append(
                    ValidationError(
                        "GROUP", f"Group {group_id} has no enrolled courses", "WARNING"
                    )
                )

            # Check size
            if group.student_count <= 0:
                self.warnings.append(
                    ValidationError(
                        "GROUP",
                        f"Group {group_id} has invalid student_count: {group.student_count}",
                        "WARNING",
                    )
                )

    def _validate_instructors(self) -> None:
        """Validate instructor data."""
        if not self.context.instructors:
            self.errors.append(
                ValidationError(
                    "INSTRUCTOR", "No instructors loaded! Cannot generate schedule."
                )
            )
            return

        for instructor_id, instructor in self.context.instructors.items():
            # Check qualified courses from ORIGINAL JSON data
            # Use original_qualified_courses if available to avoid false warnings
            # when courses are filtered out during enrollment filtering
            original_courses = getattr(instructor, "original_qualified_courses", None)

            if original_courses is None:
                # Fallback to current qualified_courses if original not stored
                original_courses = instructor.qualified_courses

            # Only warn if instructor has NO qualifications in the original JSON
            if not original_courses:
                self.warnings.append(
                    ValidationError(
                        "INSTRUCTOR",
                        f"Instructor {instructor_id} has no qualified courses in JSON data",
                        "WARNING",
                    )
                )
            # Informational: instructor has qualifications but none match enrolled courses
            elif not instructor.qualified_courses:
                # Silently skip - this is expected when instructor qualifications
                # don't match any enrolled courses. Not a data error.
                pass

            # Check availability for part-time instructors
            if not instructor.is_full_time and not instructor.available_quanta:
                self.warnings.append(
                    ValidationError(
                        "INSTRUCTOR",
                        f"Part-time instructor {instructor_id} has no availability defined",
                        "WARNING",
                    )
                )

    def _validate_rooms(self) -> None:
        """Validate room data."""
        if not self.context.rooms:
            self.errors.append(
                ValidationError("ROOM", "No rooms loaded! Cannot generate schedule.")
            )
            return

        for room_id, room in self.context.rooms.items():
            # Check capacity
            if room.capacity <= 0:
                self.errors.append(
                    ValidationError(
                        "ROOM", f"Room {room_id} has invalid capacity: {room.capacity}"
                    )
                )

    def _validate_relationships(self) -> None:
        """Validate relationships between entities."""
        # Check course-instructor relationships
        for course_id, course in self.context.courses.items():
            for instructor_id in course.qualified_instructor_ids:
                if instructor_id not in self.context.instructors:
                    self.errors.append(
                        ValidationError(
                            "RELATIONSHIP",
                            f"Course {course_id} references non-existent instructor {instructor_id}",
                        )
                    )
                else:
                    # Check reverse relationship
                    instructor = self.context.instructors[instructor_id]
                    if course_id not in instructor.qualified_courses:
                        self.warnings.append(
                            ValidationError(
                                "RELATIONSHIP",
                                f"Asymmetric relationship: Course {course_id} lists instructor {instructor_id}, "
                                f"but instructor doesn't list course",
                                "WARNING",
                            )
                        )

        # Check group-course relationships
        # Note: Groups reference course CODES, but context.courses uses course IDs
        # For pure practical courses, the ID becomes "CODE-PR"
        for group_id, group in self.context.groups.items():
            for course_code in group.enrolled_courses:
                matching_courses = self._find_courses_for_identifier(course_code)

                if not matching_courses:
                    # This is a WARNING, not an ERROR - courses with L=T=P=0 don't need scheduling
                    self.warnings.append(
                        ValidationError(
                            "RELATIONSHIP",
                            f"Group {group_id} enrolled in course {course_code} which doesn't exist "
                            f"(likely has L=T=P=0 in Course.json and doesn't need scheduling)",
                            "WARNING",
                        )
                    )

        # Check for orphaned courses (no groups enrolled)
        enrolled_courses = set()
        for group in self.context.groups.values():
            enrolled_courses.update(group.enrolled_courses)

        # Note: Removed warning for courses with no groups enrolled
        # (this is valid for elective courses or courses offered to specific groups only)

    def _validate_enrolled_courses_without_instructors(self) -> None:
        """
        Check if any enrolled courses have no qualified instructors.
        This is a CRITICAL error - courses that groups are enrolled in MUST have instructors.
        """
        # Get all courses that groups are enrolled in
        enrolled_courses = set()
        for group in self.context.groups.values():
            enrolled_courses.update(group.enrolled_courses)

        # Track courses without instructors
        courses_without_instructors = []

        for course_code in enrolled_courses:
            matching_courses = self._find_courses_for_identifier(course_code)

            # Check each matching course for qualified instructors
            for course_id, course in matching_courses:
                qualified_instructors = getattr(course, "qualified_instructor_ids", [])
                if not qualified_instructors or len(qualified_instructors) == 0:
                    # Convert course_id to string (it might be a tuple like ('CE707', 'practical'))
                    course_id_str = str(course_id)
                    courses_without_instructors.append(
                        (
                            course_id_str,
                            course.name if hasattr(course, "name") else "Unknown",
                        )
                    )

        # Report errors if found
        if courses_without_instructors:
            from rich.panel import Panel
            from rich.table import Table

            # Create table for better visualization
            table = Table(
                title="[!err] CRITICAL: Enrolled Courses Without Qualified Instructors",
                show_header=True,
                header_style="bold red",
            )
            table.add_column("Course ID", style="yellow", width=20)
            table.add_column("Course Name", style="cyan", width=40)
            table.add_column("Issue", style="red")

            for course_info in courses_without_instructors:
                cid = (
                    course_info[0]
                    if isinstance(course_info[0], str)
                    else str(course_info[0])
                )
                cname = course_info[1]
                table.add_row(cid, cname, "No qualified instructors")

            console.print()
            console.print(table)
            console.print()
            console.print(
                Panel(
                    f"[bold red]Found {len(courses_without_instructors)} enrolled course(s) without any qualified instructors![/bold red]\n\n"
                    f"[yellow]These courses have groups enrolled but no instructors can teach them.[/yellow]\n"
                    f"[cyan]Action required: Add qualified instructors in Course.json and Instructors.json[/cyan]",
                    title="[!ERR] VALIDATION ERROR",
                    border_style="red",
                )
            )
            console.print()

            # Add error for each course
            for course_info in courses_without_instructors:
                cid = (
                    course_info[0]
                    if isinstance(course_info[0], str)
                    else str(course_info[0])
                )
                cname = course_info[1]
                self.errors.append(
                    ValidationError(
                        "INSTRUCTOR_QUALIFICATION",
                        f"Enrolled course '{cid}' ({cname}) has NO qualified instructors - cannot be scheduled!",
                    )
                )

    def _validate_availability(self) -> None:
        """Validate availability constraints."""
        if not self.context.available_quanta:
            self.errors.append(
                ValidationError("AVAILABILITY", "No available time quanta defined!")
            )
            return

        # Check for sufficient time slots
        total_quanta_needed = 0
        for course_id, course in self.context.courses.items():
            # Count how many groups are enrolled
            groups_enrolled = sum(
                1
                for group in self.context.groups.values()
                if course_id in group.enrolled_courses
            )
            total_quanta_needed += course.quanta_per_week * groups_enrolled

        available_slots = len(self.context.available_quanta)

        if total_quanta_needed > available_slots:
            self.warnings.append(
                ValidationError(
                    "AVAILABILITY",
                    f"Tight schedule: Need {total_quanta_needed} quanta but only {available_slots} available. "
                    f"May result in conflicts.",
                    "WARNING",
                )
            )
        elif total_quanta_needed > available_slots * 0.8:
            self.warnings.append(
                ValidationError(
                    "AVAILABILITY",
                    f"Schedule is 80%+ full ({total_quanta_needed}/{available_slots} quanta). "
                    f"Limited flexibility for optimization.",
                    "INFO",
                )
            )

    def _validate_room_features_for_enrolled_courses(self) -> None:
        """
        Validate that rooms have required features for all enrolled courses.

        For each group:
        - Get all enrolled courses
        - Extract required room features from those courses
        - Check if at least one room exists with all those required features
        """
        # Collect all unique required features across all enrolled courses
        all_required_features = set()
        for course in self.context.courses.values():
            if course.required_room_features:
                # Handle both list and string formats
                if isinstance(course.required_room_features, list):
                    all_required_features.update(course.required_room_features)
                else:
                    all_required_features.add(course.required_room_features)

        # Check if rooms exist with each required feature
        available_room_features = set()
        for room in self.context.rooms.values():
            if room.room_features:
                # Handle both list and string formats
                if isinstance(room.room_features, list):
                    available_room_features.update(room.room_features)
                else:
                    available_room_features.add(room.room_features)

        # Find missing features
        missing_features = all_required_features - available_room_features

        if missing_features:
            self.errors.append(
                ValidationError(
                    "ROOM_FEATURES",
                    f"Required room features not available in any room: {', '.join(sorted(str(f) for f in missing_features))}. "
                    f"Courses requiring these features cannot be scheduled.",
                )
            )

        # Check for each group's enrolled courses
        for group_id, group in self.context.groups.items():
            group_required_features = set()
            problematic_courses = []

            for course_id in group.enrolled_courses:
                course_match = self._find_courses_for_identifier(course_id)
                matched_course: Course | None = (
                    course_match[0][1] if course_match else None
                )

                if matched_course and matched_course.required_room_features:
                    # Handle both list and string formats
                    features_list = (
                        matched_course.required_room_features
                        if isinstance(matched_course.required_room_features, list)
                        else [matched_course.required_room_features]
                    )
                    group_required_features.update(features_list)

                    # Check if any room can satisfy this requirement
                    has_suitable_room = False
                    for room in self.context.rooms.values():
                        # Get room features as list
                        room_features_list = (
                            room.room_features
                            if isinstance(room.room_features, list)
                            else [room.room_features]
                        )

                        # Check if any required feature matches any room feature
                        feature_match = any(
                            (req_feature in room_features_list)
                            or (
                                hasattr(room, "is_suitable_for_course_type")
                                and room.is_suitable_for_course_type(req_feature)
                            )
                            for req_feature in features_list
                        )

                        if feature_match and room.can_accommodate_group_size(
                            group.student_count
                        ):
                            has_suitable_room = True
                            break

                    if not has_suitable_room and matched_course is not None:
                        problematic_courses.append(
                            (
                                matched_course.course_id,
                                str(matched_course.required_room_features),
                            )
                        )

            # Report group-specific issues
            if problematic_courses:
                course_list = ", ".join(
                    [f"{cid}({feat})" for cid, feat in problematic_courses]
                )
                self.warnings.append(
                    ValidationError(
                        "ROOM_FEATURES",
                        f"Group {group_id} ({group.student_count} students) enrolled in courses requiring "
                        f"features that no suitable room can satisfy: {course_list}",
                        "WARNING",
                    )
                )

    def has_errors(self) -> bool:
        """Check if validation found any errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if validation found any warnings."""
        return len(self.warnings) > 0

    def print_report(self) -> None:
        """Print validation report to console."""
        # Don't print anything here - errors/warnings are already shown
        # in detailed tables during validation (e.g., enrolled courses without instructors)
        # Just show final status
        if self.errors:
            console.print()
            console.print(
                "[bold red][!err][/bold red] Validation FAILED! Fix errors before running GA."
            )
            console.print()
        elif self.warnings:
            console.print()
            console.print(
                "[bold yellow][!warn][/bold yellow] Validation passed with warnings. Review before running GA."
            )
            console.print()
        else:
            console.print()
            console.print(
                "[bold green][!ok][/bold green] Validation passed! No issues found."
            )
            console.print()


def validate_input(context: SchedulingContext, strict: bool = False) -> bool:
    """
    Validate input data and print report.

    Args:
        context: SchedulingContext to validate
        strict: If True, treat warnings as errors

    Returns:
        True if validation passed (or only warnings in non-strict mode)

    Raises:
        ValueError: If validation fails in strict mode
    """
    validator = InputValidator(context)
    validator.validate()

    validator.print_report()

    if validator.has_errors():
        if strict:
            raise ValueError("Input validation failed! See errors above.")
        return False

    if strict and validator.has_warnings():
        raise ValueError("Strict validation failed! Warnings present.")

    return True
