"""
Course-Group Pair Generator

Generates (Course, Group) pairs following parent-subgroup rules:
- Theory (L+T): Parent group (all subgroups attend together)
- Practical (P): Each subgroup separately
"""

from src.entities.course import Course
from src.entities.group import Group


def generate_course_group_pairs(
    courses: dict[tuple[str, str], Course],
    groups: dict[str, Group],
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
    silent: bool = False,
) -> list[tuple[tuple[str, str], list[str], str, int]]:
    """
    Generates (course_id, group_ids, session_type, num_quanta) tuples.

    Rules:
    - Theory sessions: Group sibling subgroups together (e.g., BAE2A + BAE2B attend together)
    - Practical sessions: Assign to each subgroup separately
    - Standalone groups: Get both theory and practical

    Args:
        courses: Dictionary keyed by (course_code, course_type) tuple -> Course
        groups: Dictionary of group_id -> Group
        hierarchy: Output from analyze_group_hierarchy()
        silent: If True, suppress warning messages (useful for worker processes)

    Returns:
        List of tuples: (course_key, group_ids, session_type, num_quanta)
        where course_key is (course_code, course_type) tuple

    Example:
        [
            (("ENME 151", "theory"), ["BAE2A", "BAE2B"], "theory", 5),     # Theory: all subgroups together
            (("ENME 151", "practical"), ["BAE2A"], "practical", 3),        # Practical for subgroup A
            (("ENME 151", "practical"), ["BAE2B"], "practical", 3),        # Practical for subgroup B
        ]

    Note: course_id is now tuple key (course_code, course_type) from courses dict.
    """
    pairs: list[tuple[tuple[str, str], list[str], str, int]] = []

    # Group all subgroups by their parent prefix (e.g., BAE2A, BAE2B -> BAE2)
    # This allows us to find siblings that should attend theory together
    from collections import defaultdict

    parent_to_subgroups = defaultdict(list)

    for group_id in groups:
        # Check if this is a subgroup (ends with letter)
        if len(group_id) > 1 and group_id[-1].isalpha():
            parent_prefix = group_id[:-1]
            parent_to_subgroups[parent_prefix].append(group_id)
        else:
            # Standalone group (no siblings) OR parent that already has subgroups
            # Only create a new entry when no subgroups were registered yet
            if group_id not in parent_to_subgroups:
                parent_to_subgroups[group_id] = [group_id]

    # Process each group of siblings
    for parent_prefix, sibling_ids in parent_to_subgroups.items():
        # Get enrolled courses from first sibling (they should all have same courses)
        first_sibling = groups[sibling_ids[0]]
        enrolled_courses = first_sibling.enrolled_courses

        for course_code in enrolled_courses:
            # Find all courses matching this course_code (theory and/or practical)
            theory_key = (course_code, "theory")
            practical_key = (course_code, "practical")

            matching_courses = []
            if theory_key in courses:
                matching_courses.append((theory_key, courses[theory_key]))
            if practical_key in courses:
                matching_courses.append((practical_key, courses[practical_key]))

            if not matching_courses:
                if not silent:
                    print(
                        f"[!] Warning: Course {course_code} not found for group {parent_prefix}"
                    )
                continue

            # Process theory and practical courses separately
            for course_key, course in matching_courses:
                if course.course_type == "theory":
                    # Theory: ALL siblings attend together
                    # List all sibling IDs explicitly (e.g., ["BAE2A", "BAE2B"])
                    theory_quanta = course.quanta_per_week
                    pairs.append(
                        (course_key, sorted(sibling_ids), "theory", theory_quanta)
                    )

                elif course.course_type == "practical":
                    # Practical: Each sibling gets separate session
                    practical_quanta = course.quanta_per_week
                    for sibling_id in sibling_ids:
                        pairs.append(
                            (course_key, [sibling_id], "practical", practical_quanta)
                        )

    return pairs


def count_total_genes(pairs: list[tuple[tuple[str, str], list[str], str, int]]) -> int:
    """Count total number of genes that will be created."""
    return sum(num_quanta for _, _, _, num_quanta in pairs)


def group_pairs_by_course(
    pairs: list[tuple[tuple[str, str], list[str], str, int]],
) -> dict[tuple[str, str], list[tuple[tuple[str, str], list[str], str, int]]]:
    """Group pairs by course for analysis."""
    from collections import defaultdict

    course_pairs = defaultdict(list)
    for pair in pairs:
        course_key = pair[0]  # (course_code, course_type) tuple
        course_pairs[course_key].append(pair)
    return dict(course_pairs)
