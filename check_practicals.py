#!/usr/bin/env python3
"""
Quick script to check if we have practical courses to trigger cohort pairing penalties.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.notebooks.core import load_data


def main() -> None:
    """Check for practical courses."""
    print("=== Checking for Practical Courses ===\n")

    # Load data
    data_dir = PROJECT_ROOT / "data"
    data = load_data(
        data_dir=data_dir,
        opening_time="10:00",
        closing_time="17:00",
        closed_days=["Saturday"],
    )

    practical_courses = []
    for (course_code, course_type), course in data.courses.items():
        if course_type.lower() == "practical":
            practical_courses.append(
                (course_code, course_type, course.enrolled_group_ids)
            )

    print(f"Total practical courses: {len(practical_courses)}")

    if practical_courses:
        print("\nPractical courses found:")
        for course_code, course_type, enrolled_groups in practical_courses[
            :10
        ]:  # Show first 10
            print(f"  {course_code} ({course_type}): {enrolled_groups}")

        if len(practical_courses) > 10:
            print(f"  ... and {len(practical_courses) - 10} more")

        # Check if any practical courses have paired cohorts
        context = data.context
        paired_practical_count = 0

        for course_code, course_type, enrolled_groups in practical_courses:
            for left, right in context.cohort_pairs or []:
                if left in enrolled_groups and right in enrolled_groups:
                    paired_practical_count += 1
                    break  # Count each course only once

        print(f"\nPractical courses with paired cohorts: {paired_practical_count}")

    else:
        print(" NO PRACTICAL COURSES FOUND!")
        print("→ This explains why paired_cohort penalty remains 0")
        print("→ All courses are theory/lecture type, no practicals to align")


if __name__ == "__main__":
    main()
