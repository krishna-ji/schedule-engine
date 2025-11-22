"""Debug script to check if all course-group pairs are being created as genes."""

import json
from pathlib import Path
from collections import defaultdict

# Load data
courses_data = json.loads(Path("data/Course.json").read_text())
groups_data = json.loads(Path("data/Groups.json").read_text())

# Build expected course-group pairs
expected_pairs = []
for group in groups_data:
    group_id = group["group_id"]
    enrolled_courses = group.get("courses", [])

    for course_code in enrolled_courses:
        # Find course details
        course = next((c for c in courses_data if c["course_id"] == course_code), None)
        if course:
            course_type = course.get("course_type", "theory")
            expected_pairs.append((course_code, course_type, group_id))

print(f"Expected course-group pairs: {len(expected_pairs)}")
print(f"\nFirst 10 pairs:")
for i, (cid, ctype, gid) in enumerate(expected_pairs[:10]):
    print(f"  {i+1}. ({cid}, {ctype}, {gid})")

# Count by course type
theory_count = sum(1 for _, ctype, _ in expected_pairs if ctype == "theory")
practical_count = sum(1 for _, ctype, _ in expected_pairs if ctype == "practical")
print(f"\nBy type:")
print(f"  Theory: {theory_count}")
print(f"  Practical: {practical_count}")

# Count quanta requirements
total_quanta = 0
for course_code, course_type, group_id in expected_pairs:
    course = next(
        (
            c
            for c in courses_data
            if c["course_id"] == course_code and c["course_type"] == course_type
        ),
        None,
    )
    if course:
        quanta = course.get("quanta_per_week", 0)
        total_quanta += quanta

print(f"\nTotal quanta needed: {total_quanta}")
print(f"Expected hc7 if missing all: {len(expected_pairs)}")
print(f"Current hc7 from log: 680")
print(f"Missing genes: {680} violations suggests incomplete coverage")
