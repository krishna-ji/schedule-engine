"""
Data Quality Diagnostic Script
Checks Groups.json for duplicate course enrollments and other data integrity issues.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def check_duplicate_enrollments(groups_file="data/Groups.json"):
    """Check for duplicate course enrollments in groups."""

    print(f"\n{'='*60}")
    print("[!report] DATA QUALITY DIAGNOSTIC REPORT")
    print(f"{'='*60}\n")
    print(f"Checking file: {groups_file}\n")

    try:
        with open(groups_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[err!] ERROR: File not found: {groups_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"[err!] ERROR: Invalid JSON: {e}")
        return []

    if not isinstance(data, list):
        print("[err!] ERROR: Expected list of groups in JSON file")
        return []

    print(f"[!yes] Loaded {len(data)} groups\n")

    # Track issues
    issues = []
    stats = {
        "total_groups": len(data),
        "groups_with_duplicates": 0,
        "total_duplicate_instances": 0,
        "groups_without_courses": 0,
    }

    for i, group in enumerate(data):
        # Get group identifier
        group_id = group.get("group_id") or group.get("id") or f"group_{i}"

        # Get enrolled courses
        enrolled = group.get("enrolled_courses", [])

        if not enrolled:
            stats["groups_without_courses"] += 1
            continue

        # Count occurrences of each course
        course_counts = Counter(enrolled)

        # Find duplicates
        duplicates = {
            course: count for course, count in course_counts.items() if count > 1
        }

        if duplicates:
            stats["groups_with_duplicates"] += 1
            stats["total_duplicate_instances"] += sum(
                count - 1 for count in duplicates.values()
            )

            issues.append(
                {
                    "group": group_id,
                    "duplicates": duplicates,
                    "total_courses": len(enrolled),
                    "unique_courses": len(course_counts),
                }
            )

    # Print results
    print(f"{'-'*60}")
    print("SUMMARY STATISTICS")
    print(f"{'-'*60}")
    print(f"Total groups analyzed:        {stats['total_groups']}")
    print(f"Groups with duplicates:       {stats['groups_with_duplicates']}")
    print(f"Groups without courses:       {stats['groups_without_courses']}")
    print(f"Total duplicate instances:    {stats['total_duplicate_instances']}")
    print(f"{'-'*60}\n")

    if issues:
        print(f"[err!] FOUND {len(issues)} GROUPS WITH DUPLICATE ENROLLMENTS:\n")

        for issue in sorted(
            issues, key=lambda x: sum(x["duplicates"].values()), reverse=True
        ):
            print(f"  [!attention] Group: {issue['group']}")
            print(
                f"     Total courses: {issue['total_courses']} ({issue['unique_courses']} unique)"
            )
            print("     Duplicates:")

            for course, count in sorted(
                issue["duplicates"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"       • {course} → appears {count} times")
            print()

        print(f"\n{'='*60}")
        print("⚠️  ACTION REQUIRED")
        print(f"{'='*60}")
        print("These duplicates will cause group overlap violations!")
        print("Each duplicate creates additional scheduling conflicts.")
        print(f"\nPlease review and fix {groups_file}:")
        print("  1. Check if duplicates are intentional (e.g., theory + lab)")
        print("  2. If unintentional, remove duplicate entries")
        print("  3. If intentional, ensure they're separate course sections")
        print(f"{'='*60}\n")

    else:
        print("[!yes] NO DUPLICATE ENROLLMENTS FOUND!")
        print("   All groups have unique course assignments.\n")

    # Additional checks
    print(f"\n{'-'*60}")
    print("ADDITIONAL CHECKS")
    print(f"{'-'*60}")

    # Check for groups with many courses
    large_groups = []
    for group in data:
        group_id = group.get("group_id") or group.get("id", "unknown")
        enrolled = group.get("enrolled_courses", [])
        if len(enrolled) > 10:
            large_groups.append((group_id, len(enrolled)))

    if large_groups:
        print("\n⚠️  Groups with >10 courses:")
        for gid, count in sorted(large_groups, key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {gid}: {count} courses")
        print("   (Large course loads may be difficult to schedule)")

    print(f"\n{'='*60}\n")

    return issues


def check_data_consistency(base_dir="data"):
    """Additional consistency checks across data files."""

    print(f"\n{'='*60}")
    print("[!search] CROSS-FILE CONSISTENCY CHECKS")
    print(f"{'='*60}\n")

    files = {
        "courses": Path(base_dir) / "Course.json",
        "groups": Path(base_dir) / "Groups.json",
        "instructors": Path(base_dir) / "Instructors.json",
        "rooms": Path(base_dir) / "Rooms.json",
    }

    data = {}
    for key, path in files.items():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data[key] = json.load(f)
            print(f"[!yes] Loaded {len(data[key])} {key}")
        except Exception as e:
            print(f"[err!] Failed to load {key}: {e}")
            return

    print()

    # Check if courses referenced by groups exist
    course_ids = {c.get("course_id") or c.get("id") for c in data["courses"]}

    missing_courses = set()
    for group in data["groups"]:
        enrolled = group.get("enrolled_courses", [])
        for course_id in enrolled:
            if course_id not in course_ids:
                missing_courses.add(course_id)

    if missing_courses:
        print(f"⚠️  {len(missing_courses)} course IDs referenced but not found:")
        for cid in sorted(missing_courses)[:20]:
            print(f"   • {cid}")
        if len(missing_courses) > 20:
            print(f"   ... and {len(missing_courses) - 20} more")
    else:
        print("[!yes] All group courses found in Course.json")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print("\n[!info] Starting Data Quality Diagnostics...\n")

    # Check for duplicate enrollments
    issues = check_duplicate_enrollments()

    # Check cross-file consistency
    check_data_consistency()

    # Exit with error code if issues found
    if issues:
        print("⚠️  Please fix data quality issues before running GA!")
        sys.exit(1)
    else:
        print("[!yes] All data quality checks passed!")
        sys.exit(0)
