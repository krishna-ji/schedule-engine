"""
Verify constraint registry single source of truth implementation.

Run this to validate:
1. All constraints are properly registered
2. Order is deterministic (hc1-hc8 mapping)
3. No duplicates or missing constraints
"""

from schedule_engine.constraints.registry import (
    get_all_hard_constraints,
    get_all_soft_constraints,
    get_registry_stats,
)


def verify_constraint_system():
    print("=" * 70)
    print("CONSTRAINT REGISTRY VERIFICATION")
    print("=" * 70)
    print()

    # Get registry stats
    stats = get_registry_stats()
    print(f"Total hard constraints: {stats['total_hard_constraints']}")
    print(f"Total soft constraints: {stats['total_soft_constraints']}")
    print()

    # Verify hard constraint order (hc1-hc8 mapping)
    print("HARD CONSTRAINT MAPPING (hc1-hc8):")
    print("-" * 70)
    hard_constraints = get_all_hard_constraints()
    for i, (name, metadata) in enumerate(hard_constraints.items(), 1):
        weight = metadata.default_weight
        needs_courses = "✓" if metadata.needs_courses else "✗"
        print(f"hc{i:2} = {name:30} (weight={weight:.1f}, courses={needs_courses})")
    print()

    # Verify soft constraint order (sc1-sc4 mapping)
    print("SOFT CONSTRAINT MAPPING (sc1-sc4):")
    print("-" * 70)
    soft_constraints = get_all_soft_constraints()
    for i, (name, metadata) in enumerate(soft_constraints.items(), 1):
        weight = metadata.default_weight
        needs_courses = "✓" if metadata.needs_courses else "✗"
        print(f"sc{i:2} = {name:30} (weight={weight:.1f}, courses={needs_courses})")
    print()

    # Verify expected constraints exist
    print("VALIDATION CHECKS:")
    print("-" * 70)

    expected_hard = [
        "student_group_exclusivity",
        "instructor_exclusivity",
        "instructor_qualifications",
        "room_suitability",  # hc4
        "instructor_time_availability",  # hc5
        "room_time_availability",  # hc6
        "course_completeness",  # hc7
        "room_exclusivity",  # hc8
    ]

    expected_soft = [
        "student_schedule_compactness",
        "instructor_schedule_compactness",
        "student_lunch_break",
        "session_continuity",
    ]

    # Check hard constraints
    actual_hard = list(hard_constraints.keys())
    if actual_hard == expected_hard:
        print("✓ Hard constraints: All 8 present in correct order")
    else:
        print("✗ Hard constraints: ORDER MISMATCH!")
        print(f"  Expected: {expected_hard}")
        print(f"  Actual:   {actual_hard}")

    # Check soft constraints
    actual_soft = list(soft_constraints.keys())
    if actual_soft == expected_soft:
        print("✓ Soft constraints: All 4 present in correct order")
    else:
        print("✗ Soft constraints: ORDER MISMATCH!")
        print(f"  Expected: {expected_soft}")
        print(f"  Actual:   {actual_soft}")

    # Verify no duplicates
    if len(set(actual_hard)) == len(actual_hard):
        print("✓ No duplicate hard constraints")
    else:
        print("✗ DUPLICATE hard constraints detected!")

    if len(set(actual_soft)) == len(actual_soft):
        print("✓ No duplicate soft constraints")
    else:
        print("✗ DUPLICATE soft constraints detected!")

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    verify_constraint_system()
