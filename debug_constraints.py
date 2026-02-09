#!/usr/bin/env python3
"""
Debug script to check why break placement and cohort pairing penalties are 0.

This script will check:
1. Break placement configuration (enforce_break_placement, break windows, etc.)
2. Cohort pairs configuration
3. The actual soft constraint function behavior
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.constraints.soft import (
    break_placement_compliance,
    paired_cohort_practical_alignment,
)
from schedule_engine.ga.run_helpers import load_data
from schedule_engine.io.decoder import decode_individual


def main() -> None:
    """Check configuration and constraint function behavior."""
    print("=== Debugging Break Placement & Cohort Pairing Issues ===\n")

    # Load data using the same approach as mode_b_memetic
    data_dir = PROJECT_ROOT / "data"
    data = load_data(
        data_dir=data_dir,
        opening_time="10:00",
        closing_time="17:00",
        closed_days=["Saturday"],
    )

    context = data.context

    print("1. DATA LOADING:")
    print(f"   {data.summary()}")

    print("\n2. COHORT PAIRS CHECK:")
    print(f"   cohort_pairs in context: {context.cohort_pairs}")
    print(f"   total pairs: {len(context.cohort_pairs or [])}")

    if context.cohort_pairs:
        for i, (left, right) in enumerate(context.cohort_pairs):
            print(f"   pair {i+1}: {left} <-> {right}")
    else:
        print("    NO COHORT PAIRS FOUND!")

    print("\n3. GROUPS ANALYSIS:")
    print(f"   total groups: {len(context.groups)}")

    # Check for subgroups pattern
    groups_with_subgroups = []
    for group_id, group in context.groups.items():
        if hasattr(group, "subgroups") and group.subgroups:
            groups_with_subgroups.append((group_id, [sg.id for sg in group.subgroups]))

    print(f"   groups with subgroups: {len(groups_with_subgroups)}")
    for group_id, subgroup_ids in groups_with_subgroups[:5]:  # Show first 5
        print(f"     {group_id}: {subgroup_ids}")

    print("\n4. BREAK PLACEMENT CONFIG:")
    # Since we're using the legacy load_data, check if break placement is configured
    # The legacy system might not have this configuration
    try:
        from schedule_engine.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        print(f"   enforce_break_placement: {qts.enforce_break_placement}")
        print(f"   break_window_start: {qts.break_window_start}")
        print(f"   break_window_end: {qts.break_window_end}")
        print(f"   break_min_quanta: {qts.break_min_quanta}")
    except Exception as e:
        print(f"    QTS NOT AVAILABLE: {e}")

    print("\n5. TEST CONSTRAINT FUNCTIONS:")
    # Create a test individual to see what happens
    from schedule_engine.ga.run_helpers import create_random_individual

    test_individual = create_random_individual(data)

    # Decode it to sessions
    sessions = decode_individual(
        test_individual,
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    print(f"   test individual: {len(test_individual)} genes")
    print(f"   decoded sessions: {len(sessions)} sessions")

    # Test cohort pairing function
    try:
        cohort_penalty = paired_cohort_practical_alignment(sessions, context.courses)
        print(f"   cohort pairing penalty: {cohort_penalty}")

        if cohort_penalty == 0:
            if not context.cohort_pairs:
                print("     → REASON: No cohort pairs configured!")
            else:
                print("     → REASON: May be no practical courses or perfectly aligned")
    except Exception as e:
        print(f"    cohort pairing test failed: {e}")

    # Test break placement function
    try:
        break_penalty = break_placement_compliance(sessions)
        print(f"   break placement penalty: {break_penalty}")

        if break_penalty == 0:
            print("     → REASON: Either disabled or no break violations")
    except Exception as e:
        print(f"    break placement test failed: {e}")

    print("\n6. RECOMMENDATIONS:")
    if not context.cohort_pairs:
        print("    FIX COHORT PAIRING:")
        print("      → Check Groups.json - ensure parent groups have subgroups")
        print("      → Or manually configure cohort_pairs in time config")

    print("    FIX BREAK PLACEMENT:")
    print("      → Use proper config system instead of legacy load_data")
    print("      → Ensure enforce_break_placement = True")
    print("      → Configure break windows appropriately")


if __name__ == "__main__":
    main()
