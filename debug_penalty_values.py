#!/usr/bin/env python3
"""
Debug script to understand why cohort pairing is 0 and break placement is so high.
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
from schedule_engine.ga.run_helpers import create_random_individual, load_data
from schedule_engine.io.decoder import decode_individual


def main() -> None:
    """Debug the constraint values."""
    print("=== Debugging Constraint Values ===\n")

    # Load data
    data_dir = PROJECT_ROOT / "data"
    data = load_data(
        data_dir=data_dir,
        opening_time="10:00",
        closing_time="17:00",
        closed_days=["Saturday"],
    )

    context = data.context
    print(f"Cohort pairs available: {len(context.cohort_pairs or [])}")

    # Create test individual
    test_individual = create_random_individual(data)
    sessions = decode_individual(
        test_individual,
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    print(f"Test sessions: {len(sessions)}")

    print("\n1. BREAK PLACEMENT ANALYSIS:")

    # Check how many groups and days we're evaluating
    from schedule_engine.constraints.soft import (
        _build_group_day_schedules,
        _get_break_window_quanta,
    )
    from schedule_engine.io.time_system import QuantumTimeSystem

    qts = QuantumTimeSystem()

    break_penalty = qts.break_violation_penalty
    min_free = qts.break_min_quanta

    print(f"   Break penalty per violation: {break_penalty}")
    print(f"   Min free quanta required: {min_free}")
    break_windows = _get_break_window_quanta(qts)
    group_schedules = _build_group_day_schedules(sessions, qts)

    print(f"   Break windows (days): {list(break_windows.keys())}")
    print(f"   Groups being evaluated: {len(group_schedules)}")

    # Count violations manually
    total_violations = 0
    violation_count = 0

    for (group_id, day_name), occupied_quanta in group_schedules.items():
        if day_name not in break_windows:
            continue

        break_quanta = break_windows[day_name]
        occupied_in_break = occupied_quanta & break_quanta
        free_in_break = len(break_quanta) - len(occupied_in_break)

        if free_in_break < min_free:
            shortage = min_free - free_in_break
            penalty = shortage * break_penalty
            total_violations += penalty
            violation_count += 1

            if violation_count <= 5:  # Show first 5 violations
                print(
                    f"   Violation {violation_count}: {group_id} on {day_name}: {shortage} quanta short = {penalty} penalty"
                )

    print(f"   Total violations: {violation_count}")
    print(f"   Total penalty: {total_violations}")
    print(f"   ➜ Penalty per quantum shortage: {break_penalty}")
    print(
        f"   ➜ RECOMMENDATION: Reduce break_violation_penalty from {break_penalty} to 1-10"
    )

    print("\n2. COHORT PAIRING ANALYSIS:")

    # Check cohort pairs access
    cohort_pairs_in_config = getattr(cfg, "cohort_pairs", [])
    print(f"   Cohort pairs in config: {len(cohort_pairs_in_config)}")
    print(f"   Cohort pairs in context: {len(context.cohort_pairs or [])}")

    if not cohort_pairs_in_config and context.cohort_pairs:
        print("    ISSUE: Config doesn't have cohort pairs, but context does!")
        print("   ➜ The constraint function gets pairs from config, not context")

    # Test the constraint with proper debugging
    import schedule_engine.constraints.soft as soft_module

    # Temporarily enable debug mode
    original_debug = getattr(cfg, "debug_sc5", False)
    cfg.debug_sc5 = True

    try:
        penalty = paired_cohort_practical_alignment(sessions, context.courses)
        print(f"   Cohort pairing penalty: {penalty}")
    finally:
        cfg.debug_sc5 = original_debug


if __name__ == "__main__":
    main()
