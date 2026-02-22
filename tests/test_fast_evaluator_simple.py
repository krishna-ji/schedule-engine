#!/usr/bin/env python3
"""
Simple evaluator equivalence test - directly testing just the evaluation functions.
"""

import pickle
import sys
import time
from pathlib import Path

import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.fast_evaluator import fast_conflict_evaluator


def create_dummy_assignments(events_data):
    """Create some test assignments for testing."""
    events = events_data["events"]
    n_events = len(events)

    # Create assignments using allowed domains
    instructor_assignments = []
    room_assignments = []
    start_assignments = []
    duration_assignments = []
    group_masks = []

    allowed_instructors = events_data["allowed_instructors"]
    allowed_rooms = events_data["allowed_rooms"]
    allowed_starts = events_data["allowed_starts"]

    import random

    import numpy as np

    random.seed(42)  # For reproducible results

    for i in range(n_events):
        event = events[i]

        if allowed_instructors[i]:
            instructor_assignments.append(random.choice(allowed_instructors[i]))
        else:
            instructor_assignments.append(0)

        if allowed_rooms[i]:
            room_assignments.append(random.choice(allowed_rooms[i]))
        else:
            room_assignments.append(0)

        if allowed_starts[i]:
            start_assignments.append(random.choice(allowed_starts[i]))
        else:
            start_assignments.append(0)

        duration_assignments.append(event["num_quanta"])
        group_masks.append(
            int(event["groups_mask"]) if event["groups_mask"] < 2**63 else 0
        )  # Handle large integers

    return (
        np.array(start_assignments),
        np.array(duration_assignments),
        np.array(room_assignments),
        np.array(instructor_assignments),
        np.array(group_masks, dtype=np.int64),
    )  # Use int64 for large integers


@pytest.mark.skip(
    reason="Depends on stale events_with_domains.pkl; schema changed (groups_mask → group_ids)"
)
def test_fast_evaluator():
    """Test just the fast evaluator functionality."""
    print("Testing fast evaluator functionality...")

    # Load events data
    try:
        with open(".cache/events_with_domains.pkl", "rb") as f:
            events_data = pickle.load(f)
        print(f"Loaded events data with {events_data['metadata']['n_events']} events")
    except FileNotFoundError:
        print("Error: events_with_domains.pkl not found. Run build_events.py first.")
        return

    # Create test assignments
    print("Creating test assignments...")
    start_assign, duration_assign, room_assign, instructor_assign, group_masks = (
        create_dummy_assignments(events_data)
    )
    print(f"Created assignments for {len(instructor_assign)} events")

    # Test fast evaluator multiple times
    print("\\nTesting fast evaluator performance and consistency...")
    results = []
    times = []

    for _i in range(10):
        start_time = time.time()
        result = fast_conflict_evaluator(
            start_assign,
            duration_assign,
            room_assign,
            instructor_assign,
            group_masks,
            events_data,
        )
        times.append(time.time() - start_time)
        results.append(result)

    # Check consistency
    first_result = results[0]
    consistent = all(
        r[0] == first_result[0]
        and r[1] == first_result[1]
        and r[2] == first_result[2]
        and abs(r[3] - first_result[3]) < 1e-6
        for r in results
    )

    print(f"Consistency test: {'PASS' if consistent else 'FAIL'}")
    print(f"Average time: {sum(times) / len(times) * 1000:.2f}ms")
    print(
        f"Result: room_conflicts={first_result[0]}, instructor_conflicts={first_result[1]}, group_conflicts={first_result[2]}, soft_penalty={first_result[3]:.2f}"
    )

    # Test different random assignments
    print("\\nTesting with 5 different random assignments...")
    different_results = []

    for seed in [42, 123, 456, 789, 999]:
        import random

        random.seed(seed)
        start_a, duration_a, room_a, instructor_a, group_m = create_dummy_assignments(
            events_data
        )
        result = fast_conflict_evaluator(
            start_a, duration_a, room_a, instructor_a, group_m, events_data
        )
        different_results.append((seed, result))
        print(
            f"Seed {seed}: hard={result[0] + result[1] + result[2]}, soft={result[3]:.2f}"
        )

    # Verify we get different results for different assignments (sanity check)
    unique_hard_totals = {r[1][0] + r[1][1] + r[1][2] for r in different_results}
    print(
        f"\\nSanity check: {len(unique_hard_totals)} unique hard constraint totals (should be > 1)"
    )

    print("\\n=== FAST EVALUATOR TEST COMPLETED ===")
    if consistent and len(unique_hard_totals) > 1:
        print(" Fast evaluator working correctly")
    else:
        print(" Fast evaluator has issues")


if __name__ == "__main__":
    test_fast_evaluator()
