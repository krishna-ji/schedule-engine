#!/usr/bin/env python3
"""
Equivalence test between original Timetable evaluator and fast evaluator.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.gene import SessionGene
from src.domain.timetable import Timetable
from src.domain.types import SchedulingContext
from src.io.data_loader import (
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from src.io.time_system import QuantumTimeSystem
from src.pipeline.fast_evaluator import fast_conflict_evaluator


def create_test_individual(events_data, seed=42):
    """Create a test individual from events data."""
    import random

    random.seed(seed)
    np.random.seed(seed)

    events = events_data["events"]
    allowed_instructors = events_data["allowed_instructors"]
    allowed_rooms = events_data["allowed_rooms"]
    allowed_starts = events_data["allowed_starts"]

    individual = []
    assignments: dict[str, list] = {
        "instructor": [],
        "room": [],
        "start": [],
        "duration": [],
        "groups_mask": [],
    }

    for i, event in enumerate(events):
        # Create random valid assignments
        if allowed_instructors[i]:
            instructor_idx = random.choice(allowed_instructors[i])
        else:
            instructor_idx = 0

        if allowed_rooms[i]:
            room_idx = random.choice(allowed_rooms[i])
        else:
            room_idx = 0

        if allowed_starts[i]:
            start_idx = random.choice(allowed_starts[i])
        else:
            start_idx = 0

        # Map back to actual IDs using metadata
        instructor_to_idx = events_data["instructor_to_idx"]
        room_to_idx = events_data["room_to_idx"]

        # Reverse mapping to get actual IDs
        instructor_id = None
        room_id = None
        for actual_id, idx in instructor_to_idx.items():
            if idx == instructor_idx:
                instructor_id = actual_id
                break
        for actual_id, idx in room_to_idx.items():
            if idx == room_idx:
                room_id = actual_id
                break

        # Create SessionGene
        gene = SessionGene(
            course_id=event["course_id"],
            course_type=event["course_type"],
            group_ids=event["group_ids"],
            instructor_id=instructor_id if instructor_id else "I001",  # Fallback
            room_id=room_id if room_id else "R001",  # Fallback
            start_quanta=start_idx,
            num_quanta=event["num_quanta"],
        )
        individual.append(gene)

        # Store assignments for fast evaluator
        assignments["instructor"].append(instructor_idx)
        assignments["room"].append(room_idx)
        assignments["start"].append(start_idx)
        assignments["duration"].append(event["num_quanta"])
        assignments["groups_mask"].append(
            int(event["groups_mask"]) if event["groups_mask"] < 2**63 else 0
        )

    return individual, assignments


def evaluate_with_original(individual, context):
    """Evaluate using original Timetable-based evaluator."""
    timetable = Timetable(individual, context)

    group_violations = timetable.count_group_violations()
    instructor_violations = timetable.count_instructor_violations()
    room_violations = timetable.count_room_violations()

    # Total hard violations
    hard_total = group_violations + instructor_violations + room_violations

    return {
        "hard_total": hard_total,
        "group_violations": group_violations,
        "instructor_violations": instructor_violations,
        "room_violations": room_violations,
        "soft_violations": 0.0,  # Not implemented in this comparison
    }


def evaluate_with_fast(assignments, events_data):
    """Evaluate using fast evaluator."""
    start_arr = np.array(assignments["start"])
    duration_arr = np.array(assignments["duration"])
    room_arr = np.array(assignments["room"])
    instructor_arr = np.array(assignments["instructor"])
    groups_mask_arr = np.array(assignments["groups_mask"], dtype=np.int64)

    room_conf, inst_conf, group_conf, soft_penalty = fast_conflict_evaluator(
        start_arr, duration_arr, room_arr, instructor_arr, groups_mask_arr, events_data
    )

    return {
        "hard_total": room_conf + inst_conf + group_conf,
        "group_violations": group_conf,
        "instructor_violations": inst_conf,
        "room_violations": room_conf,
        "soft_violations": soft_penalty,
    }


def test_equivalence():
    """Run equivalence test between original and fast evaluators."""
    print("=== EVALUATOR EQUIVALENCE TEST ===")

    # Load context data
    print("Loading scheduling context...")
    data_path = PROJECT_ROOT / "data"
    qts = QuantumTimeSystem()

    courses, skipped_courses = load_courses(str(data_path / "Course.json"))
    groups = load_groups(str(data_path / "Groups.json"), qts)
    instructors = load_instructors(str(data_path / "Instructors.json"), qts)
    rooms = load_rooms(str(data_path / "Rooms.json"), qts)

    link_courses_and_groups(courses, groups, skipped_courses=skipped_courses)
    link_courses_and_instructors(courses, instructors)

    context = SchedulingContext(
        courses, groups, instructors, rooms, available_quanta=list(range(42))
    )
    print(
        f"Loaded {len(courses)} courses, {len(groups)} groups, {len(instructors)} instructors, {len(rooms)} rooms"
    )

    # Load events data
    try:
        with open(".cache/events_with_domains.pkl", "rb") as f:
            events_data = pickle.load(f)
        print(f"Loaded {len(events_data['events'])} events")
    except FileNotFoundError:
        print("Error: events_with_domains.pkl not found. Run build_events.py first.")
        return None

    # Test on 20 random individuals
    print("\\nGenerating and testing 20 random individuals...")
    mismatches = []
    original_times = []
    fast_times = []

    for i in range(20):
        try:
            # Generate individual with different seeds
            individual, assignments = create_test_individual(events_data, seed=i * 42)

            # Original evaluator
            start_time = time.time()
            original_result = evaluate_with_original(individual, context)
            original_times.append(time.time() - start_time)

            # Fast evaluator
            start_time = time.time()
            fast_result = evaluate_with_fast(assignments, events_data)
            fast_times.append(time.time() - start_time)

            # Compare results
            has_mismatch = False
            details = {}
            for key in [
                "hard_total",
                "group_violations",
                "instructor_violations",
                "room_violations",
            ]:
                orig_val = original_result[key]
                fast_val = fast_result[key]
                if orig_val != fast_val:
                    has_mismatch = True
                    details[key] = {
                        "original": orig_val,
                        "fast": fast_val,
                        "diff": fast_val - orig_val,
                    }

            if has_mismatch:
                mismatches.append(
                    {
                        "individual_id": i,
                        "original": original_result,
                        "fast": fast_result,
                        "details": details,
                    }
                )

            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/20 individuals")

        except Exception as e:
            print(f"  Error testing individual {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Report results
    print("\\n=== RESULTS ===")
    print("Individuals tested: 20")
    print(f"Mismatches found: {len(mismatches)}")
    print(f"Average original time: {np.mean(original_times) * 1000:.2f}ms")
    print(f"Average fast time: {np.mean(fast_times) * 1000:.2f}ms")
    print(f"Speedup: {np.mean(original_times) / np.mean(fast_times):.2f}x")

    if len(mismatches) == 0:
        print("\\n EQUIVALENCE TEST PASSED")
        print("All evaluations match between original and fast evaluators")
        return True
    print("\\n EQUIVALENCE TEST FAILED")
    print("\\nShowing first 3 mismatches:")
    for i, mismatch in enumerate(mismatches[:3]):
        print(f"\\nMismatch {i + 1} (Individual {mismatch['individual_id']}):")
        print(f"  Original: {mismatch['original']}")
        print(f"  Fast:     {mismatch['fast']}")
        print("  Per-constraint breakdown:")
        for constraint, details in mismatch["details"].items():
            print(
                f"    {constraint}: {details['original']} → {details['fast']} (diff: {details['diff']})"
            )

    return False


if __name__ == "__main__":
    test_equivalence()
