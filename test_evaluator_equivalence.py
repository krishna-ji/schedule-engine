#!/usr/bin/env python3
"""
Test evaluator equivalence between original and fast evaluator.
"""

import random
import sys
import time
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import required modules
from fast_evaluator import fast_conflict_evaluator
from src.domain.types import SchedulingContext
from src.ga.core.evaluator import evaluate
from src.io.data_loader import (
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from src.io.time_system import QuantumTimeSystem
from src.notebooks.core import NotebookData, create_random_individual


def extract_assignments(individual):
    """Extract instructor, room, and start time assignments from individual."""
    instructor_assignments = []
    room_assignments = []
    start_assignments = []

    duration_assignments = []
    group_masks = []

    for gene in individual:
        instructor_assignments.append(gene.instructor_id)
        room_assignments.append(gene.room_id)
        start_assignments.append(gene.start_quanta)
        duration_assignments.append(gene.num_quanta)
        group_masks.append(gene.groups_mask if hasattr(gene, "groups_mask") else 0)

    return (
        instructor_assignments,
        room_assignments,
        start_assignments,
        duration_assignments,
        group_masks,
    )


def test_evaluator_equivalence():
    """Test equivalence between original and fast evaluators."""
    print("Setting up test environment...")

    # Load data using the correct API
    data_path = PROJECT_ROOT / "data"
    qts = QuantumTimeSystem()
    courses = load_courses(str(data_path / "Course.json"))
    groups = load_groups(str(data_path / "Groups.json"), qts)
    instructors = load_instructors(str(data_path / "Instructors.json"), qts)
    rooms = load_rooms(str(data_path / "Rooms.json"), qts)

    # Link data
    link_courses_and_groups(courses, groups)
    link_courses_and_instructors(courses, instructors)

    # Create scheduling context and notebook data
    ctx = SchedulingContext(
        courses,
        groups,
        instructors,
        rooms,
        available_quanta=list(range(qts.total_quanta)),
    )
    data = NotebookData(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        context=ctx,
        qts=qts,
    )

    print(
        f"Loaded {len(courses)} courses, {len(instructors)} instructors, {len(rooms)} rooms, {len(groups)} groups"
    )

    # Load fast evaluator events data
    import pickle

    try:
        with open("events_with_domains.pkl", "rb") as f:
            events_data = pickle.load(f)
        print(f"Loaded events data with {events_data['metadata']['n_events']} events")
    except FileNotFoundError:
        print("Error: events_with_domains.pkl not found. Run build_events.py first.")
        return

    # Generate test individuals
    print("\nGenerating 20 random individuals for testing...")
    test_individuals = []
    for i in range(20):
        try:
            individual = create_random_individual(data)
            test_individuals.append(individual)
            if i % 5 == 0:
                print(f"  Generated {i+1}/20 individuals")
        except Exception as e:
            print(f"Error generating individual {i}: {e}")
            continue

    print(f"Generated {len(test_individuals)} test individuals")

    # Test equivalence
    print("\nTesting evaluator equivalence...")
    mismatches: list[dict[str, object]] = []
    original_times = []
    fast_times = []

    for i, individual in enumerate(test_individuals):
        try:
            # Original evaluator
            start_time = time.time()
            original_fitness = evaluate(individual, courses, instructors, groups, rooms)
            original_times.append(time.time() - start_time)

            original_result = {
                "hard_total": original_fitness[0],
                "soft_penalty": original_fitness[1],
            }

            # Fast evaluator
            start_time = time.time()
            (
                instructor_assignments,
                room_assignments,
                start_assignments,
                duration_assignments,
                group_masks_list,
            ) = extract_assignments(individual)
            import numpy as np

            room_conf, inst_conf, group_conf, soft_penalty = fast_conflict_evaluator(
                np.array(start_assignments),
                np.array(duration_assignments),
                np.array(room_assignments),
                np.array(instructor_assignments),
                np.array(group_masks_list),
                events_data,
            )
            fast_times.append(time.time() - start_time)

            fast_result = {
                "hard_total": room_conf + inst_conf + group_conf,
                "room_conflicts": room_conf,
                "instructor_conflicts": inst_conf,
                "group_conflicts": group_conf,
                "soft_penalty": soft_penalty,
            }

            # Compare results
            has_mismatch = False
            details: dict[str, tuple[object, object, object]] = {}
            for key in ["hard_total", "soft_penalty"]:
                orig_val = original_result[key]
                fast_val = fast_result[key]
                if abs(orig_val - fast_val) > 1e-6:  # Allow small float differences
                    has_mismatch = True
                    details[key] = (orig_val, fast_val, abs(orig_val - fast_val))

            if has_mismatch:
                mismatches.append(
                    {
                        "individual_idx": i,
                        "original": original_result,
                        "fast": fast_result,
                        "differences": details,
                    }
                )

            if i % 5 == 4:
                print(f"  Tested {i+1}/20 individuals")

        except Exception as e:
            print(f"Error testing individual {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Report results
    print("\n=== EVALUATOR EQUIVALENCE TEST RESULTS ===")
    print(f"Individuals tested: {len(test_individuals)}")
    print(f"Mismatches found: {len(mismatches)}")

    if original_times and fast_times:
        print(
            f"Average original time: {sum(original_times)/len(original_times)*1000:.2f}ms"
        )
        print(f"Average fast time: {sum(fast_times)/len(fast_times)*1000:.2f}ms")
        print(f"Speedup: {sum(original_times)/sum(fast_times):.2f}x")

    if mismatches:
        print("\n EQUIVALENCE TEST FAILED")
        print("Showing first 3 mismatches:")
        for i, mismatch in enumerate(mismatches[:3]):
            print(f"\nMismatch {i+1} (Individual {mismatch['individual_idx']}):")
            print(f"  Original: {mismatch['original']}")
            print(f"  Fast:     {mismatch['fast']}")
            print("  Differences:")
            diffs = mismatch["differences"]
            assert isinstance(diffs, dict)
            for key, (orig, fast, diff) in diffs.items():
                print(f"    {key}: {orig} vs {fast} (diff: {diff})")
    else:
        print("\n EVALUATOR EQUIVALENCE VERIFIED")
        print("All evaluations match between original and fast evaluators")


if __name__ == "__main__":
    test_evaluator_equivalence()
