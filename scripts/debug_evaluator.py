#!/usr/bin/env python3
"""
Debug remaining differences between evaluators with detailed comparison.
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def debug_single_individual():
    """Debug a single individual to understand differences."""
    print("=== DEBUGGING SINGLE INDIVIDUAL ===")

    # Load context data
    print("Loading data...")
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

    # Load events data
    with open("events_with_domains.pkl", "rb") as f:
        events_data = pickle.load(f)
    events = events_data["events"]

    print(f"Loaded {len(events)} events")

    # Create a simple test: assign first valid choice to each event
    individual = []
    assignments: dict[str, list] = {
        "instructor": [],
        "room": [],
        "start": [],
        "duration": [],
        "groups_mask": [],
    }

    allowed_instructors = events_data["allowed_instructors"]
    allowed_rooms = events_data["allowed_rooms"]
    allowed_starts = events_data["allowed_starts"]
    instructor_to_idx = events_data["instructor_to_idx"]
    room_to_idx = events_data["room_to_idx"]

    # Reverse mappings
    idx_to_instructor = {idx: inst_id for inst_id, idx in instructor_to_idx.items()}
    idx_to_room = {idx: room_id for room_id, idx in room_to_idx.items()}

    print("Creating test individual...")
    for i, event in enumerate(events[:10]):  # Just test first 10 events
        # Get first valid assignments
        instructor_idx = allowed_instructors[i][0] if allowed_instructors[i] else 0
        room_idx = allowed_rooms[i][0] if allowed_rooms[i] else 0
        start_idx = allowed_starts[i][0] if allowed_starts[i] else 0

        instructor_id = idx_to_instructor.get(instructor_idx, "I001")
        room_id = idx_to_room.get(room_idx, "R001")

        # Create SessionGene
        gene = SessionGene(
            course_id=event["course_id"],
            course_type=event["course_type"],
            group_ids=event["group_ids"],
            instructor_id=instructor_id,
            room_id=room_id,
            start_quanta=start_idx,
            num_quanta=event["num_quanta"],
        )
        individual.append(gene)

        # Store for fast evaluator
        assignments["instructor"].append(instructor_idx)
        assignments["room"].append(room_idx)
        assignments["start"].append(start_idx)
        assignments["duration"].append(event["num_quanta"])
        assignments["groups_mask"].append(
            event["groups_mask"] if event["groups_mask"] < 2**63 else 0
        )

    print(f"Created individual with {len(individual)} genes")

    # Evaluate with original
    print("\\nEvaluating with original Timetable...")
    timetable = Timetable(individual, context)
    orig_group = timetable.count_group_violations()
    orig_instructor = timetable.count_instructor_violations()
    orig_room = timetable.count_room_violations()

    print(
        f"Original: group={orig_group}, instructor={orig_instructor}, room={orig_room}"
    )

    # Print detailed group occupancy from original
    print("\\nOriginal group occupancy analysis:")
    group_occ_orig = timetable._group_occ
    print(f"Group occupancy entries: {len(group_occ_orig)}")
    violations_orig = [
        (key, len(idxs)) for key, idxs in group_occ_orig.items() if len(idxs) > 1
    ]
    print(f"Violations: {len(violations_orig)}")
    for i, (key, count) in enumerate(violations_orig[:5]):
        print(f"  {key}: {count} events")

    # Evaluate with fast
    print("\\nEvaluating with fast evaluator...")
    start_arr = np.array(assignments["start"])
    duration_arr = np.array(assignments["duration"])
    room_arr = np.array(assignments["room"])
    instructor_arr = np.array(assignments["instructor"])
    groups_mask_arr = np.array(assignments["groups_mask"], dtype=np.int64)

    # Truncate events_data to match
    events_data_truncated = {
        "events": events_data["events"][:10],
        "allowed_instructors": events_data["allowed_instructors"][:10],
        "allowed_rooms": events_data["allowed_rooms"][:10],
        "allowed_starts": events_data["allowed_starts"][:10],
    }

    fast_room, fast_instructor, fast_group, fast_soft = fast_conflict_evaluator(
        start_arr,
        duration_arr,
        room_arr,
        instructor_arr,
        groups_mask_arr,
        events_data_truncated,
    )

    print(f"Fast: group={fast_group}, instructor={fast_instructor}, room={fast_room}")

    print("\\nDifferences:")
    print(f"  Group: {orig_group} vs {fast_group} (diff: {fast_group - orig_group})")
    print(
        f"  Instructor: {orig_instructor} vs {fast_instructor} (diff: {fast_instructor - orig_instructor})"
    )
    print(f"  Room: {orig_room} vs {fast_room} (diff: {fast_room - orig_room})")

    # Detailed conflict analysis for troubleshooting
    print("\\nDetailed conflict analysis:")
    print("Individual summary:")
    for i, gene in enumerate(individual):
        print(
            f"  Gene {i}: {gene.course_id} {gene.course_type} groups={gene.group_ids} room={gene.room_id} instructor={gene.instructor_id} start={gene.start_quanta} dur={gene.num_quanta}"
        )


if __name__ == "__main__":
    debug_single_individual()
