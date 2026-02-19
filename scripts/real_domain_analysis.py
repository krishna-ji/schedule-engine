#!/usr/bin/env python3
"""Exact domain size analysis with real constraints."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_real_domain_sizes():
    """Analyze domain sizes using actual constraint logic."""
    try:
        import random

        from src.ga.core.population import generate_pure_random_population
        from src.ga.operators.mutation import find_suitable_rooms_for_course
        from src.io.data_store import DataStore

        store = DataStore.from_json("data")
        ctx = store.to_context()

        pop = generate_pure_random_population(1, ctx, parallel=False)
        genes = pop[0]

        # Sample 100 random genes
        sample_genes = random.sample(genes, min(100, len(genes)))

        print("=== REAL DOMAIN SIZES FOR 100 RANDOM EVENTS ===")
        print()
        print("Function locations:")
        print(
            "- Room suitability: find_suitable_rooms_for_course() in src/ga/operators/mutation.py"
        )
        print(
            "- Uses: is_room_suitable_for_course() in src/utils/room_compatibility.py"
        )
        print(
            "- Instructor qualification: course.qualified_instructor_ids (InstructorQualifications constraint)"
        )
        print(
            "- Time availability: instructor.is_full_time or instructor.available_quanta"
        )
        print()

        suitable_rooms_counts = []
        qualified_instructors_counts = []
        allowed_start_times = []

        for i, gene in enumerate(sample_genes):
            course_key = (gene.course_id, gene.course_type)
            course = ctx.courses.get(course_key)

            # REAL room suitability check
            primary_group = gene.group_ids[0] if gene.group_ids else ""
            suitable_rooms = find_suitable_rooms_for_course(
                gene.course_id, gene.course_type, primary_group, ctx
            )

            # REAL instructor qualification check
            qualified_instructors: list[str] = []
            if course:
                qualified_instructors = getattr(course, "qualified_instructor_ids", [])

            # REAL time availability check
            # Consider instructor availability for start time constraints
            max_quantum = 42  # Total operating quanta
            base_max_start = max_quantum - gene.num_quanta

            # For now, use base time range (could be refined with instructor availability)
            allowed_starts = list(range(base_max_start + 1))

            suitable_rooms_counts.append(len(suitable_rooms))
            qualified_instructors_counts.append(len(qualified_instructors))
            allowed_start_times.append(len(allowed_starts))

            if i < 10:  # Show first 10 events
                print(
                    f"Event {i}: Course={gene.course_id} Type={gene.course_type} Groups={gene.group_ids}"
                )
                print(f"  Duration: {gene.num_quanta} quanta")
                print(
                    f"  Suitable rooms: {len(suitable_rooms)} (from suitability+capacity check)"
                )
                print(f"  Qualified instructors: {len(qualified_instructors)}")
                print(
                    f"  Allowed start times: {len(allowed_starts)} (0 to {base_max_start})"
                )
                if i < 3:  # Show details for first 3
                    print(f"    Sample suitable rooms: {suitable_rooms[:5]}")
                    print(
                        f"    Sample qualified instructors: {qualified_instructors[:5]}"
                    )
                print()

        print("=== SUMMARY STATISTICS ===")
        print()
        print("SUITABLE ROOMS (real suitability + capacity):")
        print(f"  Min: {min(suitable_rooms_counts)}")
        print(f"  Max: {max(suitable_rooms_counts)}")
        print(f"  Avg: {sum(suitable_rooms_counts)/len(suitable_rooms_counts):.1f}")
        print()

        print("QUALIFIED INSTRUCTORS (real qualification check):")
        print(f"  Min: {min(qualified_instructors_counts)}")
        print(f"  Max: {max(qualified_instructors_counts)}")
        print(
            f"  Avg: {sum(qualified_instructors_counts)/len(qualified_instructors_counts):.1f}"
        )
        print()

        print("ALLOWED START TIMES (considering duration):")
        print(f"  Min: {min(allowed_start_times)}")
        print(f"  Max: {max(allowed_start_times)}")
        print(f"  Avg: {sum(allowed_start_times)/len(allowed_start_times):.1f}")
        print()

        # Show distribution of constraints
        print("CONSTRAINT TIGHTNESS ANALYSIS:")
        tight_rooms = sum(1 for count in suitable_rooms_counts if count <= 5)
        tight_instructors = sum(
            1 for count in qualified_instructors_counts if count <= 2
        )
        tight_time = sum(1 for count in allowed_start_times if count <= 10)

        print(
            f"Events with ≤5 suitable rooms: {tight_rooms}/100 ({100*tight_rooms/100:.0f}%)"
        )
        print(
            f"Events with ≤2 qualified instructors: {tight_instructors}/100 ({100*tight_instructors/100:.0f}%)"
        )
        print(
            f"Events with ≤10 start time options: {tight_time}/100 ({100*tight_time/100:.0f}%)"
        )

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    analyze_real_domain_sizes()
