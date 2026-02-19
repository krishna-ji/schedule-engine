#!/usr/bin/env python3
"""Simple individual inspection script - no multiprocessing."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def inspect_individual():
    """Create and inspect a single individual."""
    try:
        # Import required modules
        from src.ga.core.evaluator import evaluate
        from src.ga.core.population import generate_pure_random_population
        from src.io.data_store import DataStore
        from src.io.decoder import decode_individual
        from src.io.time_system import QuantumTimeSystem

        print("Loading data...")
        data_store = DataStore.from_json(PROJECT_ROOT / "data")
        courses, instructors, groups, rooms = (
            data_store.courses,
            data_store.instructors,
            data_store.groups,
            data_store.rooms,
        )
        context = data_store.to_context()

        print("Data loaded:")
        print(f"  Courses: {len(courses)}")
        print(f"  Instructors: {len(instructors)}")
        print(f"  Groups: {len(groups)}")
        print(f"  Rooms: {len(rooms)}")

        print("\nGenerating individual...")
        population = generate_pure_random_population(1, context, parallel=False)
        individual = population[0]
        print(f"Individual size (genes): {len(individual)}")

        # Print first few genes
        print("\nFirst 5 genes:")
        for i, gene in enumerate(individual[:5]):
            print(
                f"  Gene {i}: Course={gene.course_id}, Groups={gene.group_ids}, "
                f"Instructor={gene.instructor_id}, Room={gene.room_id}, "
                f"Start={gene.start_quanta}, Num={gene.num_quanta}"
            )

        # Quick evaluation test
        print("\nSingle evaluation test...")
        hard, soft = evaluate(individual, courses, instructors, groups, rooms)
        print(f"Fitness: Hard={hard}, Soft={soft}")

        # Print schedule snippet for first few genes
        print("\nDecoded schedule (first 5 sessions):")
        decode_individual(individual, courses, instructors, groups, rooms)
        qts = QuantumTimeSystem()
        for i, gene in enumerate(individual[:5]):
            day, start_time = qts.quanta_to_time(gene.start_quanta)
            _, end_time = qts.quanta_to_time(gene.start_quanta + gene.num_quanta - 1)
            time_str = f"{day} {start_time}-{end_time}"
            print(
                f"  {gene.course_id} - {gene.group_ids} - {gene.instructor_id} - {gene.room_id} - {time_str}"
            )

    except Exception as e:
        print(f"Error during inspection: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    inspect_individual()
