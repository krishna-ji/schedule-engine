#!/usr/bin/env python3
"""Quick timing test for evaluation performance."""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_timing_test():
    """Run a quick timing test on individual creation and evaluation."""
    try:
        # Import required modules
        from src.ga.core.creator_registry import get_creator
        from src.ga.core.evaluator import evaluate
        from src.generation.initial import generate_random_individual
        from src.io.loaders import DataManager

        print("Loading data...")
        data_manager = DataManager(PROJECT_ROOT / "data")
        courses, instructors, groups, rooms = data_manager.load_all()

        print("Data loaded:")
        print(f"  Courses: {len(courses)}")
        print(f"  Instructors: {len(instructors)}")
        print(f"  Groups: {len(groups)}")
        print(f"  Rooms: {len(rooms)}")

        # Setup DEAP
        _ = get_creator()

        print("\nGenerating individuals and timing evaluations...")

        # Time individual generation
        gen_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            individual = generate_random_individual(courses, instructors, groups, rooms)
            gen_times.append(time.perf_counter() - t0)

        print(f"Individual generation time: {sum(gen_times)/len(gen_times):.4f}s avg")
        print(f"Individual size (genes): {len(individual) if individual else 'N/A'}")

        # Time evaluations
        if individual:
            eval_times = []
            for _ in range(50):
                # Generate a new individual each time to avoid caching effects
                test_ind = generate_random_individual(
                    courses, instructors, groups, rooms
                )
                t0 = time.perf_counter()
                hard, soft = evaluate(test_ind, courses, instructors, groups, rooms)
                eval_times.append(time.perf_counter() - t0)

            avg_eval_time = sum(eval_times) / len(eval_times)
            print(f"Evaluation time: {avg_eval_time:.4f}s avg (50 runs)")
            print(f"Example fitness: Hard={hard}, Soft={soft}")
            print(f"Estimated evals per second: {1/avg_eval_time:.1f}")

            # Timing for 200 evaluations as requested
            print("\nTiming 200 evaluations...")
            t0 = time.perf_counter()
            for _ in range(200):
                test_ind = generate_random_individual(
                    courses, instructors, groups, rooms
                )
                evaluate(test_ind, courses, instructors, groups, rooms)
            total_time = time.perf_counter() - t0
            print(f"200 evals seconds: {total_time:.4f}")

    except Exception as e:
        print(f"Error during timing test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_timing_test()
