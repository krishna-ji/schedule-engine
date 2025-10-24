"""
Diagnose pickling overhead in multiprocessing.

This measures the size and serialization time of objects
being passed to worker processes.
"""

import pickle
import time
from src.encoder.input_encoder import (
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
    link_courses_and_groups,
    link_courses_and_instructors,
)
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.core.types import SchedulingContext
from functools import partial
from src.ga.evaluator.fitness import evaluate


def main():
    print("=" * 70)
    print("PICKLING OVERHEAD DIAGNOSIS")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    qts = QuantumTimeSystem()
    all_courses = load_courses("data/Course.json")
    groups = load_groups("data/Groups.json", qts)
    instructors = load_instructors("data/Instructors.json", qts)
    rooms = load_rooms("data/Rooms.json", qts)

    enrolled = set()
    for g in groups.values():
        enrolled.update(g.enrolled_courses)
    courses = {k: v for k, v in all_courses.items() if k[0] in enrolled}

    link_courses_and_groups(courses, groups)
    link_courses_and_instructors(courses, instructors)

    context = SchedulingContext(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        available_quanta=qts.get_all_operating_quanta(),
    )

    print(f"✓ Loaded {len(courses)} courses, {len(groups)} groups")
    print()

    # Test 1: Measure context object sizes
    print("Measuring context object sizes...")
    print("-" * 70)

    courses_size = len(pickle.dumps(courses))
    groups_size = len(pickle.dumps(groups))
    instructors_size = len(pickle.dumps(instructors))
    rooms_size = len(pickle.dumps(rooms))
    quanta_size = len(pickle.dumps(context.available_quanta))

    print(f"courses:     {courses_size:>10,} bytes ({courses_size/1024:>8.1f} KB)")
    print(f"groups:      {groups_size:>10,} bytes ({groups_size/1024:>8.1f} KB)")
    print(
        f"instructors: {instructors_size:>10,} bytes ({instructors_size/1024:>8.1f} KB)"
    )
    print(f"rooms:       {rooms_size:>10,} bytes ({rooms_size/1024:>8.1f} KB)")
    print(f"quanta:      {quanta_size:>10,} bytes ({quanta_size/1024:>8.1f} KB)")
    total_size = (
        courses_size + groups_size + instructors_size + rooms_size + quanta_size
    )
    print(
        f"TOTAL:       {total_size:>10,} bytes ({total_size/1024:>8.1f} KB) ({total_size/(1024*1024):>8.2f} MB)"
    )
    print()

    # Test 2: Measure partial function size
    print("Measuring partial function size...")
    print("-" * 70)

    eval_func = partial(
        evaluate, courses=courses, instructors=instructors, groups=groups, rooms=rooms
    )

    start = time.time()
    pickled_func = pickle.dumps(eval_func)
    pickle_time = time.time() - start

    print(
        f"Partial function: {len(pickled_func):>10,} bytes ({len(pickled_func)/1024:>8.1f} KB) ({len(pickled_func)/(1024*1024):>8.2f} MB)"
    )
    print(f"Pickling time:    {pickle_time*1000:>10.1f} ms")
    print()

    # Test 3: Estimate overhead per worker
    print("Estimating overhead per pool.map() call...")
    print("-" * 70)

    # With spawn on Windows, each worker process gets a pickled copy
    num_workers = 8
    overhead_per_call = pickle_time * num_workers

    print(f"Workers:          {num_workers}")
    print(
        f"Overhead/call:    {overhead_per_call*1000:>10.1f} ms  ({overhead_per_call:>6.3f} s)"
    )
    print()

    # Test 4: Compare to evaluation time
    print("Comparing to actual evaluation time...")
    print("-" * 70)

    from src.ga.population import generate_course_group_aware_population

    population = generate_course_group_aware_population(n=10, context=context)

    start = time.time()
    for ind in population[:5]:
        evaluate(
            ind, courses=courses, instructors=instructors, groups=groups, rooms=rooms
        )
    eval_time_per_ind = (time.time() - start) / 5

    print(f"Avg evaluation time:  {eval_time_per_ind*1000:>10.1f} ms per individual")
    print(f"Pickling overhead:    {pickle_time*1000:>10.1f} ms (one-time per call)")
    print(f"Overhead ratio:       {pickle_time/eval_time_per_ind:>10.1f}x")
    print()

    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if pickle_time > eval_time_per_ind:
        print("✗ PROBLEM: Pickling overhead exceeds evaluation time!")
        print(f"  Pickling:   {pickle_time*1000:.1f}ms")
        print(f"  Evaluation: {eval_time_per_ind*1000:.1f}ms")
        print(f"  Ratio:      {pickle_time/eval_time_per_ind:.1f}x")
        print()
        print("This explains why multiprocessing is slower:")
        print("- Every pool.map() call must serialize the context")
        print("- On Windows (spawn), this happens for EACH worker")
        print("- Total overhead dominates the parallel benefit")
        print()
        print("SOLUTIONS:")
        print("1. Use shared memory (multiprocessing.Manager)")
        print("2. Initialize workers once with context (initializer=)")
        print("3. Use process-local cache")
        print("4. Switch to threading (if GIL not an issue)")
    else:
        print("✓ Pickling overhead is acceptable")
        print(f"  Overhead: {pickle_time*1000:.1f}ms")
        print(f"  Evaluation: {eval_time_per_ind*1000:.1f}ms")

    print("=" * 70)


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout, redirect_stderr

    f_out = io.StringIO()
    f_err = io.StringIO()

    # Suppress warnings during data loading
    with redirect_stdout(f_out), redirect_stderr(f_err):
        from src.encoder.input_encoder import (
            load_courses,
            load_groups,
            load_instructors,
            load_rooms,
            link_courses_and_groups,
            link_courses_and_instructors,
        )
        from src.ga.population import generate_course_group_aware_population

    main()
