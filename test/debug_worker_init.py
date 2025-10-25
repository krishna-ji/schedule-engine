"""
Debug multiprocessing worker initialization.
"""

import sys
import os
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.ga_scheduler import _worker_init, _worker_evaluate

# Create dummy context
dummy_context = {
    "courses": {"C1": "course1"},
    "instructors": {"I1": "inst1"},
    "groups": {"G1": "group1"},
    "rooms": {"R1": "room1"},
}


def test_worker():
    """Test if worker initialization works."""
    print("Testing worker initialization...")

    # Try to initialize worker
    _worker_init(
        dummy_context["courses"],
        dummy_context["instructors"],
        dummy_context["groups"],
        dummy_context["rooms"],
        seed=42,
    )

    print("  ✓ Worker initialized successfully")

    # Check if context is set
    from src.core.ga_scheduler import _WORKER_CONTEXT

    if _WORKER_CONTEXT is not None:
        print(f"  ✓ _WORKER_CONTEXT is set: {list(_WORKER_CONTEXT.keys())}")
    else:
        print("  ✗ _WORKER_CONTEXT is None!")
        return False

    # Check if creator types exist
    from deap import creator

    if hasattr(creator, "FitnessMulti"):
        print("  ✓ creator.FitnessMulti exists")
    else:
        print("  ✗ creator.FitnessMulti missing!")
        return False

    if hasattr(creator, "Individual"):
        print("  ✓ creator.Individual exists")
    else:
        print("  ✗ creator.Individual missing!")
        return False

    return True


def worker_test_function(x):
    """Function to run in worker."""
    from src.core.ga_scheduler import _WORKER_CONTEXT

    if _WORKER_CONTEXT is None:
        return "CONTEXT_IS_NONE"
    return f"OK:{len(_WORKER_CONTEXT)}"


if __name__ == "__main__":
    print("=" * 80)
    print("WORKER INITIALIZATION DEBUG")
    print("=" * 80)
    print()

    # Test 1: Direct call
    print("Test 1: Direct worker_init() call")
    print("-" * 40)
    success = test_worker()
    print()

    # Test 2: In multiprocessing pool
    print("Test 2: Worker init in Pool")
    print("-" * 40)

    pool = multiprocessing.Pool(
        processes=2,
        initializer=_worker_init,
        initargs=(
            dummy_context["courses"],
            dummy_context["instructors"],
            dummy_context["groups"],
            dummy_context["rooms"],
            42,
        ),
    )

    results = pool.map(worker_test_function, range(4))
    pool.close()
    pool.join()

    print(f"  Results from workers: {results}")

    if all("OK" in r for r in results):
        print("  ✓ All workers have context!")
    else:
        print("  ✗ Some workers don't have context!")
        for i, r in enumerate(results):
            if "CONTEXT_IS_NONE" in r:
                print(f"    Worker {i}: CONTEXT IS NONE")

    print()
    print("=" * 80)
