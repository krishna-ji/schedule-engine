"""
Comprehensive Multiprocessing Bug Hunt
======================================
Systematic search for ALL remaining multiprocessing issues beyond:
1. Pickling overhead (KNOWN)
2. Random seed propagation (KNOWN)

This script tests:
- Pool lifecycle issues (resource leaks, hanging)
- Exception handling and error propagation
- Data corruption in parallel evaluation
- Race conditions and synchronization
- Windows-specific spawn issues
- DEAP toolbox registration bugs
- Creator type pickling issues
- Memory leaks
- Zombie processes
- Deadlocks
"""

import sys
import os
import time
import random
import multiprocessing
import traceback
from deap import base, creator, tools

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=" * 80)
print("COMPREHENSIVE MULTIPROCESSING BUG HUNT")
print("=" * 80)
print()

# ============================================================================
# Issue 1: Pool Resource Leaks
# ============================================================================
print("Issue 1: POOL RESOURCE LEAKS")
print("-" * 40)


def check_pool_resource_leak():
    """Check if pools are properly cleaned up."""
    import psutil

    initial_process_count = len(psutil.Process().children())

    # Create and destroy pools repeatedly
    for i in range(5):
        pool = multiprocessing.Pool(processes=2)
        pool.close()
        pool.join()
        time.sleep(0.1)  # Give OS time to clean up

    final_process_count = len(psutil.Process().children())

    leaked = final_process_count - initial_process_count

    if leaked > 0:
        print(f"  ✗ RESOURCE LEAK: {leaked} zombie processes detected")
        print(f"    Initial children: {initial_process_count}")
        print(f"    Final children: {final_process_count}")
        return False
    else:
        print(f"  ✓ No resource leaks (children={final_process_count})")
        return True


try:
    leak_free = check_pool_resource_leak()
except Exception as e:
    print(f"  ✗ ERROR checking resource leaks: {e}")
    leak_free = False

print()

# ============================================================================
# Issue 2: Exception Propagation from Workers
# ============================================================================
print("Issue 2: EXCEPTION PROPAGATION FROM WORKERS")
print("-" * 40)


def failing_worker(x):
    """Worker that raises exception."""
    if x == 5:
        raise ValueError(f"Intentional error at x={x}")
    return x * 2


def check_exception_propagation():
    """Check if worker exceptions properly propagate to main process."""
    try:
        pool = multiprocessing.Pool(processes=2)
        results = pool.map(failing_worker, range(10))
        pool.close()
        pool.join()
        print("  ✗ EXCEPTION NOT PROPAGATED (should have raised ValueError)")
        return False
    except ValueError as e:
        print(f"  ✓ Exception properly propagated: {e}")
        return True
    except Exception as e:
        print(f"  ✗ UNEXPECTED EXCEPTION: {e}")
        return False


exception_ok = check_exception_propagation()
print()

# ============================================================================
# Issue 3: Data Corruption in Parallel Evaluation
# ============================================================================
print("Issue 3: DATA CORRUPTION IN PARALLEL EVALUATION")
print("-" * 40)


def evaluate_with_state(individual):
    """Evaluation function that should produce consistent results."""
    # Simulate complex evaluation with state
    result = sum(individual)
    time.sleep(0.001)  # Simulate work
    return (result,)


def check_data_corruption():
    """Check if parallel evaluation produces consistent results."""
    # Setup DEAP types
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create test population
    population = [creator.Individual([i] * 10) for i in range(20)]

    # Sequential evaluation
    sequential_results = [evaluate_with_state(ind) for ind in population]

    # Parallel evaluation
    pool = multiprocessing.Pool(processes=4)
    parallel_results = pool.map(evaluate_with_state, population)
    pool.close()
    pool.join()

    # Compare
    if sequential_results == parallel_results:
        print(f"  ✓ No data corruption (all {len(population)} results match)")
        return True
    else:
        mismatches = sum(
            1 for s, p in zip(sequential_results, parallel_results) if s != p
        )
        print(f"  ✗ DATA CORRUPTION: {mismatches}/{len(population)} results differ")
        print(f"    Sequential sample: {sequential_results[:3]}")
        print(f"    Parallel sample: {parallel_results[:3]}")
        return False


data_ok = check_data_corruption()
print()

# ============================================================================
# Issue 4: Pool Hanging on Large Context
# ============================================================================
print("Issue 4: POOL HANGING ON LARGE CONTEXT")
print("-" * 40)


def evaluate_with_large_context(individual, large_data):
    """Evaluation with large bound data."""
    return (sum(individual) + len(large_data),)


def check_pool_hanging():
    """Check if pool hangs when given large context."""
    from functools import partial

    # Create large context (simulate courses, groups, etc.)
    large_data = list(range(10000))  # ~40KB

    population = [[i] * 10 for i in range(10)]

    eval_func = partial(evaluate_with_large_context, large_data=large_data)

    start = time.time()
    timeout = 5.0  # Should complete in <5s

    try:
        pool = multiprocessing.Pool(processes=2)
        # Use timeout to detect hanging
        result = pool.map_async(eval_func, population)
        results = result.get(timeout=timeout)
        elapsed = time.time() - start

        pool.close()
        pool.join()

        print(f"  ✓ No hanging (completed in {elapsed:.2f}s)")
        return True

    except multiprocessing.TimeoutError:
        print(f"  ✗ POOL HANGING: Did not complete within {timeout}s")
        pool.terminate()
        pool.join()
        return False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


hanging_ok = check_pool_hanging()
print()

# ============================================================================
# Issue 5: Windows Spawn Pickling Issues
# ============================================================================
print("Issue 5: WINDOWS SPAWN PICKLING ISSUES")
print("-" * 40)


# Test unpicklable objects
class UnpicklableClass:
    """Class with lambda (unpicklable)."""

    def __init__(self):
        self.func = lambda x: x * 2  # Lambda cannot be pickled


def evaluate_with_unpicklable(individual, unpicklable_obj):
    """This should fail on Windows spawn."""
    return (sum(individual),)


def check_unpicklable_detection():
    """Check if unpicklable objects are detected early."""
    from functools import partial
    import pickle

    obj = UnpicklableClass()

    # Try to pickle it
    try:
        pickle.dumps(obj)
        print("  ⚠ Object IS picklable (unexpected)")
        return None
    except (pickle.PicklingError, AttributeError) as e:
        print(f"  ✓ Unpicklable object detected: {type(e).__name__}")

        # Now check if multiprocessing fails gracefully
        try:
            eval_func = partial(evaluate_with_unpicklable, unpicklable_obj=obj)
            pool = multiprocessing.Pool(processes=1)
            result = pool.map(eval_func, [[1, 2, 3]])
            pool.close()
            pool.join()
            print("  ✗ BUG: Multiprocessing accepted unpicklable object")
            return False
        except Exception as e:
            print(f"  ✓ Multiprocessing rejected unpicklable: {type(e).__name__}")
            return True


unpicklable_ok = check_unpicklable_detection()
print()

# ============================================================================
# Issue 6: Creator Types Not Defined in Workers
# ============================================================================
print("Issue 6: CREATOR TYPES NOT DEFINED IN WORKERS")
print("-" * 40)


def worker_check_creator(x):
    """Check if creator types are available in worker."""
    # This will fail if creator types aren't properly set up
    try:
        # Try to access creator types
        if hasattr(creator, "FitnessMin"):
            return True
        else:
            return False
    except Exception:
        return False


def check_creator_in_workers():
    """Check if DEAP creator types are available in workers."""
    # Ensure creator types exist in main
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    pool = multiprocessing.Pool(processes=2)
    results = pool.map(worker_check_creator, range(4))
    pool.close()
    pool.join()

    if all(results):
        print(f"  ✓ Creator types available in all workers")
        return True
    else:
        failed = sum(1 for r in results if not r)
        print(f"  ✗ BUG: Creator types missing in {failed}/4 worker calls")
        print(f"    This is expected on Windows (spawn) without worker init")
        print(f"    Fix: Use worker initialization to set up creator types")
        return False


creator_ok = check_creator_in_workers()
print()

# ============================================================================
# Issue 7: Memory Leak from Repeated Pickling
# ============================================================================
print("Issue 7: MEMORY LEAK FROM REPEATED PICKLING")
print("-" * 40)


def check_memory_leak():
    """Check for memory growth from repeated pool.map() calls."""
    import psutil
    import gc

    process = psutil.Process()

    # Create large context
    large_context = list(range(50000))  # ~200KB

    from functools import partial

    eval_func = partial(evaluate_with_large_context, large_data=large_context)

    population = [[i] * 10 for i in range(20)]

    gc.collect()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Repeated evaluations (simulating generations)
    pool = multiprocessing.Pool(processes=2)
    for i in range(10):
        results = pool.map(eval_func, population)
    pool.close()
    pool.join()

    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB

    memory_growth = final_memory - initial_memory

    if memory_growth > 50:  # More than 50MB growth is suspicious
        print(f"  ⚠ POSSIBLE MEMORY LEAK: {memory_growth:.1f} MB growth")
        print(f"    Initial: {initial_memory:.1f} MB")
        print(f"    Final: {final_memory:.1f} MB")
        print(f"    This may be normal for large contexts with repeated pickling")
        return None
    else:
        print(f"  ✓ No significant memory leak ({memory_growth:.1f} MB growth)")
        return True


memory_ok = check_memory_leak()
print()

# ============================================================================
# Issue 8: Pool Not Utilizing All Cores
# ============================================================================
print("Issue 8: POOL NOT UTILIZING ALL CORES")
print("-" * 40)


def cpu_intensive_work(x):
    """CPU-bound work to test parallelization."""
    result = 0
    for i in range(1000000):
        result += i % 7
    return result


def check_cpu_utilization():
    """Check if pool actually uses multiple cores."""
    import psutil

    num_cores = multiprocessing.cpu_count()
    print(f"  System has {num_cores} CPU cores")

    # Sequential baseline
    start = time.time()
    sequential_results = [cpu_intensive_work(x) for x in range(8)]
    sequential_time = time.time() - start

    # Parallel with 4 workers
    pool = multiprocessing.Pool(processes=4)
    start = time.time()
    parallel_results = pool.map(cpu_intensive_work, range(8))
    parallel_time = time.time() - start
    pool.close()
    pool.join()

    speedup = sequential_time / parallel_time

    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Parallel (4 workers): {parallel_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")

    if speedup < 1.5:
        print(f"  ✗ POOR SPEEDUP: Expected >1.5x, got {speedup:.2f}x")
        print(f"    Pool may not be utilizing multiple cores effectively")
        return False
    else:
        print(f"  ✓ Good speedup achieved")
        return True


cpu_ok = check_cpu_utilization()
print()

# ============================================================================
# Issue 9: Toolbox.map Registration Bug
# ============================================================================
print("Issue 9: TOOLBOX.MAP REGISTRATION BUG")
print("-" * 40)


def simple_eval(individual):
    return (sum(individual),)


def check_toolbox_map_registration():
    """Check if toolbox.map correctly delegates to pool.map."""
    toolbox = base.Toolbox()

    pool = multiprocessing.Pool(processes=2)
    toolbox.register("map", pool.map)
    toolbox.register("evaluate", simple_eval)

    population = [[i] * 5 for i in range(10)]

    # Evaluate using toolbox.map
    start = time.time()
    results = list(toolbox.map(toolbox.evaluate, population))
    elapsed = time.time() - start

    pool.close()
    pool.join()

    # Verify results
    expected = [(sum(ind),) for ind in population]

    if results == expected:
        print(f"  ✓ toolbox.map correctly delegates to pool.map")
        print(f"    Evaluated {len(population)} individuals in {elapsed:.3f}s")
        return True
    else:
        print(f"  ✗ BUG: toolbox.map produced incorrect results")
        print(f"    Expected: {expected[:3]}")
        print(f"    Got: {results[:3]}")
        return False


toolbox_ok = check_toolbox_map_registration()
print()

# ============================================================================
# Issue 10: Pool Cleanup After Exception
# ============================================================================
print("Issue 10: POOL CLEANUP AFTER EXCEPTION")
print("-" * 40)


def check_pool_cleanup_after_exception():
    """Check if pool is properly cleaned up after exception."""
    import psutil

    initial_children = len(psutil.Process().children())

    try:
        pool = multiprocessing.Pool(processes=2)
        # This will raise an exception
        results = pool.map(failing_worker, range(10))
    except ValueError:
        # Exception caught, now clean up
        try:
            pool.close()
            pool.join()
        except Exception as e:
            print(f"  ✗ ERROR during cleanup: {e}")
            return False

    time.sleep(0.2)  # Give OS time to clean up
    final_children = len(psutil.Process().children())

    if final_children > initial_children:
        leaked = final_children - initial_children
        print(f"  ✗ CLEANUP FAILURE: {leaked} zombie processes after exception")
        return False
    else:
        print(f"  ✓ Pool properly cleaned up after exception")
        return True


cleanup_ok = check_pool_cleanup_after_exception()
print()

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

issues = {
    "Resource Leaks": leak_free,
    "Exception Propagation": exception_ok,
    "Data Corruption": data_ok,
    "Pool Hanging": hanging_ok,
    "Unpicklable Detection": unpicklable_ok,
    "Creator Types in Workers": creator_ok,
    "Memory Leak": memory_ok,
    "CPU Utilization": cpu_ok,
    "Toolbox.map Registration": toolbox_ok,
    "Pool Cleanup After Exception": cleanup_ok,
}

print("CRITICAL ISSUES:")
critical_found = False
for name, status in issues.items():
    if status is False:
        print(f"  ✗ {name}")
        critical_found = True

if not critical_found:
    print("  (none)")

print()
print("WARNINGS:")
warnings_found = False
for name, status in issues.items():
    if status is None:
        print(f"  ⚠ {name}")
        warnings_found = True

if not warnings_found:
    print("  (none)")

print()
print("ALL CHECKS PASSED:")
passed_found = False
for name, status in issues.items():
    if status is True:
        print(f"  ✓ {name}")
        passed_found = True

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)

critical_count = sum(1 for v in issues.values() if v is False)
warning_count = sum(1 for v in issues.values() if v is None)
passed_count = sum(1 for v in issues.values() if v is True)

print(f"  {critical_count} CRITICAL ISSUES")
print(f"  {warning_count} WARNINGS")
print(f"  {passed_count} PASSED")
print()

if critical_count > 0:
    print("[!ERR] MULTIPROCESSING HAS BUGS - REQUIRES FIXES")
else:
    print("NO CRITICAL BUGS FOUND (besides known pickling overhead)")

print()
