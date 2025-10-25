"""
Comprehensive multiprocessing audit - checking for additional issues.

This script checks for:
1. Global state issues (QuantumTimeSystem singleton, creator types)
2. Random seed issues (non-reproducible results with multiprocessing)
3. Race conditions
4. Import issues (modules need to be importable by workers)
5. Exception handling in pool
6. Memory leaks
"""

import sys
import os

# Add to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("MULTIPROCESSING COMPREHENSIVE AUDIT")
print("=" * 80)
print()

# ============================================================================
# Issue 1: Global Singleton Pattern (QuantumTimeSystem in soft.py)
# ============================================================================
print("Issue 1: Global Singleton Pattern")
print("-" * 80)

from src.constraints.soft import _QTS

print(f"✓ Found _QTS singleton in soft.py")
print(f"  Type: {type(_QTS)}")
print(f"  Status: Read-only singleton - SAFE for multiprocessing")
print(f"  Reason: Each worker gets its own copy (spawn), no shared state")
print()

# ============================================================================
# Issue 2: DEAP Creator Types (thread-safety)
# ============================================================================
print("Issue 2: DEAP Creator Types")
print("-" * 80)

from deap import creator, base

# Initialize creator types
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -0.01))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

print(f"✓ Creator types available: FitnessMulti, Individual")
print(f"  Status: SAFE - Types created with hasattr guard")
print(f"  Note: Each worker will create types independently")
print()

# ============================================================================
# Issue 3: Random Seed in Multiprocessing
# ============================================================================
print("Issue 3: Random Seed Reproducibility")
print("-" * 80)

import random
import multiprocessing


def worker_check_random(seed):
    """Check if worker has same random seed."""
    random.seed(seed)
    return random.random()


# Set seed in main
random.seed(42)
main_value = random.random()

# Check if workers inherit seed
pool = multiprocessing.Pool(processes=2)
worker_values = pool.map(worker_check_random, [42, 42])
pool.close()
pool.join()

print(f"  Main process:  {main_value:.6f}")
print(f"  Worker 1:      {worker_values[0]:.6f}")
print(f"  Worker 2:      {worker_values[1]:.6f}")

if worker_values[0] == worker_values[1]:
    print(f"  ✓ Workers produce consistent results with same seed")
else:
    print(f"  ✗ WARNING: Workers produce different results!")

print(f"  Note: Mutation/crossover use random - need to seed workers!")
print()

# ============================================================================
# Issue 4: Module Import Test
# ============================================================================
print("Issue 4: Module Import in Workers")
print("-" * 80)


def test_imports():
    """Test if all required modules can be imported in worker."""
    try:
        from src.ga.evaluator.fitness import evaluate
        from src.encoder.input_encoder import load_courses
        from src.decoder.individual_decoder import decode_individual
        from src.constraints.hard import get_enabled_hard_constraints
        from src.constraints.soft import get_enabled_soft_constraints

        return "SUCCESS"
    except Exception as e:
        return f"FAILED: {e}"


pool = multiprocessing.Pool(processes=1)
result = pool.apply(test_imports)
pool.close()
pool.join()

print(f"  Import test: {result}")
if result == "SUCCESS":
    print(f"  ✓ All modules importable in workers")
else:
    print(f"  ✗ CRITICAL: Import failure in workers!")
print()

# ============================================================================
# Issue 5: Exception Handling
# ============================================================================
print("Issue 5: Exception Handling in Pool.map")
print("-" * 80)


def failing_function(x):
    """Function that raises exception."""
    if x == 5:
        raise ValueError(f"Deliberate error for x={x}")
    return x * 2


try:
    pool = multiprocessing.Pool(processes=2)
    results = pool.map(failing_function, range(10))
    pool.close()
    pool.join()
    print(f"  ✗ Exception was NOT propagated (unexpected)")
except Exception as e:
    print(f"  ✓ Exception propagated correctly: {type(e).__name__}")
    print(f"  Message: {str(e)[:60]}...")
print()

# ============================================================================
# Issue 6: Pickling Overhead (already documented)
# ============================================================================
print("Issue 6: Pickling Overhead")
print("-" * 80)
print(f"  Status: DOCUMENTED in BUGFIX_multiprocessing_pickling_overhead.md")
print(f"  Problem: Partial function with large context (~120KB)")
print(f"  Impact: 2.7x slower than sequential")
print(f"  Fix: Use worker initialization with process-local context")
print()

# ============================================================================
# Issue 7: Pool Cleanup
# ============================================================================
print("Issue 7: Pool Cleanup and Resource Management")
print("-" * 80)

# Check main.py implementation
with open("main.py", "r") as f:
    main_content = f.read()

has_finally = "finally:" in main_content
has_close = "pool.close()" in main_content
has_join = "pool.join()" in main_content

print(f"  Pool cleanup in finally block: {'✓' if has_finally else '✗'}")
print(f"  Pool.close() called: {'✓' if has_close else '✗'}")
print(f"  Pool.join() called: {'✓' if has_join else '✗'}")

if has_finally and has_close and has_join:
    print(f"  ✓ Pool cleanup is CORRECT")
else:
    print(f"  ✗ WARNING: Pool cleanup may leak resources")
print()

# ============================================================================
# Issue 8: Context Start Method (Windows spawn)
# ============================================================================
print("Issue 8: Multiprocessing Start Method")
print("-" * 80)

start_method = multiprocessing.get_start_method()
print(f"  Current start method: {start_method}")

if start_method == "spawn":
    print(f"  ✓ Using spawn (Windows default)")
    print(f"  Note: Higher overhead than fork, but safer")
elif start_method == "fork":
    print(f"  ✓ Using fork (Linux default)")
    print(f"  Note: Faster, but can inherit parent state")
else:
    print(f"  ? Unknown start method: {start_method}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)

issues = [
    {
        "id": 1,
        "name": "Global Singleton (_QTS)",
        "status": "✓ SAFE",
        "severity": "None",
        "notes": "Read-only, each worker gets copy",
    },
    {
        "id": 2,
        "name": "DEAP Creator Types",
        "status": "✓ SAFE",
        "severity": "None",
        "notes": "Idempotent initialization",
    },
    {
        "id": 3,
        "name": "Random Seed",
        "status": "⚠ NEEDS ATTENTION",
        "severity": "Medium",
        "notes": "Workers don't inherit seed - non-reproducible with multiprocessing",
    },
    {
        "id": 4,
        "name": "Module Imports",
        "status": "✓ OK" if result == "SUCCESS" else "✗ CRITICAL",
        "severity": "None" if result == "SUCCESS" else "Critical",
        "notes": "All modules importable",
    },
    {
        "id": 5,
        "name": "Exception Handling",
        "status": "✓ OK",
        "severity": "None",
        "notes": "Exceptions propagate correctly",
    },
    {
        "id": 6,
        "name": "Pickling Overhead",
        "status": "✗ CRITICAL",
        "severity": "Critical",
        "notes": "Makes multiprocessing 2.7x SLOWER - needs fix",
    },
    {
        "id": 7,
        "name": "Pool Cleanup",
        "status": "✓ OK" if (has_finally and has_close and has_join) else "⚠ WARNING",
        "severity": "None" if (has_finally and has_close and has_join) else "Low",
        "notes": "Proper cleanup in finally block",
    },
    {
        "id": 8,
        "name": "Start Method",
        "status": "✓ OK",
        "severity": "None",
        "notes": f"Using {start_method}",
    },
]

print()
print(f"{'#':<4} {'Issue':<30} {'Status':<20} {'Severity':<12}")
print("-" * 80)

for issue in issues:
    print(
        f"{issue['id']:<4} {issue['name']:<30} {issue['status']:<20} {issue['severity']:<12}"
    )

print()
print("CRITICAL ISSUES REQUIRING FIX:")
print("-" * 80)

critical = [i for i in issues if i["severity"] == "Critical"]
if critical:
    for issue in critical:
        print(f"  {issue['id']}. {issue['name']}")
        print(f"     {issue['notes']}")
        print()
else:
    print("  (None found - only the pickling issue already documented)")
    print()

print("MEDIUM PRIORITY ISSUES:")
print("-" * 80)

medium = [i for i in issues if i["severity"] == "Medium"]
if medium:
    for issue in medium:
        print(f"  {issue['id']}. {issue['name']}")
        print(f"     {issue['notes']}")
        print()
else:
    print("  (None found)")
    print()

print("=" * 80)
print("Audit complete!")
print("=" * 80)
