"""
Quick multiprocessing issues check.
"""

print("=" * 80)
print("MULTIPROCESSING ISSUES CHECK")
print("=" * 80)
print()

# Issue 1: Pickling Overhead (Already Documented)
print("✗ Issue 1: PICKLING OVERHEAD")
print("  Status: CRITICAL - Makes multiprocessing 2.7x SLOWER")
print("  Cause: Partial function with 120KB context re-pickled every pool.map()")
print("  Fix: Use worker initialization with process-local context")
print("  Documented in: docs/BUGFIX_multiprocessing_pickling_overhead.md")
print()

# Issue 2: Random Seed Not Propagated to Workers
print("⚠ Issue 2: RANDOM SEED NOT PROPAGATED")
print("  Status: WARNING - Non-reproducible results with multiprocessing")
print("  Cause: Workers don't inherit random.seed() from main process")
print("  Impact: Same seed in main produces different results with multiprocessing")
print("  Fix: Seed workers in initializer function")
print()

# Issue 3: Global Singleton (_QTS)
print("✓ Issue 3: GLOBAL SINGLETON")
print("  Status: SAFE - No issues")
print("  Reason: Read-only, each worker gets its own copy (spawn)")
print()

# Issue 4: DEAP Creator Types
print("✓ Issue 4: DEAP CREATOR TYPES")
print("  Status: SAFE - Uses hasattr guard")
print("  Reason: Idempotent initialization in each worker")
print()

# Issue 5: Pool Cleanup
print("✓ Issue 5: POOL CLEANUP")
print("  Status: OK - Uses try/finally with close() and join()")
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("CRITICAL (must fix):")
print("  1. Pickling overhead - multiprocessing is 2.7x SLOWER than sequential")
print()
print("MEDIUM (should fix):")
print("  2. Random seed - results not reproducible with multiprocessing")
print()
print("Total issues found: 2")
print()
print("See docs/BUGFIX_multiprocessing_pickling_overhead.md for fix details")
print("=" * 80)
