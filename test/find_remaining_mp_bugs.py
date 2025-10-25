"""
Find Remaining Multiprocessing Bugs - SAFE VERSION
==================================================
Quick check for additional bugs beyond the 2 known issues.
No actual multiprocessing calls to avoid hanging.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=" * 80)
print("MULTIPROCESSING BUG DETECTION - SAFE MODE")
print("=" * 80)
print()

print("Known Issues (already identified):")
print("  1. ✗ Pickling overhead - makes parallel 2.7x SLOWER than sequential")
print("  2. ⚠ Random seed not propagated to workers")
print()

print("Checking for additional bugs...")
print()

# ============================================================================
# Check 1: Pool creation without initializer
# ============================================================================
print("Check 1: Pool Creation Pattern")
print("-" * 40)

with open("main.py", "r", encoding="utf-8") as f:
    main_content = f.read()

if "Pool(processes=" in main_content:
    if "initializer=" not in main_content:
        print("  ✗ BUG FOUND: Pool created without initializer")
        print("    Location: main.py")
        print("    Impact: Cannot use worker initialization pattern")
        print("    Fix: Add initializer=_worker_init, initargs=(...)")
        bug_1 = True
    else:
        print("  ✓ Pool has initializer")
        bug_1 = False
else:
    print("  ⚠ Could not find Pool creation")
    bug_1 = None

print()

# ============================================================================
# Check 2: Partial function with large context
# ============================================================================
print("Check 2: Evaluation Function Registration")
print("-" * 40)

with open("src/core/ga_scheduler.py", "r", encoding="utf-8") as f:
    scheduler_content = f.read()

if 'toolbox.register("evaluate"' in scheduler_content:
    # Check if it's registering with bound parameters
    lines = scheduler_content.split("\n")
    for i, line in enumerate(lines):
        if 'toolbox.register("evaluate"' in line:
            if "courses=" in line or "instructors=" in line:
                print("  ✗ BUG FOUND: Evaluation uses partial function with bound data")
                print("    Location: src/core/ga_scheduler.py")
                print("    Impact: Large context pickled on every pool.map() call")
                print("    Fix: Use worker initialization with module-level context")
                bug_2 = True
                break
    else:
        print("  ✓ Evaluation does not use bound parameters")
        bug_2 = False
else:
    print("  ⚠ Could not find evaluate registration")
    bug_2 = None

print()

# ============================================================================
# Check 3: Pool cleanup in exception handler
# ============================================================================
print("Check 3: Pool Cleanup Pattern")
print("-" * 40)

if "finally:" in main_content:
    if "pool.close()" in main_content and "pool.join()" in main_content:
        # Check they're in finally block
        finally_idx = main_content.find("finally:")
        close_idx = main_content.find("pool.close()")
        join_idx = main_content.find("pool.join()")

        if close_idx > finally_idx and join_idx > finally_idx:
            print("  ✓ Pool cleanup in finally block")
            bug_3 = False
        else:
            print("  ⚠ Pool cleanup not in finally block")
            bug_3 = None
    else:
        print("  ✗ BUG FOUND: Missing pool.close() or pool.join()")
        print("    Location: main.py")
        print("    Impact: Resource leak, zombie processes")
        bug_3 = True
else:
    print("  ⚠ No finally block found")
    bug_3 = None

print()

# ============================================================================
# Check 4: Creator types setup
# ============================================================================
print("Check 4: DEAP Creator Types in Workers")
print("-" * 40)

# Check if there's worker init that sets up creator
if "_worker_init" in scheduler_content or "def worker_init" in scheduler_content:
    if "creator.create" in scheduler_content:
        print("  ✓ Worker init sets up creator types")
        bug_4 = False
    else:
        print("  ⚠ Worker init exists but doesn't set up creator")
        bug_4 = None
else:
    print("  ✗ BUG FOUND: No worker initialization for creator types")
    print("    Location: src/core/ga_scheduler.py")
    print("    Impact: Windows spawn can't access creator.Individual in workers")
    print("    Fix: Add creator.create() calls in worker_init()")
    bug_4 = True

print()

# ============================================================================
# Check 5: Random seed in worker init
# ============================================================================
print("Check 5: Random Seed Propagation")
print("-" * 40)

if "_worker_init" in scheduler_content or "def worker_init" in scheduler_content:
    if "random.seed" in scheduler_content:
        print("  ✓ Worker init sets random seed")
        bug_5 = False
    else:
        print("  ✗ BUG FOUND: Worker init doesn't set random seed")
        print("    Location: src/core/ga_scheduler.py")
        print("    Impact: Non-reproducible results with multiprocessing")
        print("    Fix: Add random.seed(seed) in worker_init()")
        bug_5 = True
else:
    print("  ⚠ No worker init (seed cannot be propagated)")
    bug_5 = None

print()

# ============================================================================
# Check 6: Pool passed but not used
# ============================================================================
print("Check 6: Pool Utilization")
print("-" * 40)

if "self.pool = pool" in scheduler_content:
    if (
        'toolbox.register("map", self.pool.map)' in scheduler_content
        or 'toolbox.register("map", pool.map)' in scheduler_content
    ):
        print("  ✓ Pool registered with toolbox.map")
        bug_6 = False
    else:
        print("  ✗ BUG FOUND: Pool stored but never used")
        print("    Location: src/core/ga_scheduler.py")
        print("    Impact: Parallel evaluation disabled despite pool being passed")
        print("    Fix: Add toolbox.register('map', pool.map) when pool is not None")
        bug_6 = True
else:
    print("  ⚠ Pool parameter not stored")
    bug_6 = None

print()

# ============================================================================
# Check 7: Multiprocessing context not set
# ============================================================================
print("Check 7: Multiprocessing Context")
print("-" * 40)

if "multiprocessing.set_start_method" in main_content:
    print("  ✓ Start method explicitly set")
    bug_7 = False
else:
    print("  ⚠ Start method not explicitly set (using system default)")
    print("    Windows: spawn (safest)")
    print("    Linux: fork (faster but can have issues)")
    print("    Not a bug, but explicit is better")
    bug_7 = None

print()

# ============================================================================
# Check 8: Pool size configuration
# ============================================================================
print("Check 8: Pool Size Configuration")
print("-" * 40)

if "Pool(processes=NUM_WORKERS)" in main_content:
    # Check if NUM_WORKERS can be None
    with open("config/ga_params.py", "r", encoding="utf-8") as f:
        config_content = f.read()

    if "NUM_WORKERS" in config_content:
        if (
            "NUM_WORKERS = None" in config_content
            or "NUM_WORKERS: Optional" in config_content
        ):
            print("  ✓ NUM_WORKERS can be None (auto-detect)")
            bug_8 = False
        else:
            print("  ✓ NUM_WORKERS has fixed value")
            bug_8 = False
    else:
        print("  ⚠ NUM_WORKERS not found in config")
        bug_8 = None
else:
    print("  ⚠ Pool processes configuration not found")
    bug_8 = None

print()

# ============================================================================
# Check 9: Pool timeout not set
# ============================================================================
print("Check 9: Pool Evaluation Timeout")
print("-" * 40)

if "map_async" in scheduler_content and "get(timeout=" in scheduler_content:
    print("  ✓ Using map_async with timeout")
    bug_9 = False
elif "pool.map(" in scheduler_content or "toolbox.map(" in scheduler_content:
    print("  ⚠ No timeout on pool.map() calls")
    print("    Impact: Can hang indefinitely if worker crashes")
    print("    Recommendation: Use map_async with timeout for production")
    bug_9 = None
else:
    print("  ⚠ Could not find pool.map calls")
    bug_9 = None

print()

# ============================================================================
# Check 10: Shared memory not used
# ============================================================================
print("Check 10: Shared Memory Usage")
print("-" * 40)

if (
    "multiprocessing.Manager" in scheduler_content
    or "multiprocessing.Array" in scheduler_content
):
    print("  ✓ Using shared memory")
    bug_10 = False
else:
    print("  ⚠ Not using shared memory")
    print("    Current: Pickling full context on every evaluation")
    print("    Alternative: Use Manager() for large read-only data")
    print("    Note: Worker init is better for this use case")
    bug_10 = None

print()

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

bugs = {
    "Pool without initializer": bug_1,
    "Partial function with large context": bug_2,
    "Pool cleanup pattern": bug_3,
    "Creator types in workers": bug_4,
    "Random seed propagation": bug_5,
    "Pool not registered with toolbox": bug_6,
    "Multiprocessing context": bug_7,
    "Pool size configuration": bug_8,
    "Pool timeout": bug_9,
    "Shared memory usage": bug_10,
}

critical_bugs = [name for name, status in bugs.items() if status is True]
warnings = [name for name, status in bugs.items() if status is None]
passing = [name for name, status in bugs.items() if status is False]

print("🔴 CRITICAL BUGS FOUND:")
if critical_bugs:
    for bug in critical_bugs:
        print(f"  ✗ {bug}")
else:
    print("  (none - only known issues remain)")

print()
print("⚠️  WARNINGS:")
if warnings:
    for warning in warnings:
        print(f"  ⚠ {warning}")
else:
    print("  (none)")

print()
print("✅ PASSING CHECKS:")
if passing:
    for check in passing:
        print(f"  ✓ {check}")
else:
    print("  (none)")

print()
print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)

if critical_bugs:
    print(f"❌ Found {len(critical_bugs)} NEW critical bugs (plus 2 known)")
    print()
    print("All bugs must be fixed for proper multiprocessing:")
    print("  1. Pickling overhead (KNOWN)")
    print("  2. Random seed propagation (KNOWN)")
    for i, bug in enumerate(critical_bugs, 3):
        print(f"  {i}. {bug} (NEW)")
else:
    print("✅ NO NEW BUGS - Only 2 known issues remain:")
    print("  1. Pickling overhead (causes 2.7x slowdown)")
    print("  2. Random seed not propagated (non-reproducible results)")

print()
print("Fix implementation: See docs/BUGFIX_multiprocessing_pickling_overhead.md")
print()
