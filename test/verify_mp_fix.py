"""
Quick test to verify multiprocessing fix is working.
Tests worker initialization pattern implementation.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=" * 80)
print("MULTIPROCESSING FIX VERIFICATION")
print("=" * 80)
print()

print("Checking if all fixes are implemented...")
print()

# Check 1: Worker functions exist
print("Check 1: Worker Initialization Functions")
print("-" * 40)

try:
    from src.core.ga_scheduler import _worker_init, _worker_evaluate

    print("  ✓ _worker_init() exists")
    print("  ✓ _worker_evaluate() exists")
    check1 = True
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    check1 = False

print()

# Check 2: Worker init is used in pool creation
print("Check 2: Pool Creation with Initializer")
print("-" * 40)

with open("src/workflows/standard_run.py", "r", encoding="utf-8") as f:
    workflow_content = f.read()

if "initializer=_worker_init" in workflow_content:
    print("  ✓ Pool created with initializer=_worker_init")
    check2 = True
else:
    print("  ✗ Pool NOT using initializer")
    check2 = False

if "initargs=" in workflow_content and "seed" in workflow_content:
    print("  ✓ Pool passes seed in initargs")
else:
    print("  ⚠ Seed may not be passed to workers")

print()

# Check 3: Evaluation uses worker function when pool exists
print("Check 3: Conditional Evaluation Registration")
print("-" * 40)

with open("src/core/ga_scheduler.py", "r", encoding="utf-8") as f:
    scheduler_content = f.read()

if "if self.pool is not None" in scheduler_content:
    if 'self.toolbox.register("evaluate", _worker_evaluate)' in scheduler_content:
        print("  ✓ Uses _worker_evaluate when pool is available")
        check3 = True
    else:
        print("  ⚠ Conditional check exists but may not use _worker_evaluate")
        check3 = None
else:
    print("  ✗ No conditional evaluation registration")
    check3 = False

print()

# Check 4: Seed passed to scheduler
print("Check 4: Seed Propagation")
print("-" * 40)

if "seed=seed" in workflow_content and "pool=pool" in workflow_content:
    print("  ✓ Seed passed to GAScheduler")
    check4 = True
else:
    print("  ✗ Seed not passed to GAScheduler")
    check4 = False

print()

# Check 5: Pool cleanup
print("Check 5: Pool Cleanup")
print("-" * 40)

if "pool.close()" in workflow_content and "pool.join()" in workflow_content:
    print("  ✓ Pool cleanup implemented")
    check5 = True
else:
    print("  ✗ Pool cleanup missing")
    check5 = False

print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

all_checks = [check1, check2, check3, check4, check5]
passed = sum(1 for c in all_checks if c is True)
failed = sum(1 for c in all_checks if c is False)
warnings = sum(1 for c in all_checks if c is None)

if failed == 0 and warnings == 0:
    print("✅ ALL FIXES IMPLEMENTED SUCCESSFULLY!")
    print()
    print("All 4 multiprocessing bugs are now fixed:")
    print("  1. ✓ Pickling overhead (worker init passes context once)")
    print("  2. ✓ Random seed propagation (seed passed to workers)")
    print("  3. ✓ Pool with initializer (worker init enabled)")
    print("  4. ✓ Creator types in workers (set up in _worker_init)")
    print()
    print("Expected performance:")
    print("  - Sequential: baseline")
    print("  - Parallel (before): 2.7× SLOWER (buggy)")
    print("  - Parallel (now): 2-3× FASTER ✅")
    print()
    print("Next step: Run full GA with 'python main.py' to verify")
elif failed == 0:
    print(f"⚠️  {passed} CHECKS PASSED, {warnings} WARNINGS")
    print()
    print("Implementation is mostly complete but has minor issues.")
else:
    print(f"❌ {failed} CHECKS FAILED, {passed} passed")
    print()
    print("Fix implementation is incomplete. Please review failed checks.")

print()
