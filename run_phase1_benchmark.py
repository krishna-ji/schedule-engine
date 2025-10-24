"""
Quick Benchmark Script for Phase 1 Enhancements

Runs a short test to verify the enhancements are working.
For full validation, increase NGEN to 500 and run multiple times.
"""

import time
from main import main
from config import ga_params

print("=" * 70)
print("PHASE 1 QUICK BENCHMARK")
print("=" * 70)

print("\nCurrent Configuration:")
print(f"  Population Size: {ga_params.POP_SIZE}")
print(f"  Generations: {ga_params.NGEN}")
print(f"  Memetic Mode: {ga_params.REPAIR_HEURISTICS_CONFIG['memetic_mode']}")
print(
    f"  Max Repair Iterations: {ga_params.REPAIR_HEURISTICS_CONFIG['max_iterations']}"
)
print(
    f"  Memetic Iterations: {ga_params.REPAIR_HEURISTICS_CONFIG['memetic_iterations']}"
)
print(f"  Multiprocessing: {ga_params.USE_MULTIPROCESSING}")

print("\n" + "=" * 70)
print("Starting GA Run...")
print("=" * 70)

start_time = time.time()

try:
    main()
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total Time: {elapsed:.1f}s")
    print("\nTo run full benchmark:")
    print("  1. Set NGEN=500 in config/ga_params.py")
    print("  2. Set POP_SIZE=100 for production runs")
    print("  3. Run this script 10 times and compare results")
    print("\nExpected improvements:")
    print("  ✓ Faster convergence (fewer generations to best solution)")
    print("  ✓ Lower hard constraint violations")
    print("  ✓ Best fitness never degrades (elitism guarantee)")

except Exception as e:
    print(f"\n❌ Error during benchmark: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
