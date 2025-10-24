"""
Phase 4: Comprehensive Benchmark for Metaheuristic Enhancements

Validates that the enhanced GA achieves ≥80% success rate
(reaching 0 hard constraint violations within 500 generations).

Usage:
    python run_phase4_benchmark.py

This will:
1. Run 30 independent trials with enhanced configuration
2. Track HC violations, convergence speed, diversity
3. Generate comprehensive performance report
4. Compare against target metrics

Expected Results (from enhance_metaheuristic.md):
- HC Violations: 0-20 (typically)
- Feasibility Rate: ~80% (80% of runs reach 0 violations)
- Convergence: 200-400 generations
"""

import os
import sys
import json
import time
import statistics
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.ga_params import (
    NGEN,
    POP_SIZE,
    USE_MULTIPROCESSING,
    ELITE_PRESERVATION,
    USE_ADAPTIVE_PROBABILITIES,
    USE_CONSTRAINT_GUIDED_MUTATION,
    POPULATION_STRATEGY,
    REPAIR_HEURISTICS_CONFIG,
)
from main import setup_workflow, run_ga_with_context


def run_single_trial(trial_num, total_trials, results_dir):
    """
    Run a single GA trial and collect metrics.

    Args:
        trial_num: Trial number (1-indexed)
        total_trials: Total number of trials
        results_dir: Directory to save trial results

    Returns:
        Dict with trial metrics
    """
    print(f"\n{'='*80}")
    print(f"TRIAL {trial_num}/{total_trials}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Run GA
        result = setup_workflow()

        if result is None:
            print(f"[ERROR] Trial {trial_num} returned None")
            return None

        # Extract metrics
        best_individual = result.get("best_individual")
        if not best_individual or not hasattr(best_individual, "fitness"):
            print(f"[ERROR] Trial {trial_num} has no valid best individual")
            return None

        hc_violations = int(best_individual.fitness.values[0])
        soft_penalty = float(best_individual.fitness.values[1])

        # Get evolution history
        history = result.get("history", {})
        hard_history = history.get("hard_violations", [])

        # Calculate convergence generation (when best HC was reached)
        convergence_gen = len(hard_history) - 1  # Default to last generation
        if hard_history:
            best_hc = min(hard_history)
            for gen, hc in enumerate(hard_history):
                if hc == best_hc:
                    convergence_gen = gen
                    break

        elapsed = time.time() - start_time

        metrics = {
            "trial_num": trial_num,
            "success": hc_violations == 0,
            "hc_violations": hc_violations,
            "soft_penalty": soft_penalty,
            "convergence_gen": convergence_gen,
            "total_generations": len(hard_history),
            "elapsed_seconds": round(elapsed, 2),
            "output_dir": result.get("output_dir", "unknown"),
        }

        print(f"\n✓ Trial {trial_num} Complete:")
        print(f"  HC Violations: {hc_violations}")
        print(f"  Soft Penalty: {soft_penalty:.2f}")
        print(f"  Convergence: Gen {convergence_gen}/{len(hard_history)}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Success: {'✅ YES' if metrics['success'] else '❌ NO'}")

        return metrics

    except Exception as e:
        print(f"[ERROR] Trial {trial_num} failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def analyze_results(all_results, results_dir):
    """
    Analyze benchmark results and generate report.

    Args:
        all_results: List of trial metric dicts
        results_dir: Directory to save report
    """
    # Filter out failed trials
    valid_results = [r for r in all_results if r is not None]

    if not valid_results:
        print("\n[ERROR] No valid results to analyze!")
        return

    # Calculate statistics
    total_trials = len(valid_results)
    successes = [r for r in valid_results if r["success"]]
    success_rate = len(successes) / total_trials * 100

    hc_violations = [r["hc_violations"] for r in valid_results]
    soft_penalties = [r["soft_penalty"] for r in valid_results]
    convergence_gens = [r["convergence_gen"] for r in valid_results]
    elapsed_times = [r["elapsed_seconds"] for r in valid_results]

    # Generate report
    report = {
        "benchmark_info": {
            "date": datetime.now().isoformat(),
            "total_trials": total_trials,
            "configuration": {
                "NGEN": NGEN,
                "POP_SIZE": POP_SIZE,
                "USE_MULTIPROCESSING": USE_MULTIPROCESSING,
                "ELITE_PRESERVATION": ELITE_PRESERVATION,
                "USE_ADAPTIVE_PROBABILITIES": USE_ADAPTIVE_PROBABILITIES,
                "USE_CONSTRAINT_GUIDED_MUTATION": USE_CONSTRAINT_GUIDED_MUTATION,
                "POPULATION_STRATEGY": POPULATION_STRATEGY,
                "memetic_mode": REPAIR_HEURISTICS_CONFIG.get("memetic_mode"),
                "max_iterations": REPAIR_HEURISTICS_CONFIG.get("max_iterations"),
                "memetic_iterations": REPAIR_HEURISTICS_CONFIG.get(
                    "memetic_iterations"
                ),
            },
        },
        "primary_metrics": {
            "success_rate_percent": round(success_rate, 2),
            "target_success_rate": 80.0,
            "meets_target": success_rate >= 80.0,
        },
        "hard_constraint_violations": {
            "mean": round(statistics.mean(hc_violations), 2),
            "median": statistics.median(hc_violations),
            "stdev": (
                round(statistics.stdev(hc_violations), 2)
                if len(hc_violations) > 1
                else 0
            ),
            "min": min(hc_violations),
            "max": max(hc_violations),
            "target_range": "0-20",
        },
        "convergence_speed": {
            "mean_generations": round(statistics.mean(convergence_gens), 2),
            "median_generations": statistics.median(convergence_gens),
            "min_generations": min(convergence_gens),
            "max_generations": max(convergence_gens),
            "target_range": "200-400",
        },
        "soft_penalties": {
            "mean": round(statistics.mean(soft_penalties), 2),
            "median": round(statistics.median(soft_penalties), 2),
            "min": round(min(soft_penalties), 2),
            "max": round(max(soft_penalties), 2),
        },
        "performance": {
            "mean_time_seconds": round(statistics.mean(elapsed_times), 2),
            "total_time_seconds": round(sum(elapsed_times), 2),
            "total_time_hours": round(sum(elapsed_times) / 3600, 2),
        },
        "detailed_results": valid_results,
    }

    # Save JSON report
    report_file = results_dir / "benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal Trials: {total_trials}")
    print(f"\n🎯 PRIMARY GOAL: ≥80% Success Rate")
    print(f"   Actual: {success_rate:.1f}%")
    print(f"   Status: {'✅ ACHIEVED' if success_rate >= 80 else '❌ NOT MET'}")

    print(f"\n📊 Hard Constraint Violations:")
    print(f"   Mean: {statistics.mean(hc_violations):.2f}")
    print(f"   Median: {statistics.median(hc_violations)}")
    print(f"   Range: {min(hc_violations)} - {max(hc_violations)}")
    print(f"   Target: 0-20")

    print(f"\n⚡ Convergence Speed:")
    print(f"   Mean: {statistics.mean(convergence_gens):.1f} generations")
    print(f"   Median: {statistics.median(convergence_gens)} generations")
    print(f"   Target: 200-400 generations")

    print(f"\n💎 Soft Penalties (Secondary):")
    print(f"   Mean: {statistics.mean(soft_penalties):.2f}")
    print(f"   Median: {statistics.median(soft_penalties):.2f}")

    print(f"\n⏱️  Performance:")
    print(f"   Mean Time: {statistics.mean(elapsed_times):.1f}s per trial")
    print(f"   Total Time: {sum(elapsed_times)/3600:.2f} hours")

    print(f"\n📁 Full Report: {report_file}")
    print("=" * 80)

    return report


def main():
    """Run Phase 4 comprehensive benchmark."""
    print("=" * 80)
    print("PHASE 4: COMPREHENSIVE BENCHMARK")
    print("Metaheuristic Enhancement Strategy Validation")
    print("=" * 80)

    # Configuration check
    print("\n📋 Configuration:")
    print(f"   Generations: {NGEN}")
    print(f"   Population: {POP_SIZE}")
    print(f"   Multiprocessing: {USE_MULTIPROCESSING}")
    print(f"   Elitism: {ELITE_PRESERVATION}")
    print(f"   Adaptive Probabilities: {USE_ADAPTIVE_PROBABILITIES}")
    print(f"   Constraint-Guided Mutation: {USE_CONSTRAINT_GUIDED_MUTATION}")
    print(f"   Population Strategy: {POPULATION_STRATEGY}")
    print(f"   Memetic Mode: {REPAIR_HEURISTICS_CONFIG.get('memetic_mode')}")
    print(f"   Max Iterations: {REPAIR_HEURISTICS_CONFIG.get('max_iterations')}")
    print(
        f"   Memetic Iterations: {REPAIR_HEURISTICS_CONFIG.get('memetic_iterations')}"
    )

    # Verify production settings
    if NGEN < 500:
        print(f"\n⚠️  WARNING: NGEN={NGEN} is less than recommended 500 for benchmark")
        print("   Consider setting NGEN=500 in config/ga_params.py")

    if POP_SIZE < 100:
        print(
            f"\n⚠️  WARNING: POP_SIZE={POP_SIZE} is less than recommended 100 for benchmark"
        )
        print("   Consider setting POP_SIZE=100 in config/ga_params.py")

    # Number of trials
    num_trials = 30  # As per strategy document

    print(f"\n🎯 Target: ≥80% success rate (0 HC violations)")
    print(f"   Running {num_trials} trials...")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"benchmark_results/phase4_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run trials
    all_results = []
    for trial_num in range(1, num_trials + 1):
        result = run_single_trial(trial_num, num_trials, results_dir)
        if result:
            all_results.append(result)

    # Analyze and report
    if all_results:
        analyze_results(all_results, results_dir)
    else:
        print("\n[ERROR] All trials failed!")

    print(f"\n✓ Benchmark complete!")
    print(f"   Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
