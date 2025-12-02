"""
Benchmarking script for LNS-CP hybrid system.

Compares baseline GA performance against GA+LNS-CP on the test dataset.

Usage:
    python scripts/benchmark_lns_cp.py --mode baseline
    python scripts/benchmark_lns_cp.py --mode lns-cp
    python scripts/benchmark_lns_cp.py --mode compare
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.experiments.baseline import BaselineTestConfig
from src.config.models import Config as PydanticConfig
from src.workflows.standard_run import run_standard_workflow


def run_baseline_benchmark(output_dir: Path) -> dict[str, Any]:
    """
    Run baseline GA (without LNS-CP) and collect metrics.

    Args:
        output_dir: Directory to save results

    Returns:
        Dictionary of benchmark results
    """
    print("\n" + "=" * 80)
    print("BASELINE GA BENCHMARK (without LNS-CP)")

    # Load test config with LNS disabled
    config = cast(PydanticConfig, BaselineTestConfig(lns_enabled=False).to_pydantic())

    print(f"\nConfiguration: {config.name} (LNS disabled)")
    print(f"Generations: {config.ga.ngen}")
    print(f"Population: {config.ga.pop_size}")
    print(f"Results will be saved to: {output_dir}")

    # Run workflow
    start_time = time.time()

    try:
        workflow_output = run_standard_workflow(
            pop_size=config.ga.pop_size,
            generations=config.ga.ngen,
            crossover_prob=config.ga.cxpb,
            mutation_prob=config.ga.mutpb,
            data_dir=getattr(config.io, "data_dir", "data"),
            output_dir=str(output_dir / "baseline_run"),
            config=config,
        )

        best_individual = workflow_output["best_individual"]
        eval_dir = workflow_output["output_path"]

        elapsed = time.time() - start_time

        # Extract metrics from best individual
        results = {
            "mode": "baseline",
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "hard_violations": best_individual.fitness.values[0],
            "soft_penalties": best_individual.fitness.values[1],
            "feasible": best_individual.fitness.values[0] == 0,
            "config": {
                "generations": config.ga.ngen,
                "population_size": config.ga.pop_size,
                "lns_enabled": config.lns.enabled,
            },
            "evaluation_dir": str(eval_dir),
        }

        # Save results
        results_path = output_dir / "baseline_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 80)
        print("BASELINE RESULTS")

        print(f"Runtime: {elapsed / 60:.2f} minutes")
        print(f"Final Hard Violations: {results['hard_violations']:.0f}")
        print(f"Final Soft Penalties: {results['soft_penalties']:.2f}")
        print(f"Feasible: {results['feasible']}")
        print(f"Results saved to: {results_path}")

        return results

    except Exception as e:
        print(f"\n[ERROR] Baseline benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return {"mode": "baseline", "error": str(e)}


def run_lns_cp_benchmark(output_dir: Path) -> dict[str, Any]:
    """
    Run GA+LNS-CP and collect metrics.

    Args:
        output_dir: Directory to save results

    Returns:
        Dictionary of benchmark results
    """
    print("\n" + "=" * 80)
    print("LNS-CP HYBRID BENCHMARK (with LNS-CP)")

    # Load test config with LNS enabled
    config = cast(PydanticConfig, BaselineTestConfig(lns_enabled=True).to_pydantic())

    print(f"\nConfiguration: {config.name} (LNS enabled)")
    print(f"Generations: {config.ga.ngen}")
    print(f"Population: {config.ga.pop_size}")
    print(f"LNS Trigger Interval: {config.lns.trigger_interval}")
    print(f"LNS CP Time Limit: {config.lns.cp_time_limit}s")
    print(f"Results will be saved to: {output_dir}")

    # Run workflow
    start_time = time.time()

    try:
        workflow_output = run_standard_workflow(
            pop_size=config.ga.pop_size,
            generations=config.ga.ngen,
            crossover_prob=config.ga.cxpb,
            mutation_prob=config.ga.mutpb,
            data_dir=getattr(config.io, "data_dir", "data"),
            output_dir=str(output_dir / "lns_cp_run"),
            config=config,
        )

        best_individual = workflow_output["best_individual"]
        eval_dir = workflow_output["output_path"]

        elapsed = time.time() - start_time

        # Extract LNS statistics
        from src.lns.lns_operator import get_lns_stats

        lns_stats = get_lns_stats()

        # Extract metrics
        results = {
            "mode": "lns-cp",
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "hard_violations": best_individual.fitness.values[0],
            "soft_penalties": best_individual.fitness.values[1],
            "feasible": best_individual.fitness.values[0] == 0,
            "lns_stats": {
                "total_attempts": lns_stats.total_attempts,
                "successful_repairs": lns_stats.successful_repairs,
                "failed_repairs": lns_stats.failed_repairs,
                "success_rate": (
                    lns_stats.successful_repairs / lns_stats.total_attempts * 100
                    if lns_stats.total_attempts > 0
                    else 0.0
                ),
                "avg_subproblem_size": lns_stats.avg_subproblem_size,
                "total_repair_time": lns_stats.total_repair_time,
            },
            "config": {
                "generations": config.ga.ngen,
                "population_size": config.ga.pop_size,
                "lns_enabled": config.lns.enabled,
                "lns_trigger_interval": config.lns.trigger_interval,
                "lns_cp_time_limit": config.lns.cp_time_limit,
            },
            "evaluation_dir": str(eval_dir),
        }

        # Save results
        results_path = output_dir / "lns_cp_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 80)
        print("LNS-CP RESULTS")

        print(f"Runtime: {elapsed / 60:.2f} minutes")
        print(f"Final Hard Violations: {results['hard_violations']:.0f}")
        print(f"Final Soft Penalties: {results['soft_penalties']:.2f}")
        print(f"Feasible: {results['feasible']}")
        print("\nLNS-CP Statistics:")
        print(f"  Total Attempts: {lns_stats.total_attempts}")
        print(f"  Successful: {lns_stats.successful_repairs}")
        print(f"  Failed: {lns_stats.failed_repairs}")
        print(f"  Success Rate: {results['lns_stats']['success_rate']:.1f}%")
        print(f"  Avg Subproblem Size: {lns_stats.avg_subproblem_size:.1f}")
        print(f"  Total Repair Time: {lns_stats.total_repair_time:.1f}s")
        print(f"\nResults saved to: {results_path}")

        return results

    except Exception as e:
        print(f"\n[ERROR] LNS-CP benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return {"mode": "lns-cp", "error": str(e)}


def compare_results(baseline_path: Path, lns_cp_path: Path, output_dir: Path):
    """
    Compare baseline and LNS-CP results.

    Args:
        baseline_path: Path to baseline results JSON
        lns_cp_path: Path to LNS-CP results JSON
        output_dir: Directory to save comparison report
    """
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")

    # Load results
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(lns_cp_path) as f:
        lns_cp = json.load(f)

    improvements: dict[str, Any] = {}
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "hard_violations": baseline.get("hard_violations", -1),
            "soft_penalties": baseline.get("soft_penalties", -1),
            "runtime_minutes": baseline.get("elapsed_minutes", -1),
            "feasible": baseline.get("feasible", False),
        },
        "lns_cp": {
            "hard_violations": lns_cp.get("hard_violations", -1),
            "soft_penalties": lns_cp.get("soft_penalties", -1),
            "runtime_minutes": lns_cp.get("elapsed_minutes", -1),
            "feasible": lns_cp.get("feasible", False),
            "lns_stats": lns_cp.get("lns_stats", {}),
        },
        "improvements": improvements,
    }

    baseline_summary = cast(dict[str, Any], comparison["baseline"])
    lns_summary = cast(dict[str, Any], comparison["lns_cp"])

    # Calculate improvements
    if (
        baseline.get("hard_violations", -1) >= 0
        and lns_cp.get("hard_violations", -1) >= 0
    ):
        hc_improvement = baseline["hard_violations"] - lns_cp["hard_violations"]
        hc_improvement_pct = (
            (hc_improvement / baseline["hard_violations"] * 100)
            if baseline["hard_violations"] > 0
            else 0.0
        )
        improvements["hard_violations"] = {
            "absolute": hc_improvement,
            "percentage": hc_improvement_pct,
        }

    if (
        baseline.get("soft_penalties", -1) >= 0
        and lns_cp.get("soft_penalties", -1) >= 0
    ):
        soft_improvement = baseline["soft_penalties"] - lns_cp["soft_penalties"]
        soft_improvement_pct = (
            (soft_improvement / baseline["soft_penalties"] * 100)
            if baseline["soft_penalties"] > 0
            else 0.0
        )
        improvements["soft_penalties"] = {
            "absolute": soft_improvement,
            "percentage": soft_improvement_pct,
        }

    runtime_overhead = lns_cp.get("elapsed_minutes", 0) - baseline.get(
        "elapsed_minutes", 0
    )
    improvements["runtime_overhead_minutes"] = runtime_overhead

    # Save comparison
    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Print report
    print("\nBaseline GA:")
    print(f"  Hard Violations: {baseline_summary['hard_violations']:.0f}")
    print(f"  Soft Penalties: {baseline_summary['soft_penalties']:.2f}")
    print(f"  Runtime: {baseline_summary['runtime_minutes']:.2f} minutes")
    print(f"  Feasible: {baseline_summary['feasible']}")

    print("\nLNS-CP Hybrid:")
    print(f"  Hard Violations: {lns_summary['hard_violations']:.0f}")
    print(f"  Soft Penalties: {lns_summary['soft_penalties']:.2f}")
    print(f"  Runtime: {lns_summary['runtime_minutes']:.2f} minutes")
    print(f"  Feasible: {lns_summary['feasible']}")

    if "hard_violations" in improvements:
        hc_imp = improvements["hard_violations"]
        print("\nHard Constraint Improvement:")
        print(f"  Absolute: {hc_imp['absolute']:.0f} violations")
        print(f"  Percentage: {hc_imp['percentage']:.1f}%")

    if "soft_penalties" in improvements:
        soft_imp = improvements["soft_penalties"]
        print("\nSoft Constraint Improvement:")
        print(f"  Absolute: {soft_imp['absolute']:.2f}")
        print(f"  Percentage: {soft_imp['percentage']:.1f}%")

    print(f"\nRuntime Overhead: {runtime_overhead:.2f} minutes")

    if "lns_stats" in lns_summary:
        stats = cast(dict[str, Any], lns_summary["lns_stats"])
        print("\nLNS-CP Statistics:")
        print(f"  Total Attempts: {stats.get('total_attempts', 0)}")
        print(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")

    print(f"\nComparison saved to: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LNS-CP hybrid system against baseline GA"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "lns-cp", "compare"],
        required=True,
        help="Benchmark mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: output/benchmark_TIMESTAMP)",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Path to baseline results JSON (for compare mode)",
    )
    parser.add_argument(
        "--lns-cp-results",
        type=str,
        default=None,
        help="Path to LNS-CP results JSON (for compare mode)",
    )

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / f"benchmark_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    if args.mode == "baseline":
        run_baseline_benchmark(output_dir)
    elif args.mode == "lns-cp":
        run_lns_cp_benchmark(output_dir)
    elif args.mode == "compare":
        if not args.baseline_results or not args.lns_cp_results:
            print(
                "[ERROR] Compare mode requires --baseline-results and --lns-cp-results"
            )
            return 1

        compare_results(
            Path(args.baseline_results),
            Path(args.lns_cp_results),
            output_dir,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
