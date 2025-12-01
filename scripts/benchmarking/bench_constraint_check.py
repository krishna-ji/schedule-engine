"""
Constraint Checking Performance Benchmark Script

Measures the performance of constraint evaluation functions with various
dataset sizes to validate complexity analysis and identify bottlenecks.

Usage:
    # Basic benchmark
    uv run python scripts/bench_constraint_check.py

    # With profiling
    python -m cProfile -o profile.out scripts/bench_constraint_check.py
    py-spy record -o profile.svg -- python scripts/bench_constraint_check.py

    # Custom parameters
    python scripts/bench_constraint_check.py --pop 100 --genes 200 --runs 50
"""

import argparse
import json
import statistics

# Add src to path
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constraints.registry import (
    constraint_needs_courses,
    get_enabled_hard_constraints,
    get_enabled_soft_constraints,
)
from src.core.types import SchedulingContext
from src.decoder.individual_decoder import decode_individual
from src.encoder.input_encoder import (
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.ga.evaluator.fitness import evaluate
from src.ga.population import generate_course_group_aware_population


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    dataset_size: str
    num_sessions: int
    num_quanta: int
    population_size: int

    # Timing results (ms)
    decode_time_ms: float
    hard_constraints_time_ms: float
    soft_constraints_time_ms: float
    total_time_ms: float

    # Per-constraint breakdown
    constraint_times: dict[str, float]

    # Statistical measures
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_size": self.dataset_size,
            "num_sessions": self.num_sessions,
            "num_quanta": self.num_quanta,
            "population_size": self.population_size,
            "timings": {
                "decode_ms": round(self.decode_time_ms, 3),
                "hard_constraints_ms": round(self.hard_constraints_time_ms, 3),
                "soft_constraints_ms": round(self.soft_constraints_time_ms, 3),
                "total_ms": round(self.total_time_ms, 3),
                "std_dev_ms": round(self.std_dev_ms, 3),
                "min_ms": round(self.min_time_ms, 3),
                "max_ms": round(self.max_time_ms, 3),
            },
            "per_constraint_ms": {
                name: round(t, 3) for name, t in self.constraint_times.items()
            },
        }


def benchmark_decoding(individual, context, num_runs: int = 100) -> tuple[float, float]:
    """
    Benchmark decoding performance.

    Returns:
        (mean_time_ms, std_dev_ms)
    """
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        decode_individual(
            individual,
            context.courses,
            context.instructors,
            context.groups,
            context.rooms,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return statistics.mean(times), statistics.stdev(times)


def benchmark_constraint(
    constraint_name: str, constraint_func, sessions, courses=None, num_runs: int = 100
) -> tuple[float, float]:
    """
    Benchmark a single constraint function.

    Returns:
        (mean_time_ms, std_dev_ms)
    """
    times = []

    needs_courses = constraint_needs_courses(constraint_name)

    for _ in range(num_runs):
        start = time.perf_counter()
        if needs_courses:
            constraint_func(sessions, courses)
        else:
            constraint_func(sessions)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return statistics.mean(times), statistics.stdev(times)


def benchmark_full_evaluation(
    individual, context, num_runs: int = 50
) -> tuple[list[float], dict[str, float]]:
    """
    Benchmark full evaluation with per-constraint breakdown.

    Returns:
        (list of total times, dict of per-constraint times)
    """
    total_times = []
    constraint_times = {}

    # Pre-decode for constraint benchmarking
    sessions = decode_individual(
        individual, context.courses, context.instructors, context.groups, context.rooms
    )

    # Benchmark each constraint individually
    enabled_hard = get_enabled_hard_constraints()
    for name, info in enabled_hard.items():
        mean_time, _ = benchmark_constraint(
            name, info["function"], sessions, context.courses, num_runs=50
        )
        constraint_times[f"hard.{name}"] = mean_time

    enabled_soft = get_enabled_soft_constraints()
    for name, info in enabled_soft.items():
        mean_time, _ = benchmark_constraint(
            name, info["function"], sessions, context.courses, num_runs=50
        )
        constraint_times[f"soft.{name}"] = mean_time

    # Benchmark full evaluation
    for _ in range(num_runs):
        start = time.perf_counter()
        evaluate(
            individual,
            context.courses,
            context.instructors,
            context.groups,
            context.rooms,
        )
        elapsed = time.perf_counter() - start
        total_times.append(elapsed * 1000)

    return total_times, constraint_times


def run_benchmark(
    data_dir: Path, population_size: int = 50, num_runs: int = 50, verbose: bool = True
) -> BenchmarkResult:
    """
    Run complete benchmark suite.

    Args:
        data_dir: Path to data directory
        population_size: Number of individuals to generate
        num_runs: Number of timing runs per test
        verbose: Print progress messages

    Returns:
        BenchmarkResult with all timing data
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {data_dir.name}")
        print(f"{'=' * 60}")

    # Load context
    if verbose:
        print("Loading scheduling context...")

    # Load data
    courses_dict = load_courses(data_dir)
    groups_dict = load_groups(data_dir)
    instructors_dict = load_instructors(data_dir)
    rooms_dict = load_rooms(data_dir)

    # Link relationships
    link_courses_and_groups(courses_dict, groups_dict)
    link_courses_and_instructors(courses_dict, instructors_dict)

    # Create time system
    qts = QuantumTimeSystem()

    # Create context
    context = SchedulingContext(
        courses=courses_dict,
        groups=groups_dict,
        instructors=instructors_dict,
        rooms=rooms_dict,
        time_system=qts,
    )

    # Generate population
    if verbose:
        print(f"Generating population (size={population_size})...")
    population = generate_course_group_aware_population(
        n=population_size,
        context=context,
        parallel=True,  # Use production settings
    )

    # Use first individual for benchmarking
    individual = population[0]
    num_sessions = len(individual)

    # Estimate num_quanta
    num_quanta = sum(len(gene.quanta) for gene in individual)

    if verbose:
        print(f"Individual: {num_sessions} sessions, ~{num_quanta} quanta")
        print()

    # Benchmark decoding
    if verbose:
        print("Benchmarking decoding...")
    decode_mean, decode_std = benchmark_decoding(individual, context, num_runs)
    if verbose:
        print(f"  Decode: {decode_mean:.3f} ± {decode_std:.3f} ms")

    # Benchmark full evaluation
    if verbose:
        print("Benchmarking full evaluation...")
    total_times, constraint_times = benchmark_full_evaluation(
        individual, context, num_runs
    )

    total_mean = statistics.mean(total_times)
    total_std = statistics.stdev(total_times)
    total_min = min(total_times)
    total_max = max(total_times)

    # Separate hard and soft constraint times
    hard_time = sum(
        t for name, t in constraint_times.items() if name.startswith("hard.")
    )
    soft_time = sum(
        t for name, t in constraint_times.items() if name.startswith("soft.")
    )

    if verbose:
        print(f"  Total: {total_mean:.3f} ± {total_std:.3f} ms")
        print(f"  Range: [{total_min:.3f}, {total_max:.3f}] ms")
        print(f"  Hard constraints: {hard_time:.3f} ms")
        print(f"  Soft constraints: {soft_time:.3f} ms")
        print()

        # Top 5 slowest constraints
        print("Top 5 slowest constraints:")
        sorted_constraints = sorted(
            constraint_times.items(), key=lambda x: x[1], reverse=True
        )
        for name, time_ms in sorted_constraints[:5]:
            print(f"  {name:40s}: {time_ms:6.3f} ms")

    return BenchmarkResult(
        dataset_size=data_dir.name,
        num_sessions=num_sessions,
        num_quanta=num_quanta,
        population_size=population_size,
        decode_time_ms=decode_mean,
        hard_constraints_time_ms=hard_time,
        soft_constraints_time_ms=soft_time,
        total_time_ms=total_mean,
        constraint_times=constraint_times,
        std_dev_ms=total_std,
        min_time_ms=total_min,
        max_time_ms=total_max,
    )


def compare_datasets(
    data_dirs: list[Path], population_size: int = 50, num_runs: int = 50
) -> list[BenchmarkResult]:
    """Run benchmarks on multiple datasets and compare."""
    results = []

    for data_dir in data_dirs:
        result = run_benchmark(
            data_dir, population_size=population_size, num_runs=num_runs, verbose=True
        )
        results.append(result)

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Dataset':<15} {'Sessions':<10} {'Decode':<10} {'Hard':<10} {'Soft':<10} {'Total':<10}"
    )
    print(f"{'-' * 80}")

    for result in results:
        print(
            f"{result.dataset_size:<15} "
            f"{result.num_sessions:<10} "
            f"{result.decode_time_ms:<10.2f} "
            f"{result.hard_constraints_time_ms:<10.2f} "
            f"{result.soft_constraints_time_ms:<10.2f} "
            f"{result.total_time_ms:<10.2f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark constraint checking performance"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory (default: data/)",
    )
    parser.add_argument(
        "--pop", type=int, default=50, help="Population size (default: 50)"
    )
    parser.add_argument(
        "--runs", type=int, default=50, help="Number of timing runs (default: 50)"
    )
    parser.add_argument(
        "--output", type=Path, help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple dataset sizes (requires archived test data)",
    )

    args = parser.parse_args()

    # Run benchmarks
    if args.compare:
        # Look for multiple dataset sizes
        data_dirs = [
            args.data_dir / "small",
            args.data_dir / "medium",
            args.data_dir / "large",
        ]
        data_dirs = [d for d in data_dirs if d.exists()]

        if not data_dirs:
            print("No dataset directories found for comparison.")
            print("Using default data directory.")
            data_dirs = [args.data_dir]

        results = compare_datasets(data_dirs, args.pop, args.runs)
    else:
        result = run_benchmark(
            args.data_dir, population_size=args.pop, num_runs=args.runs, verbose=True
        )
        results = [result]

    # Save results to JSON
    if args.output:
        output_data = {
            "benchmark_config": {
                "population_size": args.pop,
                "num_runs": args.runs,
                "data_dir": str(args.data_dir),
            },
            "results": [r.to_dict() for r in results],
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")

    print(f"\n{'=' * 80}")
    print("Benchmark complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
