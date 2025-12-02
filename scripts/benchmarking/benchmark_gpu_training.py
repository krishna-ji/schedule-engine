"""
GPU vs CPU Training Benchmark Script

Compares RL training performance on CPU vs GPU to quantify speedup.

Usage:
    uv run python scripts/benchmark_gpu_training.py

    # Custom timesteps
    uv run python scripts/benchmark_gpu_training.py --timesteps 20000

    # Save results
    uv run python scripts/benchmark_gpu_training.py --output benchmark_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.core.types import SchedulingContext
from src.encoder.input_encoder import (
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.ga.population import generate_course_group_aware_population
from src.rl.agents import create_ppo_agent
from src.rl.gym_env.schedule_env import create_schedule_env


def load_context(data_dir: str = "data") -> SchedulingContext:
    """Load scheduling context from data directory."""
    print(f"Loading data from {data_dir}...")

    courses = load_courses(data_dir)
    groups = load_groups(data_dir)
    instructors = load_instructors(data_dir)
    rooms = load_rooms(data_dir)

    link_courses_and_groups(courses, groups)
    link_courses_and_instructors(courses, instructors)

    qts = QuantumTimeSystem()

    context = SchedulingContext(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        time_system=qts,
    )

    print(f"  ✓ Loaded {len(courses)} courses, {len(instructors)} instructors")
    return context


def benchmark_device(
    device: str,
    timesteps: int = 10000,
    context: SchedulingContext | None = None,
    verbose: bool = True,
) -> dict:
    """
    Benchmark RL training on specific device.

    Args:
        device: "cpu" or "cuda"
        timesteps: Number of training timesteps
        context: Scheduling context (will load if None)
        verbose: Print progress

    Returns:
        Dictionary with timing results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {device.upper()} Training ({timesteps:,} timesteps)")
        print(f"{'=' * 60}")

    # Load context if not provided
    if context is None:
        context = load_context()

    # Create environment
    if verbose:
        print("Creating environment...")
    population = generate_course_group_aware_population(
        n=50, context=context, parallel=False
    )
    env = create_schedule_env(
        initial_population=population,
        context=context,
        max_steps_per_episode=20,
        fast_evaluation=True,
    )

    # Create agent
    if verbose:
        print(f"Creating PPO agent on {device}...")
    agent = create_ppo_agent(
        env=env,
        device=device,
        verbose=0,
    )

    # Benchmark training
    if verbose:
        print(f"Training for {timesteps:,} timesteps...")

    start_time = time.time()
    agent.learn(
        total_timesteps=timesteps,
        progress_bar=verbose,
    )
    elapsed = time.time() - start_time

    # Calculate metrics
    steps_per_second = timesteps / elapsed

    results = {
        "device": device,
        "timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
        "steps_per_second": steps_per_second,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Results for {device.upper()}:")
        print(f"  Total time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        print(f"  Speed: {steps_per_second:.1f} steps/sec")
        print(f"{'=' * 60}")

    return results


def print_comparison(cpu_results: dict, gpu_results: dict):
    """Print comparison table."""
    speedup = cpu_results["elapsed_seconds"] / gpu_results["elapsed_seconds"]
    time_saved = cpu_results["elapsed_seconds"] - gpu_results["elapsed_seconds"]
    percent_faster = (
        1 - gpu_results["elapsed_seconds"] / cpu_results["elapsed_seconds"]
    ) * 100

    print(f"\n{'=' * 60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Device':<15} {'Time (s)':<12} {'Time (min)':<12} {'Steps/sec':<12}")
    print(f"{'-' * 60}")
    print(
        f"{'CPU':<15} {cpu_results['elapsed_seconds']:<12.2f} "
        f"{cpu_results['elapsed_minutes']:<12.2f} "
        f"{cpu_results['steps_per_second']:<12.1f}"
    )
    print(
        f"{'GPU (CUDA)':<15} {gpu_results['elapsed_seconds']:<12.2f} "
        f"{gpu_results['elapsed_minutes']:<12.2f} "
        f"{gpu_results['steps_per_second']:<12.1f}"
    )
    print(f"{'-' * 60}")
    print(f"{'Speedup':<15} {speedup:.2f}×")
    print(f"{'Time Saved':<15} {time_saved:.2f}s ({time_saved / 60:.2f} min)")
    print(f"{'Percent Faster':<15} {percent_faster:.1f}%")
    print(f"{'=' * 60}")

    # Extrapolate to full training
    print("\nEXTRAPOLATION TO FULL TRAINING:")
    print(f"{'-' * 60}")

    training_scenarios = [
        ("Quick test", 50000),
        ("Curriculum phase", 100000),
        ("Full training", 300000),
    ]

    for name, steps in training_scenarios:
        cpu_time = (steps / cpu_results["steps_per_second"]) / 60
        gpu_time = (steps / gpu_results["steps_per_second"]) / 60
        saved = cpu_time - gpu_time
        print(f"{name:20s} ({steps:6,} steps):")
        print(
            f"  CPU: {cpu_time:6.1f} min  →  GPU: {gpu_time:6.1f} min  "
            f"(saves {saved:6.1f} min)"
        )

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU for RL training")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Number of training timesteps (default: 10000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--output", type=Path, help="Save results to JSON file (optional)"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Only benchmark CPU (skip GPU)"
    )
    parser.add_argument(
        "--gpu-only", action="store_true", help="Only benchmark GPU (skip CPU)"
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("GPU vs CPU Training Benchmark")
    print(f"{'=' * 60}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Data directory: {args.data_dir}")

    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("️  CUDA not available - GPU benchmark will be skipped")

    # Load context once (reuse for both benchmarks)
    context = load_context(args.data_dir)

    results = {}

    # CPU Benchmark
    if not args.gpu_only:
        cpu_results = benchmark_device(
            device="cpu", timesteps=args.timesteps, context=context, verbose=True
        )
        results["cpu"] = cpu_results

    # GPU Benchmark
    if gpu_available and not args.cpu_only:
        gpu_results = benchmark_device(
            device="cuda", timesteps=args.timesteps, context=context, verbose=True
        )
        results["gpu"] = gpu_results

    # Comparison
    if "cpu" in results and "gpu" in results:
        print_comparison(results["cpu"], results["gpu"])
    elif not gpu_available:
        print(f"\n{'=' * 60}")
        print("️  GPU benchmark skipped - CUDA not available")
        print(f"{'=' * 60}")
        print("\nTo enable GPU:")
        print("1. Install NVIDIA drivers")
        print("2. Install PyTorch with CUDA:")
        print(
            "   uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    # Save results
    if args.output:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timesteps": args.timesteps,
            "gpu_available": gpu_available,
            "results": results,
        }

        if "cpu" in results and "gpu" in results:
            output_data["speedup"] = (
                results["cpu"]["elapsed_seconds"] / results["gpu"]["elapsed_seconds"]
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to: {args.output}")

    print(f"\n{'=' * 60}")
    print("Benchmark Complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
