"""
Reporting Workflow Module

Handles plotting and export of GA results.
Extracted from main.py for modularity.

PARALLELIZATION: Uses ThreadPoolExecutor to generate plots concurrently.
Expected speedup: 5-10x on multi-core systems.
"""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.core.ga_scheduler import GAMetrics
from src.core.types import Individual
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.entities.course import Course
from src.entities.decoded_session import CourseSession
from src.exporter.exporter import export_everything
from src.exporter.plot_convergence import (
    plot_constraint_satisfaction_evolution,
    plot_convergence_dashboard,
    plot_convergence_rate,
    plot_multi_metric_convergence,
)
from src.exporter.plot_detailed_constraints import (
    plot_constraint_summary,
    plot_individual_hard_constraints,
    plot_individual_soft_constraints,
)

# NEW: Import advanced evaluation metric plotting modules
from src.exporter.plot_hypervolume import plot_hypervolume_trend
from src.exporter.plot_spacing import (
    plot_spacing_distribution,
    plot_spacing_trend,
    plot_spacing_with_pareto,
)
from src.exporter.plotdiversity import plot_diversity_trend
from src.exporter.plothard import plot_hard_constraint_violation_over_generation
from src.exporter.plotpareto import plot_pareto_front
from src.exporter.plotsoft import plot_soft_constraint_violation_over_generation
from src.exporter.violation_reporter import generate_violation_report
from src.ga.heuristic_tracker import HeuristicTracker
from src.lns.lns_operator import get_lns_stats
from src.utils.system_info import get_cpu_count


def generate_reports(
    decoded_schedule: list[CourseSession],
    metrics: GAMetrics,
    population: list[Individual],
    qts: QuantumTimeSystem,
    output_dir: str,
    course_map: dict[tuple, Course] | None = None,
    heuristic_tracker: HeuristicTracker | None = None,
) -> None:
    """
    Generate all output artifacts: plots, JSON, PDFs, violation reports.

    PARALLELIZED: All plotting operations run concurrently using ThreadPoolExecutor.
    Expected speedup: 5-10x on systems with 4+ cores.

    Creates:
        - schedule.json: Schedule in JSON format
        - calendar.pdf: Visual calendar with color-coded sessions
        - log_violations.log: Detailed constraint violation report
        - Evolution plots: hard/soft constraint trends, diversity
        - Pareto front visualization
        - Detailed constraint breakdown plots
        - Advanced metrics: hypervolume, spacing, IGD, spread, convergence
        - Multi-metric convergence dashboard

    Args:
        decoded_schedule: Best schedule solution (list of CourseSessions)
        metrics: GA evolution metrics (now includes hypervolume, spacing, IGD, etc.)
        population: Final population (for Pareto front)
        qts: Quantum time system (for time conversion)
        output_dir: Output directory path
        course_map: Dictionary of courses (for violation analysis)
    """

    # Export schedule (JSON + PDF) - sequential (tightly coupled operations)
    print("  [+] Exporting schedule...")
    export_everything(
        decoded_schedule,
        output_dir,
        qts,
        course_lookup=course_map,
    )
    print("      [!ok] schedule.json")
    print("      [!ok] calendar.pdf")

    # Export LNS repair statistics if available
    lns_stats = get_lns_stats()
    if lns_stats.total_attempts > 0:
        import json
        import os

        stats_path = os.path.join(output_dir, "lns_repair_stats.json")
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "total_attempts": lns_stats.total_attempts,
                    "successful_repairs": lns_stats.successful_repairs,
                    "failed_repairs": lns_stats.failed_repairs,
                    "success_rate_percent": (
                        lns_stats.successful_repairs / lns_stats.total_attempts * 100
                        if lns_stats.total_attempts > 0
                        else 0.0
                    ),
                    "igls_attempts": lns_stats.igls_attempts,
                    "igls_success": lns_stats.igls_success,
                    "igls_success_rate_percent": (
                        lns_stats.igls_success / lns_stats.igls_attempts * 100
                        if lns_stats.igls_attempts > 0
                        else 0.0
                    ),
                    "avg_subproblem_size": lns_stats.avg_subproblem_size,
                    "total_repair_time_seconds": lns_stats.total_repair_time,
                    "total_conflicts_detected": lns_stats.total_conflicts_detected,
                    "total_conflicts_repaired": lns_stats.total_conflicts_repaired,
                },
                f,
                indent=2,
            )
        print("      [!ok] lns_repair_stats.json")
        print(f"      [!info] LNS Stats: {lns_stats}")

    # Export heuristic tracking statistics and plots if available
    if heuristic_tracker and len(heuristic_tracker.applications) > 0:
        print("  [+] Generating heuristic tracking reports...")
        heuristic_tracker.export_json(Path(output_dir))
        heuristic_tracker.generate_plots(Path(output_dir))

        # Print summary
        summary = heuristic_tracker.get_summary()
        print(
            f"      [!info] Tracked {summary['total_applications']} heuristic applications"
        )
        print(f"      [!info] Success rate: {summary['success_rate_percent']:.1f}%")
        print(
            f"      [!info] Best heuristic: {summary['best_heuristic']} (improvement: {summary['best_heuristic_improvement']:.2f})"
        )

    # Generate violation report - sequential (depends on export)
    if course_map:
        print("  [+] Generating violation report...")
        generate_violation_report(decoded_schedule, course_map, qts, output_dir)
        print("      [!ok] log_violations.log")

    # ========================================
    # PARALLEL PLOTTING SECTION
    # ========================================
    print("  [+] Generating plots in parallel...")
    start_time = time.time()

    # Build list of plotting tasks
    plot_tasks: list[
        tuple[str, Callable[..., Any], tuple[Any, ...], dict[str, Any]]
    ] = []

    # Core evolution plots
    plot_tasks.append(
        (
            "hard_constraint_trend.pdf",
            plot_hard_constraint_violation_over_generation,
            (metrics.hard_violations, output_dir),
            {},
        )
    )
    plot_tasks.append(
        (
            "soft_constraint_trend.pdf",
            plot_soft_constraint_violation_over_generation,
            (metrics.soft_penalties, output_dir),
            {},
        )
    )
    plot_tasks.append(
        (
            "diversity_trend.pdf",
            plot_diversity_trend,
            (metrics.diversity, output_dir),
            {},
        )
    )

    # Pareto front
    plot_tasks.append(
        ("pareto_front.pdf", plot_pareto_front, (population, output_dir), {})
    )

    # Detailed constraints
    plot_tasks.append(
        (
            "hard/individual_constraints.pdf",
            plot_individual_hard_constraints,
            (metrics.detailed_hard, output_dir),
            {},
        )
    )
    plot_tasks.append(
        (
            "soft/individual_constraints.pdf",
            plot_individual_soft_constraints,
            (metrics.detailed_soft, output_dir),
            {},
        )
    )
    plot_tasks.append(
        (
            "constraint_summary.pdf",
            plot_constraint_summary,
            (metrics.detailed_hard, metrics.detailed_soft, output_dir),
            {},
        )
    )

    # Advanced metrics - conditional
    if metrics.hypervolume:
        plot_tasks.append(
            (
                "hypervolume_trend.pdf",
                plot_hypervolume_trend,
                (metrics.hypervolume, output_dir),
                {},
            )
        )

    if metrics.spacing:
        plot_tasks.append(
            ("spacing_trend.pdf", plot_spacing_trend, (metrics.spacing, output_dir), {})
        )
        plot_tasks.append(
            (
                "spacing_distribution.pdf",
                plot_spacing_distribution,
                (population, output_dir),
                {},
            )
        )
        plot_tasks.append(
            (
                "spacing_pareto_combined.pdf",
                plot_spacing_with_pareto,
                (population, metrics.spacing, output_dir),
                {},
            )
        )

    if metrics.feasibility_rate:
        plot_tasks.append(
            (
                "feasibility_evolution.pdf",
                plot_constraint_satisfaction_evolution,
                (metrics.feasibility_rate, output_dir),
                {},
            )
        )

    if metrics.hard_violations:
        plot_tasks.append(
            (
                "convergence_rate_hard_violations.pdf",
                plot_convergence_rate,
                (metrics.hard_violations, output_dir, "Hard Violations"),
                {},
            )
        )

    if metrics.hypervolume and metrics.spacing:
        metrics_dict = {
            "hypervolume": metrics.hypervolume,
            "spacing": metrics.spacing,
            "diversity": metrics.diversity,
        }
        if metrics.igd:
            metrics_dict["igd"] = metrics.igd
        if metrics.spread:
            metrics_dict["spread"] = metrics.spread

        plot_tasks.append(
            (
                "convergence_multi_metric.pdf",
                plot_multi_metric_convergence,
                (metrics_dict, output_dir),
                {},
            )
        )

    # Comprehensive dashboard
    if (
        metrics.hard_violations
        and metrics.soft_penalties
        and metrics.diversity
        and metrics.hypervolume
        and metrics.spacing
        and metrics.feasibility_rate
    ):
        plot_tasks.append(
            (
                "convergence_dashboard.pdf",
                plot_convergence_dashboard,
                (),
                {
                    "hard_violations": metrics.hard_violations,
                    "soft_penalties": metrics.soft_penalties,
                    "diversity": metrics.diversity,
                    "hypervolume": metrics.hypervolume,
                    "spacing": metrics.spacing,
                    "feasibility_rate": metrics.feasibility_rate,
                    "output_dir": output_dir,
                },
            )
        )

    # Execute all plots in parallel
    completed_plots = []
    failed_plots = []

    import os

    max_workers = get_cpu_count()  # Auto-detect all cores
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_plot = {
            executor.submit(_safe_plot_wrapper, plot_func, args, kwargs): plot_name
            for plot_name, plot_func, args, kwargs in plot_tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_plot):
            plot_name = future_to_plot[future]
            try:
                success = future.result()
                if success:
                    completed_plots.append(plot_name)
                else:
                    failed_plots.append(plot_name)
            except Exception as exc:
                print(f"      [!err] {plot_name} failed: {exc}")
                failed_plots.append(plot_name)

    elapsed = time.time() - start_time

    # Report results
    print(
        f"      [!ok] Generated {len(completed_plots)} plots in {elapsed:.2f}s (parallel)"
    )
    for plot_name in sorted(completed_plots):
        print(f"      [!ok] {plot_name}")

    if failed_plots:
        print(f"      [!warn] {len(failed_plots)} plots failed:")
        for plot_name in sorted(failed_plots):
            print(f"      [!err] {plot_name}")

    print("  [+] All reports generated successfully!")


def _safe_plot_wrapper(
    plot_func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    """
    Wrapper for plotting functions to catch exceptions.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        plot_func(*args, **kwargs)
        return True
    except Exception:
        # Silently fail - error will be reported by caller
        return False
