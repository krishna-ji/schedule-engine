"""
Reporting Workflow Module

Handles plotting and export of GA results.
Extracted from main.py for modularity.
"""

from typing import List, Dict
from src.entities.decoded_session import CourseSession
from src.entities.course import Course
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.core.ga_scheduler import GAMetrics
from src.exporter.exporter import export_everything
from src.exporter.violation_reporter import generate_violation_report
from src.exporter.plotdiversity import plot_diversity_trend
from src.exporter.plothard import plot_hard_constraint_violation_over_generation
from src.exporter.plotsoft import plot_soft_constraint_violation_over_generation
from src.exporter.plotpareto import plot_pareto_front
from src.exporter.plot_detailed_constraints import (
    plot_individual_hard_constraints,
    plot_individual_soft_constraints,
    plot_constraint_summary,
)

# NEW: Import advanced evaluation metric plotting modules
from src.exporter.plot_hypervolume import plot_hypervolume_trend
from src.exporter.plot_spacing import (
    plot_spacing_trend,
    plot_spacing_distribution,
    plot_spacing_with_pareto,
)
from src.exporter.plot_convergence import (
    plot_multi_metric_convergence,
    plot_convergence_dashboard,
    plot_convergence_rate,
    plot_constraint_satisfaction_evolution,
)


def generate_reports(
    decoded_schedule: List[CourseSession],
    metrics: GAMetrics,
    population: List,
    qts: QuantumTimeSystem,
    output_dir: str,
    course_map: Dict[tuple, Course] = None,
):
    """
    Generate all output artifacts: plots, JSON, PDFs, violation reports.

    Creates:
        - schedule.json: Schedule in JSON format
        - schedule.pdf: Visual calendar with color-coded sessions
        - violation_report.txt: Detailed constraint violation report
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

    # Export schedule (JSON + PDF)
    print("  [+] Exporting schedule...")
    export_everything(decoded_schedule, output_dir, qts)
    print("      [OK] schedule.json")
    print("      [OK] schedule.pdf")

    # Generate violation report
    if course_map:
        print("  [+] Generating violation report...")
        generate_violation_report(decoded_schedule, course_map, qts, output_dir)
        print("      [OK] violation_report.txt")

    # Plot evolution trends
    print("  [+] Generating evolution plots...")
    plot_hard_constraint_violation_over_generation(metrics.hard_violations, output_dir)
    print("      [OK] hard_constraint_trend.pdf")

    plot_soft_constraint_violation_over_generation(metrics.soft_penalties, output_dir)
    print("      [OK] soft_constraint_trend.pdf")

    plot_diversity_trend(metrics.diversity, output_dir)
    print("      [OK] diversity_trend.pdf")

    # Plot Pareto front
    print("  [+] Generating Pareto front plot...")
    plot_pareto_front(population, output_dir)
    print("      [OK] pareto_front.pdf")

    # Plot detailed constraints
    print("  [+] Generating detailed constraint plots...")
    plot_individual_hard_constraints(metrics.detailed_hard, output_dir)
    print("      [OK] hard/individual_constraints.pdf")

    plot_individual_soft_constraints(metrics.detailed_soft, output_dir)
    print("      [OK] soft/individual_constraints.pdf")

    plot_constraint_summary(metrics.detailed_hard, metrics.detailed_soft, output_dir)
    print("      [OK] constraint_summary.pdf")

    # NEW: Generate advanced evaluation metric plots
    print("  [+] Generating advanced evaluation metrics...")

    # Phase 1: Essential metrics
    if metrics.hypervolume:
        plot_hypervolume_trend(metrics.hypervolume, output_dir)
        print("      [OK] hypervolume_trend.pdf")

    if metrics.spacing:
        plot_spacing_trend(metrics.spacing, output_dir)
        print("      [OK] spacing_trend.pdf")

        # Spacing distribution for final population
        plot_spacing_distribution(population, output_dir)
        print("      [OK] spacing_distribution.pdf")

        # Combined spacing + Pareto front view
        plot_spacing_with_pareto(population, metrics.spacing, output_dir)
        print("      [OK] spacing_pareto_combined.pdf")

    if metrics.feasibility_rate:
        plot_constraint_satisfaction_evolution(metrics.feasibility_rate, output_dir)
        print("      [OK] feasibility_evolution.pdf")

    # Convergence rate analysis
    if metrics.hard_violations:
        plot_convergence_rate(metrics.hard_violations, output_dir, "Hard Violations")
        print("      [OK] convergence_rate_hard_violations.pdf")

    # Phase 2: Multi-metric convergence visualization
    if metrics.hypervolume and metrics.spacing:
        metrics_dict = {
            "hypervolume": metrics.hypervolume,
            "spacing": metrics.spacing,
            "diversity": metrics.diversity,
        }

        # Add IGD and spread if available
        if metrics.igd:
            metrics_dict["igd"] = metrics.igd
        if metrics.spread:
            metrics_dict["spread"] = metrics.spread

        plot_multi_metric_convergence(metrics_dict, output_dir)
        print("      [OK] convergence_multi_metric.pdf")

    # Comprehensive dashboard
    if (
        metrics.hard_violations
        and metrics.soft_penalties
        and metrics.diversity
        and metrics.hypervolume
        and metrics.spacing
        and metrics.feasibility_rate
    ):
        plot_convergence_dashboard(
            hard_violations=metrics.hard_violations,
            soft_penalties=metrics.soft_penalties,
            diversity=metrics.diversity,
            hypervolume=metrics.hypervolume,
            spacing=metrics.spacing,
            feasibility_rate=metrics.feasibility_rate,
            output_dir=output_dir,
        )
        print("      [OK] convergence_dashboard.pdf")

    print("  [+] All reports generated successfully!")
