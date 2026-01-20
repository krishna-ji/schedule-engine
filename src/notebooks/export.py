"""Full export functionality for notebook experiments.

Generates the same production-quality outputs as CLI runs:
- schedule.json: Full decoded schedule
- calendar.pdf: Visual calendar with color-coded sessions
- plots/constraints/: Individual constraint trend plots
- plots/nsga/: NSGA metrics (Pareto front, convergence, etc.)
- csv/: Constraint metrics, Pareto front data
- log_violations.log: Detailed violation report
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
from src.notebooks.data_loader import ScheduleData
from src.notebooks.evolution import EvolutionStats

if TYPE_CHECKING:
    from deap.base import Toolbox

    from src.entities.course import Course


def create_output_structure(output_dir: Path, mode_name: str) -> dict[str, Path]:
    """Create standardized output directory structure.

    Args:
        output_dir: Base output directory
        mode_name: Mode identifier (e.g., "mode_a", "mode_b")

    Returns:
        Dictionary with paths to subdirectories
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{mode_name}_{timestamp}"

    # Create directory structure
    paths = {
        "root": run_dir,
        "plots": run_dir / "plots",
        "constraints": run_dir / "plots" / "constraints",
        "nsga": run_dir / "plots" / "nsga",
        "csv": run_dir / "csv",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def export_schedule_json(
    population: list,
    data: ScheduleData,
    output_path: Path,
    course_lookup: dict[tuple[str, str], Course] | None = None,
) -> Path:
    """Export best solution as schedule.json with human-readable times.

    Args:
        population: Final NSGA-II population
        data: Schedule data containing entities
        output_path: Directory to save schedule.json
        course_lookup: Optional course lookup for names

    Returns:
        Path to the saved schedule.json
    """
    from src.entities.decoded_session import CourseSession

    # Get best individual
    best = min(population, key=lambda x: (x.fitness.values[0], x.fitness.values[1]))

    # Inline decode to avoid module caching issues with isinstance checks
    # This happens when modules are reloaded during notebook development
    decoded = []
    for gene in best:
        course_key = (gene.course_id, gene.course_type)
        if course_key not in data.courses:
            continue  # Skip invalid genes

        course = data.courses[course_key]
        instructor = data.instructors.get(gene.instructor_id)
        group = data.groups.get(gene.group_ids[0]) if gene.group_ids else None
        room = data.rooms.get(gene.room_id)

        if not all([course, instructor, room]):
            continue  # Skip if missing references

        session = CourseSession(
            course_id=gene.course_id,
            instructor_id=gene.instructor_id,
            group_ids=gene.group_ids,
            room_id=gene.room_id,
            session_quanta=gene.get_quanta_list(),
            required_room_features=course.required_room_features,
            course_type=gene.course_type,
            instructor=instructor,
            group=group,
            room=room,
        )
        decoded.append(session)

    # Use the exporter
    export_everything(
        decoded,
        str(output_path),
        data.qts,
        course_lookup=course_lookup if course_lookup else data.courses,
    )

    return output_path / "schedule.json"


def export_csv_data(
    stats: EvolutionStats,
    population: list,
    output_path: Path,
) -> dict[str, Path]:
    """Export evolution data as CSV files.

    Args:
        stats: Evolution statistics from run_nsga2
        population: Final population

    Returns:
        Dictionary of CSV file paths
    """
    csv_files = {}

    # Constraint metrics over generations
    # EvolutionStats uses: min_hard, avg_hard, max_hard, min_soft, avg_soft
    metrics_path = output_path / "constraint_metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("generation,min_hard,avg_hard,max_hard,min_soft,avg_soft,feasible\n")
        for i, (min_h, avg_h, max_h, min_s, avg_s, feas) in enumerate(
            zip(
                stats.min_hard,
                stats.avg_hard,
                stats.max_hard,
                stats.min_soft,
                stats.avg_soft,
                stats.feasible_count,
            )
        ):
            f.write(f"{i},{min_h},{avg_h},{max_h},{min_s},{avg_s},{feas}\n")
    csv_files["metrics"] = metrics_path

    # Pareto front (final population)
    pareto_path = output_path / "pareto_front.csv"
    with open(pareto_path, "w") as f:
        f.write("individual,hard_violations,soft_penalty\n")
        for i, ind in enumerate(population):
            f.write(f"{i},{ind.fitness.values[0]},{ind.fitness.values[1]}\n")
    csv_files["pareto"] = pareto_path

    # Population fitness history (using min values as best)
    fitness_path = output_path / "population_fitness.csv"
    with open(fitness_path, "w") as f:
        f.write("generation,min_hard,avg_hard,max_hard,min_soft,avg_soft,max_soft\n")
        for i, (min_h, avg_h, max_h, min_s, avg_s) in enumerate(
            zip(
                stats.min_hard,
                stats.avg_hard,
                stats.max_hard,
                stats.min_soft,
                stats.avg_soft,
            )
        ):
            # max_soft not tracked, use avg_soft
            f.write(f"{i},{min_h},{avg_h},{max_h},{min_s},{avg_s},{avg_s}\n")
    csv_files["fitness"] = fitness_path

    return csv_files


def export_constraint_plots(
    stats: EvolutionStats,
    population: list,
    output_path: Path,
) -> list[Path]:
    """Generate all constraint-related plots.

    Args:
        stats: Evolution statistics
        population: Final population
        output_path: Directory for plots

    Returns:
        List of generated plot paths
    """
    plots = []

    # Hard constraint trend (using min_hard from EvolutionStats)
    try:
        plot_hard_constraint_violation_over_generation(stats.min_hard, str(output_path))
        plots.append(output_path / "hard_constraint_trend.pdf")
    except Exception as e:
        print(f"  [warn] hard_constraint_trend failed: {e}")

    # Soft constraint trend (using min_soft from EvolutionStats)
    try:
        plot_soft_constraint_violation_over_generation(stats.min_soft, str(output_path))
        plots.append(output_path / "soft_constraint_trend.pdf")
    except Exception as e:
        print(f"  [warn] soft_constraint_trend failed: {e}")

    # Diversity trend - EvolutionStats doesn't have diversity, use feasible_count
    try:
        plot_diversity_trend(stats.feasible_count, str(output_path))
        plots.append(output_path / "diversity_trend.pdf")
    except Exception as e:
        print(f"  [warn] diversity_trend failed: {e}")

    return plots


def export_nsga_plots(
    stats: EvolutionStats,
    population: list,
    output_path: Path,
) -> list[Path]:
    """Generate NSGA-specific plots (Pareto front, etc.).

    Args:
        stats: Evolution statistics
        population: Final population
        output_path: Directory for plots

    Returns:
        List of generated plot paths
    """
    plots = []

    # Pareto front
    try:
        plot_pareto_front(population, str(output_path))
        plots.append(output_path / "pareto_front.pdf")
    except Exception as e:
        print(f"  [warn] pareto_front failed: {e}")

    # Convergence rate (using min_hard from EvolutionStats)
    try:
        plot_convergence_rate(stats.min_hard, str(output_path), "Hard Violations")
        plots.append(output_path / "convergence_rate.pdf")
    except Exception as e:
        print(f"  [warn] convergence_rate failed: {e}")

    return plots


def export_full_results(
    population: list,
    stats: EvolutionStats,
    data: ScheduleData,
    output_dir: Path,
    mode_name: str = "mode_a",
    course_lookup: dict[tuple[str, str], Course] | None = None,
) -> dict[str, Path | list[Path]]:
    """Generate all production-quality outputs (same as CLI).

    This is the main export function that mirrors the CLI's generate_reports().
    Call this at the end of a notebook run to get the full artifact set.

    Args:
        population: Final NSGA-II population
        stats: Evolution statistics from run_nsga2
        data: Schedule data containing entities
        output_dir: Base output directory
        mode_name: Mode identifier for folder naming
        course_lookup: Optional course lookup for human-readable names

    Returns:
        Dictionary mapping output types to their paths

    Example:
        >>> paths = export_full_results(
        ...     population=final_pop,
        ...     stats=stats,
        ...     data=data,
        ...     output_dir=OUTPUT_DIR,
        ...     mode_name="mode_a",
        ... )
        >>> print(paths["schedule"])  # Path to schedule.json
    """
    print(f" Exporting full results for {mode_name}...")

    # Create directory structure
    paths = create_output_structure(output_dir, mode_name)
    results: dict[str, Path | list[Path]] = {"output_dir": paths["root"]}

    # 1. Export schedule.json and calendar.pdf
    print("  [+] Exporting schedule and calendar...")
    try:
        schedule_path = export_schedule_json(
            population, data, paths["root"], course_lookup
        )
        results["schedule"] = schedule_path
        results["calendar"] = paths["root"] / "calendar.pdf"
        print("      ✓ schedule.json")
        print("      ✓ calendar.pdf")
    except Exception as e:
        print(f"      ✗ Schedule export failed: {e}")

    # 2. Export CSVs
    print("  [+] Exporting CSV data...")
    try:
        csv_files = export_csv_data(stats, population, paths["csv"])
        results["csv"] = list(csv_files.values())
        for name, path in csv_files.items():
            print(f"      ✓ {name}.csv")
    except Exception as e:
        print(f"      ✗ CSV export failed: {e}")

    # 3. Constraint plots
    print("  [+] Generating constraint plots...")
    try:
        constraint_plots = export_constraint_plots(
            stats, population, paths["constraints"]
        )
        results["constraint_plots"] = constraint_plots
        print(f"      ✓ {len(constraint_plots)} constraint plots")
    except Exception as e:
        print(f"      ✗ Constraint plots failed: {e}")

    # 4. NSGA plots
    print("  [+] Generating NSGA plots...")
    try:
        nsga_plots = export_nsga_plots(stats, population, paths["nsga"])
        results["nsga_plots"] = nsga_plots
        print(f"      ✓ {len(nsga_plots)} NSGA plots")
    except Exception as e:
        print(f"      ✗ NSGA plots failed: {e}")

    # 5. Summary
    print(f"\n Full export complete: {paths['root']}")
    print(f"    {paths['root'].relative_to(output_dir.parent)}")

    return results


def export_comparison_results(
    mode_results: dict[str, tuple[list, EvolutionStats]],
    output_dir: Path,
) -> Path:
    """Export comparison data from multiple mode runs.

    Args:
        mode_results: Dict mapping mode name to (population, stats) tuple
        output_dir: Output directory

    Returns:
        Path to comparison CSV
    """
    comparison_path = output_dir / "mode_comparison.csv"

    with open(comparison_path, "w") as f:
        f.write("mode,final_hard,final_soft,generations,best_gen\n")

        for mode_name, (population, stats) in mode_results.items():
            best = min(
                population, key=lambda x: (x.fitness.values[0], x.fitness.values[1])
            )
            final_hard = best.fitness.values[0]
            final_soft = best.fitness.values[1]
            ngen = len(stats.min_hard)

            # Find generation with best hard violations
            best_gen = stats.min_hard.index(min(stats.min_hard))

            f.write(f"{mode_name},{final_hard},{final_soft},{ngen},{best_gen}\n")

    print(f" Mode comparison saved: {comparison_path}")
    return comparison_path
