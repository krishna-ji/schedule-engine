"""
Export Utilities for Notebooks.

Provides production-like export functionality for notebook experiments:
- Schedule JSON export
- Calendar PDF generation
- CSV data export
- Comprehensive result bundles

Mirrors CLI production output structure for consistency.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from schedule_engine.domain.gene import SessionGene
from schedule_engine.io.decoder import decode_individual
from schedule_engine.notebooks.core import EvolutionStats, NotebookData, get_best_individual

__all__ = [
    "export_schedule_json",
    "export_stats_csv",
    "export_full_results",
]


def export_schedule_json(
    individual: list[SessionGene],
    data: NotebookData,
    output_path: Path | str,
) -> Path:
    """
    Export decoded schedule to JSON file.

    Args:
        individual: Best individual to export
        data: NotebookData for decoding
        output_path: Path to save JSON file

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Decode individual to CourseSession objects
    sessions = decode_individual(
        individual,
        data.courses,
        data.instructors,
        data.groups,
        data.rooms,
    )

    # Convert to serializable format
    schedule_data = []
    for session in sessions:
        session_dict = {
            "course_id": session.course_id,
            "course_type": session.course_type,
            "instructor_id": (
                session.instructor.instructor_id if session.instructor else None
            ),
            "instructor_name": session.instructor.name if session.instructor else None,
            "room_id": session.room.room_id if session.room else None,
            "room_name": session.room.name if session.room else None,
            "groups": session.group_ids,  # group_ids is the list
            "start_quanta": session.session_quanta[0] if session.session_quanta else 0,
            "num_quanta": len(session.session_quanta),
            "day": _quanta_to_day(
                session.session_quanta[0] if session.session_quanta else 0, data
            ),
            "start_time": _quanta_to_time(
                session.session_quanta[0] if session.session_quanta else 0, data
            ),
            "end_time": _quanta_to_time(
                (session.session_quanta[-1] + 1) if session.session_quanta else 0, data
            ),
        }
        schedule_data.append(session_dict)

    # Sort by day/time for readability
    schedule_data.sort(key=lambda x: (x["day"], x["start_quanta"]))

    # Export
    export_obj = {
        "generated_at": datetime.now().isoformat(),
        "total_sessions": len(schedule_data),
        "schedule": schedule_data,
    }

    with open(output_path, "w") as f:
        json.dump(export_obj, f, indent=2)

    print(f" Saved: {output_path}")
    return output_path


def _quanta_to_day(quanta: int, data: NotebookData) -> str:
    """Convert quanta index to day name."""
    try:
        day, _ = data.qts.quanta_to_time(quanta)
        return day
    except (ValueError, KeyError):
        return "Unknown"


def _quanta_to_time(quanta: int, data: NotebookData) -> str:
    """Convert quanta index to time string."""
    try:
        _, time_str = data.qts.quanta_to_time(quanta)
        return time_str
    except (ValueError, KeyError):
        return "00:00"


def export_stats_csv(
    stats: EvolutionStats,
    output_path: Path | str,
) -> Path:
    """
    Export evolution statistics to CSV.

    Args:
        stats: EvolutionStats object
        output_path: Path to save CSV file

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(output_path, "w") as f:
        # Header
        f.write(
            "generation,min_hard,avg_hard,max_hard,min_soft,avg_soft,feasible_count\n"
        )

        # Data rows
        for i, gen in enumerate(stats.generations):
            f.write(
                f"{gen},"
                f"{stats.min_hard[i]:.1f},"
                f"{stats.avg_hard[i]:.1f},"
                f"{stats.max_hard[i]:.1f},"
                f"{stats.min_soft[i]:.1f},"
                f"{stats.avg_soft[i]:.1f},"
                f"{stats.feasible_count[i]}\n"
            )

    print(f" Saved: {output_path}")
    return output_path


def export_full_results(
    population: list[Any],
    stats: EvolutionStats,
    data: NotebookData,
    output_dir: Path | str,
    mode_name: str = "experiment",
) -> dict[str, Path]:
    """
    Export comprehensive results (schedule JSON + stats CSV + summary).

    Args:
        population: Final population
        stats: Evolution statistics
        data: NotebookData for decoding
        output_dir: Output directory
        mode_name: Name for file prefixes

    Returns:
        Dict mapping result type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {"output_dir": output_dir}

    # Get best individual
    best = get_best_individual(population)

    # Export schedule JSON
    schedule_path = output_dir / f"{mode_name}_schedule.json"
    export_schedule_json(best, data, schedule_path)
    paths["schedule"] = schedule_path

    # Export stats CSV
    stats_path = output_dir / f"{mode_name}_stats.csv"
    export_stats_csv(stats, stats_path)
    paths["stats"] = stats_path

    # Export summary JSON
    summary_path = output_dir / f"{mode_name}_summary.json"
    _export_summary(population, stats, data, summary_path)
    paths["summary"] = summary_path

    print(f"\n All exports complete: {output_dir}")
    return paths


def _export_summary(
    population: list[Any],
    stats: EvolutionStats,
    data: NotebookData,
    output_path: Path,
) -> None:
    """Export experiment summary to JSON."""
    best = min(
        population, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
    )
    hard_vals = [ind.fitness.values[0] for ind in population]
    soft_vals = [ind.fitness.values[1] for ind in population]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "evolution": {
            "generations": len(stats.generations),
            "population_size": len(population),
            "elapsed_time_seconds": stats.elapsed_time,
        },
        "best_solution": {
            "hard_violations": best.fitness.values[0],
            "soft_penalty": best.fitness.values[1],
            "feasible": best.fitness.values[0] == 0,
        },
        "final_population": {
            "feasible_count": sum(1 for h in hard_vals if h == 0),
            "min_hard": min(hard_vals),
            "avg_hard": sum(hard_vals) / len(hard_vals),
            "min_soft": min(soft_vals),
            "avg_soft": sum(soft_vals) / len(soft_vals),
        },
        "data": {
            "courses": len(data.courses),
            "groups": len(data.groups),
            "instructors": len(data.instructors),
            "rooms": len(data.rooms),
            "total_quanta": data.qts.total_quanta,
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f" Saved: {output_path}")
