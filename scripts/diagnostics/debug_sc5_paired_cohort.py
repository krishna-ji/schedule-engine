"""Debug SC5 (Paired Cohort Practical Alignment) constraint evaluation.

This script provides detailed tracking of the S5/SC5 constraint logic to identify bugs
in the penalty calculation for paired cohort practical alignment.

Usage:
    uv run python scripts/diagnostics/debug_sc5_paired_cohort.py <schedule.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.experiment_a_baseline import get_config as get_baseline_config
from configs.profiles import Profile
from src.config import get_config, init_config
from src.encoder.input_encoder import derive_cohort_pairs_from_groups
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.entities import CourseSession

console = Console()


def _safe_time_to_quanta(
    qts: QuantumTimeSystem, day: str, start: str, end: str
) -> set[int]:
    """Convert a time block to quanta, returning an empty set if invalid."""

    try:
        start_q = qts.time_to_quanta(day, start)
        end_q = qts.time_to_quanta(day, end)
        return set(range(start_q, end_q))
    except ValueError as exc:
        console.print(
            f"[red]Warning:[/red] Unable to encode {day} {start}-{end}: {exc}"
        )
        return set()


def load_sessions_from_schedule(schedule_file: Path) -> list[CourseSession]:
    """Load CourseSession objects directly from schedule.json output."""

    with open(schedule_file, encoding="utf-8") as handle:
        raw_sessions: list[dict[str, Any]] = json.load(handle)

    qts = QuantumTimeSystem()
    sessions: list[CourseSession] = []

    for entry in raw_sessions:
        time_map: dict[str, list[dict[str, str]]] = entry.get("time", {})
        session_quanta: set[int] = set()

        for day, periods in time_map.items():
            for period in periods:
                start = period.get("start")
                end = period.get("end")
                if start is None or end is None:
                    continue
                session_quanta.update(_safe_time_to_quanta(qts, day, start, end))

        sessions.append(
            CourseSession(
                course_id=entry.get("course_id", ""),
                instructor_id=entry.get("instructor_id", ""),
                group_ids=list(entry.get("group_ids", [])),
                room_id=entry.get("room_id", ""),
                session_quanta=sorted(session_quanta),
                required_room_features=entry.get("required_room_features", ""),
                course_type=entry.get("course_type", "theory"),
            )
        )

    return sessions


def ensure_config_initialized() -> None:
    """Ensure global config is initialized for standalone debugging."""

    try:
        cfg = get_config()
    except RuntimeError:
        baseline_cfg = get_baseline_config(Profile.TEST)
        cfg = init_config(config_obj=baseline_cfg)

    data_dir = Path(cfg.io.data_dir)
    groups_path = data_dir / "Groups.json"
    derived_pairs = derive_cohort_pairs_from_groups(str(groups_path))

    if not cfg.time.cohort_pairs:
        cfg.time.cohort_pairs = derived_pairs
    else:
        existing = {
            tuple(sorted((left.lower(), right.lower())))
            for left, right in cfg.time.cohort_pairs
        }
        for left, right in derived_pairs:
            canonical = tuple(sorted((left.lower(), right.lower())))
            if canonical in existing:
                continue
            cfg.time.cohort_pairs.append((left, right))
            existing.add(canonical)


def analyze_sc5_constraint(sessions: list[CourseSession]) -> dict[str, Any]:
    """Analyze SC5 constraint with detailed tracking.

    Returns:
        Dictionary with analysis results including:
        - penalty: Total penalty
        - cohort_pairs: List of configured pairs
        - practical_courses: Courses found per cohort
        - quanta_analysis: Detailed quanta comparison
        - bugs_found: List of potential bugs
    """
    cfg = get_config()

    # Get cohort pairs
    cohort_pairs: list[tuple[str, str]] = getattr(
        getattr(cfg, "time", cfg), "cohort_pairs", []
    )

    console.print("\n[bold cyan]SC5 Constraint Analysis[/bold cyan]")
    console.print(f"Configured cohort pairs: {cohort_pairs}")

    # Index quanta per (course_id, course_type, group_id)
    course_group_quanta: dict[tuple[str, str, str], set[int]] = defaultdict(set)

    console.print("\n[yellow]Step 1: Building course-group-quanta index...[/yellow]")
    practical_session_count = 0

    for session in sessions:
        if session.course_type.lower() != "practical":
            continue

        practical_session_count += 1
        course_id = session.course_id
        course_type = session.course_type

        for group_id in session.group_ids:
            key = (course_id, course_type, group_id)
            course_group_quanta[key].update(session.session_quanta)

    console.print(f"  Total practical sessions: {practical_session_count}")
    console.print(f"  Unique (course, type, group) keys: {len(course_group_quanta)}")

    # Show the index
    console.print("\n[yellow]Course-Group-Quanta Index:[/yellow]")
    for (course_id, course_type, group_id), quanta in sorted(
        course_group_quanta.items()
    ):
        console.print(f"  {course_id} ({course_type}) - {group_id}: {sorted(quanta)}")

    # Analyze each cohort pair
    penalty = 0
    analysis_results: dict[str, Any] = {
        "penalty": 0,
        "cohort_pairs": cohort_pairs,
        "pair_analyses": [],
        "bugs_found": [],
    }

    console.print("\n[yellow]Step 2: Analyzing cohort pairs...[/yellow]")

    for pair_idx, (left_id, right_id) in enumerate(cohort_pairs, 1):
        console.print(
            f"\n[bold green]Pair {pair_idx}: {left_id} <-> {right_id}[/bold green]"
        )

        pair_analysis: dict[str, Any] = {
            "left": left_id,
            "right": right_id,
            "practical_courses": [],
            "penalty": 0,
        }

        # Find practical courses present for at least one side
        practical_courses: set[tuple[str, str]] = set()

        for course_id, course_type, group_id in course_group_quanta:
            if course_type.lower() != "practical":
                continue
            if group_id in (left_id, right_id):
                practical_courses.add((course_id, course_type))

        console.print(f"  Practical courses found: {len(practical_courses)}")
        for course_id, course_type in sorted(practical_courses):
            console.print(f"    - {course_id} ({course_type})")

        # Analyze each practical course
        pair_penalty = 0

        for course_id, course_type in sorted(practical_courses):
            console.print(f"\n  [cyan]Analyzing: {course_id} ({course_type})[/cyan]")

            key_left = (course_id, course_type, left_id)
            key_right = (course_id, course_type, right_id)

            # Check presence
            left_present = key_left in course_group_quanta
            right_present = key_right in course_group_quanta

            console.print(f"    {left_id} present: {left_present}")
            console.print(f"    {right_id} present: {right_present}")

            if not left_present and not right_present:
                console.print(
                    "    [yellow]→ Neither cohort has this course, skipping[/yellow]"
                )
                # BUG CHECK: Should we penalize if course was found but neither has it?
                analysis_results["bugs_found"].append(
                    {
                        "type": "missing_both_cohorts",
                        "course": course_id,
                        "pair": (left_id, right_id),
                        "note": "Course in practical_courses but neither cohort has it",
                    }
                )
                continue

            if not left_present or not right_present:
                console.print(
                    "    [yellow]→ Only one cohort has this course, skipping[/yellow]"
                )

                # BUG CHECK: Should we penalize if only one cohort has the course?
                missing_cohort = right_id if not right_present else left_id
                present_cohort = left_id if left_present else right_id
                analysis_results["bugs_found"].append(
                    {
                        "type": "missing_one_cohort",
                        "course": course_id,
                        "pair": (left_id, right_id),
                        "missing": missing_cohort,
                        "present": present_cohort,
                        "note": (
                            "One cohort has course, other doesn't - no penalty applied"
                        ),
                    }
                )
                continue

            # Both present - calculate penalty
            quanta_left = course_group_quanta[key_left]
            quanta_right = course_group_quanta[key_right]

            console.print(f"    {left_id} quanta: {sorted(quanta_left)}")
            console.print(f"    {right_id} quanta: {sorted(quanta_right)}")

            if not quanta_left and not quanta_right:
                console.print(
                    "    [yellow]→ Both have empty quanta sets, skipping[/yellow]"
                )
                # BUG CHECK: Is this even possible if they're in the index?
                analysis_results["bugs_found"].append(
                    {
                        "type": "both_empty_quanta",
                        "course": course_id,
                        "pair": (left_id, right_id),
                        "note": "Both cohorts in index but both have empty quanta",
                    }
                )
                continue

            # Calculate symmetric difference
            diff = quanta_left.symmetric_difference(quanta_right)
            course_penalty = len(diff)

            console.print(f"    Symmetric difference: {sorted(diff)}")
            console.print(f"    [bold]Penalty: {course_penalty}[/bold]")

            # Detailed breakdown
            only_left = quanta_left - quanta_right
            only_right = quanta_right - quanta_left
            both = quanta_left & quanta_right

            console.print(f"    Only {left_id}: {sorted(only_left)}")
            console.print(f"    Only {right_id}: {sorted(only_right)}")
            console.print(f"    Both cohorts: {sorted(both)}")

            if course_penalty > 0:
                console.print(
                    f"    [red]→ MISALIGNED (penalty: {course_penalty})[/red]"
                )
            else:
                console.print("    [green]→ PERFECTLY ALIGNED[/green]")

            pair_penalty += course_penalty

            pair_analysis["practical_courses"].append(
                {
                    "course_id": course_id,
                    "course_type": course_type,
                    "left_quanta": sorted(quanta_left),
                    "right_quanta": sorted(quanta_right),
                    "only_left": sorted(only_left),
                    "only_right": sorted(only_right),
                    "both": sorted(both),
                    "penalty": course_penalty,
                }
            )

        pair_analysis["penalty"] = pair_penalty
        analysis_results["pair_analyses"].append(pair_analysis)
        penalty += pair_penalty

        console.print(f"\n  [bold]Total penalty for this pair: {pair_penalty}[/bold]")

    analysis_results["penalty"] = penalty
    console.print(f"\n[bold magenta]TOTAL SC5 PENALTY: {penalty}[/bold magenta]")

    return analysis_results


def print_bugs_summary(bugs: list[dict[str, Any]]) -> None:
    """Print summary of potential bugs found."""
    if not bugs:
        console.print("\n[bold green]✓ No obvious bugs detected[/bold green]")
        return

    console.print(f"\n[bold red]⚠ Potential Issues Found: {len(bugs)}[/bold red]")

    for idx, bug in enumerate(bugs, 1):
        console.print(f"\n[yellow]Issue {idx}:[/yellow] {bug['type']}")
        console.print(f"  Course: {bug['course']}")
        console.print(f"  Pair: {bug['pair']}")
        console.print(f"  Note: {bug['note']}")
        if "missing" in bug:
            console.print(f"  Missing cohort: {bug['missing']}")
            console.print(f"  Present cohort: {bug['present']}")


def main() -> None:
    """Main entry point."""
    console.print("[bold]SC5 Paired Cohort Practical Alignment Debugger[/bold]\n")

    if len(sys.argv) <= 1:
        console.print("[red]Error: Please provide a schedule JSON file[/red]")
        console.print(
            "Usage: uv run python scripts/diagnostics/debug_sc5_paired_cohort.py <schedule.json>"
        )
        sys.exit(1)

    schedule_file = Path(sys.argv[1])
    if not schedule_file.exists():
        console.print(f"[red]Error: File not found: {schedule_file}[/red]")
        sys.exit(1)

    console.print(f"Loading schedule from: {schedule_file}")
    sessions = load_sessions_from_schedule(schedule_file)
    console.print(f"Loaded {len(sessions)} sessions from schedule output")

    ensure_config_initialized()

    # Analyze SC5
    results = analyze_sc5_constraint(sessions)

    # Print bugs summary
    print_bugs_summary(results["bugs_found"])

    # Save results
    output_file = project_root / "output" / "sc5_debug_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_file}[/green]")


if __name__ == "__main__":
    main()
