"""Shared feasibility checks for run scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging

from src.io.feasibility import (
    FeasibilityReport,
    check_feasibility,
    generate_feasibility_report_file,
)


def _log_capacity_warnings(report: FeasibilityReport, logger: logging.Logger) -> None:
    """Log capacity shortages and pigeonhole violations in a compact, readable form."""
    failed = report.get_failed_checks()
    if not failed:
        return

    for result in failed:
        details = result.details or {}

        if result.check_name == "Instructor Workload vs Availability":
            demand = details.get("total_demand_quanta")
            supply = details.get("total_supply_quanta")
            shortage = details.get("shortage_quanta")
            logger.warning(
                "Instructor capacity shortfall: demand=%s, supply=%s, shortage=%s quanta",
                demand,
                supply,
                shortage,
            )

        elif result.check_name == "Room Capacity Bottleneck":
            demand = details.get("total_student_hours")
            supply = details.get("total_seat_hours")
            shortage = details.get("shortage")
            largest_class = details.get("largest_class_size")
            largest_room = details.get("largest_room_capacity")
            logger.warning(
                "Room capacity shortfall: demand=%s, supply=%s, shortage=%s seat-hours",
                demand,
                supply,
                shortage,
            )
            logger.warning(
                "Largest class vs room: class=%s, largest_room=%s",
                largest_class,
                largest_room,
            )

        elif result.check_name == "Instructor Qualification Bottleneck":
            bottlenecks = details.get("bottlenecks", [])
            logger.warning(
                "Qualification bottlenecks: %s course(s) lack qualified capacity",
                len(bottlenecks),
            )
            for b in bottlenecks[:5]:
                course_name = b.get("course_name")
                course_display = b.get("course_display", b.get("course_key"))
                logger.warning(
                    "  %s (%s): demand=%s, supply=%s, shortage=%s",
                    course_name,
                    course_display,
                    b.get("demand"),
                    b.get("supply"),
                    b.get("shortage"),
                )

        elif result.check_name == "Room Feature Bottleneck":
            bottlenecks = details.get("bottlenecks", [])
            logger.warning(
                "Room feature bottlenecks: %s feature(s) short on capacity",
                len(bottlenecks),
            )
            for b in bottlenecks[:5]:
                logger.warning(
                    "  feature=%s: demand=%s, supply=%s, shortage=%s",
                    b.get("feature"),
                    b.get("demand"),
                    b.get("supply"),
                    b.get("shortage"),
                )

        elif result.check_name == "Specific Lab Feature Availability":
            missing = details.get("missing_features", [])
            logger.warning(
                "Specific lab feature check: %s feature(s) missing from rooms",
                details.get("missing_count", len(missing)),
            )
            for m in missing[:5]:
                logger.warning(
                    "  feature='%s': needed by %s course(s), %s quanta",
                    m.get("feature"),
                    m.get("required_by_courses"),
                    m.get("total_quanta_demand"),
                )

        elif result.check_name == "Group Pigeonhole Problem":
            overloaded = details.get("details", [])
            logger.warning(
                "Group pigeonhole violations: %s group(s) overloaded (max util=%.1f%%)",
                details.get("overloaded_groups", len(overloaded)),
                float(details.get("max_utilization", 0.0)),
            )
            for g in overloaded[:5]:
                logger.warning(
                    "  %s (%s): demand=%s, available=%s, util=%.0f%%",
                    g.get("group_name"),
                    g.get("group_id"),
                    g.get("demand"),
                    g.get("available"),
                    float(g.get("utilization", 0.0)),
                )


def _safe_for_console(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def run_feasibility_checks(
    data: Any,
    output_dir: Path | str,
    logger: logging.Logger,
    expected_quanta: int = 42,
    force_report: bool = True,
) -> tuple[bool, FeasibilityReport]:
    """Run feasibility checks, enforce fixed quanta, and persist a report."""
    output_dir = Path(output_dir)

    total_operating_quanta = len(data.qts.get_all_operating_quanta())
    if total_operating_quanta != expected_quanta:
        msg = (
            f"Operating quanta={total_operating_quanta}, expected {expected_quanta}. "
            "Hours must not be extended."
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        "Operating quanta: %d (expected %d)",
        total_operating_quanta,
        expected_quanta,
    )

    # Feasibility defaults (FeasibilityConfig removed — values inlined)
    _fail_on_infeasibility = True
    _generate_report = True
    _save_report_on_success = False

    is_feasible, report = check_feasibility(
        data.courses, data.instructors, data.rooms, data.groups, data.qts
    )

    # Always define report_path
    report_path = output_dir / "feasibility.log"

    # Write report if forced, configured, or infeasible (always save issues)
    should_write = (
        force_report
        or (_generate_report and (is_feasible or _save_report_on_success))
        or not is_feasible
    )  # Always write when infeasible

    if should_write:
        generate_feasibility_report_file(report, str(report_path))
        logger.info("Feasibility report saved: %s", report_path)

    summary = report.summary or {}
    logger.info(
        "Feasibility: %s (passed=%s, failed=%s, critical=%s)",
        summary.get("status", "unknown"),
        summary.get("passed", 0),
        summary.get("failed", 0),
        summary.get("critical_failures", 0),
    )

    _log_capacity_warnings(report, logger)

    if not is_feasible:
        if _fail_on_infeasibility:
            # Show error panel to console (same as was shown in check_feasibility)
            from rich import box
            from rich.panel import Panel

            from src.utils.console_service import get_console

            console = get_console()
            critical_count = summary.get("critical_failures", 0)
            console.print()
            console.print(
                Panel(
                    "[bold red][!ERR] PROBLEM IS INFEASIBLE[/bold red]\n\n"
                    f"Found {critical_count} critical issue(s) that make this problem unsolvable.\n"
                    "Please review the detailed report above and fix the identified issues.\n\n"
                    f"[dim]Feasibility report saved to: {report_path}[/dim]\n"
                    "[dim]Infeasibility detected. Review the report above and fix the identified issues.[/dim]",
                    border_style="red",
                    box=box.DOUBLE,
                )
            )
            console.print()
            console.print("[bold red] Exiting program...[/bold red]\n")
            logger.error("Problem is infeasible; stopping run (see feasibility.log).")
            raise SystemExit(1)
        logger.warning("Proceeding despite infeasibility (see feasibility.log).")

    return is_feasible, report
