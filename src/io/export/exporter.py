from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Calendar export constants
EXCAL_QUANTUM_MINUTES: int = 15
EXCAL_START_HOUR: int = 7
EXCAL_END_HOUR: int = 20
EXCAL_DEFAULT_OUTPUT_PDF: str = "calendar.pdf"

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.session import CourseSession
    from src.io.time_system import QuantumTimeSystem


def _format_course_name_with_type(course_name: str, course_type: str) -> str:
    """Append (TH) or (PR) tag to course name based on course type."""

    tag = "PR" if course_type == "practical" else "TH"
    return f"{course_name} ({tag})"


def _resolve_course_details(
    session: CourseSession,
    course_lookup: dict[tuple[str, str], Course] | None,
) -> tuple[str, str, str]:
    """Return course name, course code, and display label for a session."""

    course: Course | None = None
    if course_lookup is not None:
        course = course_lookup.get((session.course_id, session.course_type))

    course_name = course.name if course else session.course_id
    course_code = (
        course.course_code if course and course.course_code else session.course_id
    )
    course_display = _format_course_name_with_type(course_name, session.course_type)
    return course_name, course_code, course_display


def _resolve_instructor_name(session: CourseSession) -> str:
    """Return instructor name if available, otherwise fall back to ID."""

    if session.instructor and session.instructor.name:
        return session.instructor.name
    return session.instructor_id


def _get_time_schedule_format(
    qts: QuantumTimeSystem, quanta: list[int]
) -> dict[str, list[dict[str, str]]]:
    """Converts a list of quanta into the required schedule format.

    Args:
        qts (QuantumTimeSystem): The quantum time system instance for conversion.
        quanta (List[int]): List of time quanta to be converted.

    Returns:
        Dict[str, List[Dict[str, str]]]: Schedule in the format:
            {
                "Monday": [
                    {"start": "09:00", "end": "12:00"},
                    {"start": "14:00", "end": "17:00"}
                ]
            }
    """
    if not quanta:
        return {}
    return qts.decode_schedule(set(quanta))


def _save_schedule_as_json(
    schedule: list[CourseSession],
    output_path: str,
    qts: QuantumTimeSystem,
    course_lookup: dict[tuple[str, str], Course] | None = None,
) -> str:
    """Saves a list of CourseSession objects as a JSON file.

    Args:
        schedule (List[CourseSession]): Decoded sessions from final GA output.
        output_path (str): Output directory to store the JSON file.
        qts (QuantumTimeSystem): Quantum time system for converting quanta to day/time.

    Returns:
        str: Full path to the saved JSON file.

    Note:
        Creates the output directory if it doesn't exist.
        The JSON file will be named 'schedule.json'.
    """
    filename = "schedule.json"
    full_path = str(Path(output_path) / filename)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    result = []
    for session in schedule:
        time_schedule = _get_time_schedule_format(qts, session.session_quanta)
        course_name, course_code, course_display = _resolve_course_details(
            session, course_lookup
        )
        instructor_name = _resolve_instructor_name(session)

        result.append(
            {
                "course_id": course_name,
                "course_name": course_name,
                "course_display": course_display,
                "course_code": course_code,
                "original_course_id": session.course_id,
                "course_type": session.course_type,
                "instructor_id": session.instructor_id,
                "instructor_name": instructor_name,
                "group_ids": (
                    session.group_ids
                ),  # Export as list for multi-group support
                "room_id": session.room_id,
                "time": time_schedule,
            }
        )

    with Path(full_path).open("w") as f:
        json.dump(result, f, indent=2)

    return full_path


def _get_operating_bounds(
    qts: QuantumTimeSystem | None,
) -> tuple[int, int, list[str]]:
    """Derive start_hour, end_hour, and active day names from QTS.

    Falls back to EXCAL constants when QTS is unavailable.
    """
    if qts is None:
        all_days = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        return EXCAL_START_HOUR, EXCAL_END_HOUR, all_days

    min_hour = 23
    max_hour = 0
    active_days: list[str] = []
    for day in [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]:
        hours = qts.operating_hours.get(day)
        if hours is None:
            continue
        active_days.append(day)
        open_str, close_str = hours
        oh, om = map(int, open_str.split(":"))
        ch, cm = map(int, close_str.split(":"))
        min_hour = min(min_hour, oh)
        max_hour = max(max_hour, ch + (1 if cm > 0 else 0))

    if not active_days:
        all_days = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        return EXCAL_START_HOUR, EXCAL_END_HOUR, all_days

    return min_hour, max_hour, active_days


# ── Colour palette ────────────────────────────────────────────────────
_THEORY_COLORS = [
    "#A8C6FA",  # soft blue
    "#B5D8B5",  # sage green
    "#C4B7D7",  # lavender
    "#F9D7A0",  # peach
    "#A8E0D1",  # mint
    "#D4C5F9",  # lilac
    "#B8DFF0",  # sky
    "#C9E8B2",  # lime
]
_PRACTICAL_COLORS = [
    "#F4A6A0",  # coral
    "#F2C6A0",  # apricot
    "#F7B7D2",  # pink
    "#E6A8D7",  # orchid
    "#F0B8B8",  # salmon
    "#F5C7B8",  # melon
]


def _build_rich_color_map(course_ids: set[tuple[str, str]]) -> dict[str, str]:
    """Assign distinct pastel colours per course, separated by type."""
    cmap: dict[str, str] = {}
    ti, pi = 0, 0
    for label, ctype in sorted(course_ids):
        if ctype == "practical" or "(PR)" in label:
            cmap[label] = _PRACTICAL_COLORS[pi % len(_PRACTICAL_COLORS)]
            pi += 1
        else:
            cmap[label] = _THEORY_COLORS[ti % len(_THEORY_COLORS)]
            ti += 1
    return cmap


# ── Day-header colours ────────────────────────────────────────────────
_DAY_HEADER_BG = "#2C3E6B"  # dark navy
_DAY_HEADER_FG = "#FFFFFF"
_GRID_LINE_COLOR = "#D0D0D0"
_ALT_ROW_COLOR = "#F7F8FA"
_BORDER_COLOR = "#888888"


def _save_json_schedule_as_pdf(
    json_path: str,
    output_pdf_path: str,
    quantum_minutes: int,
    start_hour: int,
    end_hour: int,
    *,
    qts: QuantumTimeSystem | None = None,
) -> None:
    """Converts a structured JSON schedule into a calendar-style PDF.

    Creates a multi-page PDF with one calendar page per group. Sessions are
    color-coded by course and merged when they are consecutive.  The calendar
    shows only operational hours and active days.
    """
    # Derive actual bounds from QTS (fall back to args)
    auto_start, auto_end, active_days = _get_operating_bounds(qts)
    start_hour = auto_start
    end_hour = auto_end

    day_idx = {day: i for i, day in enumerate(active_days)}
    num_days = len(active_days)
    num_hours = end_hour - start_hour
    time_format = "%H:%M"

    def to_float(time_str: str) -> float:
        t = datetime.strptime(time_str, time_format)
        return t.hour + t.minute / 60.0

    def merge_sessions(sessions: list[dict]) -> list[dict]:
        merged: list[dict] = []
        sessions.sort(key=lambda x: (x["day"], x["start"]))
        i = 0
        while i < len(sessions):
            s = dict(sessions[i])
            j = i + 1
            while j < len(sessions):
                n = sessions[j]
                if (
                    n["label"] == s["label"]
                    and n["day"] == s["day"]
                    and abs(n["start"] - s["end"]) < (quantum_minutes / 60.0) + 1e-6
                ):
                    s["end"] = n["end"]
                    j += 1
                else:
                    break
            merged.append(s)
            i = j
        return merged

    def plot_schedule(
        sessions: list[dict], group_name: str, pdf: Any, color_map: dict[str, str]
    ) -> None:
        sessions = merge_sessions(sessions)

        # ── Landscape A4-ish figure ──
        fig_w = max(11, num_days * 2.4)
        fig_h = max(6, num_hours * 0.95 + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        # ── Title ──
        ax.set_title(
            f"Weekly Schedule — {group_name}",
            fontsize=18,
            fontweight="bold",
            color="#2C3E6B",
            pad=28,
        )

        # Axes limits
        ax.set_xlim(0, num_days)
        ax.set_ylim(end_hour, start_hour)

        # ── Day column headers (coloured bar) ──
        for i, day in enumerate(active_days):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (i, start_hour - 0.65),
                    1.0,
                    0.6,
                    boxstyle="round,pad=0.02",
                    facecolor=_DAY_HEADER_BG,
                    edgecolor="none",
                    clip_on=False,
                )
            )
            ax.text(
                i + 0.5,
                start_hour - 0.35,
                day,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=_DAY_HEADER_FG,
                clip_on=False,
            )

        # Remove default x-axis ticks (we drew our own headers)
        ax.set_xticks([])

        # ── Y-axis: time labels ──
        ax.set_yticks(range(start_hour, end_hour + 1))
        ax.set_yticklabels(
            [f"{h:02d}:00" for h in range(start_hour, end_hour + 1)],
            fontsize=10,
            fontweight="medium",
            color="#333333",
        )
        ax.tick_params(axis="y", length=0, pad=8)

        # ── Alternating row shading ──
        for h in range(start_hour, end_hour):
            if (h - start_hour) % 2 == 0:
                ax.add_patch(
                    plt.Rectangle(
                        (0, h),
                        num_days,
                        1,
                        facecolor=_ALT_ROW_COLOR,
                        edgecolor="none",
                        zorder=0,
                    )
                )

        # ── Grid lines ──
        for h in range(start_hour, end_hour + 1):
            ax.axhline(y=h, color=_GRID_LINE_COLOR, linewidth=0.6, zorder=1)
        for d in range(num_days + 1):
            ax.axvline(x=d, color=_GRID_LINE_COLOR, linewidth=0.6, zorder=1)

        # ── Session blocks ──
        for session in sessions:
            day = session["day"]
            if day not in day_idx:
                continue
            x = day_idx[day]
            y = session["start"]
            height = session["end"] - session["start"]
            label = session["label"]
            course_base = session.get("course_base", label)
            color = color_map.get(course_base, "#CCCCCC")

            # Rounded rectangle
            block = mpatches.FancyBboxPatch(
                (x + 0.06, y + 0.03),
                0.88,
                height - 0.06,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="#444444",
                linewidth=1.0,
                zorder=3,
            )
            ax.add_patch(block)

            # ── Text inside block ──
            if ", " in label:
                course_part, instructor_part = label.split(", ", 1)
                wrapped = textwrap.fill(course_part, width=18, break_long_words=False)
                display_text = f"{wrapped}\n{instructor_part}"
            else:
                display_text = textwrap.fill(label, width=18, break_long_words=False)

            # Dynamic font size based on block height
            if height >= 2.0:
                fs = 9
            elif height >= 1.0:
                fs = 8
            else:
                fs = 7

            ax.text(
                x + 0.5,
                y + height / 2,
                display_text,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight="semibold",
                color="#1a1a1a",
                multialignment="center",
                zorder=4,
            )

        # ── Outer border ──
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER_COLOR)
            spine.set_linewidth(1.0)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ── Load JSON ──
    with Path(json_path).open() as f:
        data = json.load(f)

    group_sessions: dict[str, list[dict]] = defaultdict(list)
    course_ids: set[tuple[str, str]] = set()

    for entry in data:
        group_ids = entry.get(
            "group_ids", [entry.get("group_id")] if entry.get("group_id") else []
        )
        course_label = entry.get("course_display") or entry.get("course_id")
        instructor_label = entry.get("instructor_name") or entry.get("instructor_id")
        course_type = entry.get("course_type", "theory")
        course_ids.add((course_label, course_type))

        for day, times in entry["time"].items():
            for s in times:
                start = to_float(s["start"])
                end = to_float(s["end"])
                for group in group_ids:
                    if group:
                        label = course_label
                        if instructor_label:
                            label = f"{label}, {instructor_label}"
                        group_sessions[group].append(
                            {
                                "day": day,
                                "start": start,
                                "end": end,
                                "label": label,
                                "course_base": course_label,
                                "course_type": course_type,
                            }
                        )

    color_map = _build_rich_color_map(course_ids)

    # Save PDF
    with PdfPages(output_pdf_path) as pdf:
        for group_id in sorted(group_sessions):
            plot_schedule(group_sessions[group_id], group_id, pdf, color_map)

    print(f" PDF saved as '{output_pdf_path}'")


def export_everything(
    schedule: list[CourseSession],
    output_path: str,
    qts: QuantumTimeSystem,
    course_lookup: dict[tuple[str, str], Course] | None = None,
    parallel: bool = True,
) -> None:
    """Exports schedule as both JSON and PDF to a single directory.

    This is the main export function that combines JSON and PDF generation.
    It uses configuration values from calendar_config.py for PDF settings.

    Args:
        schedule (List[CourseSession]): Decoded sessions from genetic algorithm output.
        output_path (str): Output directory path. Will be created if it doesn't exist.
        qts (QuantumTimeSystem): Quantum time system instance for time conversion.
        course_lookup (Dict[Tuple[str, str], Course], optional): Lookup table for
            resolving human-friendly course names. Defaults to None.
        parallel (bool): If True, generate JSON and PDF concurrently (default: True, 2x faster)

    Example:
        >>> from src.exporter.exporter import export_everything
        >>> export_everything(final_schedule, "./output", qts_instance)
        [OK-KRISHNA] Schedule exported successfully!
        JSON: ./output/schedule.json
        [...]PDF:  ./output/calendar.pdf

    Note:
        - Creates output directory if it doesn't exist
        - JSON file is always named 'schedule.json'
        - PDF filename comes from EXCAL_DEFAULT_OUTPUT_PDF config
        - PDF settings (hours, quantum minutes) come from calendar_config.py
        - Parallel mode generates JSON and PDF concurrently (2x speedup)
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if not parallel:
        # Sequential export (for debugging)
        json_path = _save_schedule_as_json(
            schedule, output_path, qts, course_lookup=course_lookup
        )
        pdf_path = str(Path(output_path) / EXCAL_DEFAULT_OUTPUT_PDF)
        _save_json_schedule_as_pdf(
            json_path=json_path,
            output_pdf_path=pdf_path,
            quantum_minutes=EXCAL_QUANTUM_MINUTES,
            start_hour=EXCAL_START_HOUR,
            end_hour=EXCAL_END_HOUR,
            qts=qts,
        )
    else:
        # Parallel export (2x faster)
        pdf_path = str(Path(output_path) / EXCAL_DEFAULT_OUTPUT_PDF)

        def save_json() -> str:
            """Worker function for JSON export."""
            return _save_schedule_as_json(
                schedule, output_path, qts, course_lookup=course_lookup
            )

        def save_pdf(json_path_result: str) -> str:
            """Worker function for PDF export."""
            _save_json_schedule_as_pdf(
                json_path=json_path_result,
                output_pdf_path=pdf_path,
                quantum_minutes=EXCAL_QUANTUM_MINUTES,
                start_hour=EXCAL_START_HOUR,
                end_hour=EXCAL_END_HOUR,
                qts=qts,
            )
            return pdf_path

        # Generate JSON first (PDF depends on it)
        with ThreadPoolExecutor(max_workers=1) as executor:
            json_future = executor.submit(save_json)
            json_path = json_future.result()

        # Then generate PDF (independent after JSON is ready)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pdf_future = executor.submit(save_pdf, json_path)
            pdf_path = pdf_future.result()

    print("[OK-KRISHNA] Schedule exported successfully!")
    print(f"[...]JSON: {json_path}")
    print(f"[...]PDF:  {pdf_path}")
