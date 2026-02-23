"""
Instructor-wise & Room-wise Schedule PDF Generators.

Produces two PDF files alongside the existing group-wise calendar.pdf:
  - instructor_schedules.pdf  — one page per instructor
  - room_schedules.pdf        — one page per room

Each page has:
  - Entity header (name, availability / capacity / features)
  - Constraint violation summary
  - Weekly calendar grid with session blocks
"""

from __future__ import annotations

import logging
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from src.domain.course import Course
    from src.domain.instructor import Instructor
    from src.domain.room import Room
    from src.domain.session import CourseSession
    from src.io.time_system import QuantumTimeSystem


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DAYS = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]
DAY_IDX = {d: i for i, d in enumerate(DAYS)}

# Visual constants
_DAY_HEADER_BG = "#2C3E6B"
_DAY_HEADER_FG = "#FFFFFF"
_GRID_LINE_COLOR = "#D0D0D0"
_ALT_ROW_COLOR = "#F7F8FA"
_BORDER_COLOR = "#888888"

# Pastel palettes
_THEORY_COLORS = [
    "#A8C6FA",
    "#B5D8B5",
    "#C4B7D7",
    "#F9D7A0",
    "#A8E0D1",
    "#D4C5F9",
    "#B8DFF0",
    "#C9E8B2",
]
_PRACTICAL_COLORS = [
    "#F4A6A0",
    "#F2C6A0",
    "#F7B7D2",
    "#E6A8D7",
    "#F0B8B8",
    "#F5C7B8",
]


def _to_float(time_str: str) -> float:
    t = datetime.strptime(time_str, "%H:%M")
    return t.hour + t.minute / 60.0


def _get_operating_bounds(
    qts: Any | None,
) -> tuple[int, int, list[str]]:
    """Derive start_hour, end_hour, and active day names from QTS."""
    if qts is None:
        return 7, 20, list(DAYS)
    min_h, max_h = 23, 0
    active: list[str] = []
    for day in DAYS:
        hours = qts.operating_hours.get(day)
        if hours is None:
            continue
        active.append(day)
        oh, _ = map(int, hours[0].split(":"))
        ch, cm = map(int, hours[1].split(":"))
        min_h = min(min_h, oh)
        max_h = max(max_h, ch + (1 if cm > 0 else 0))
    if not active:
        return 7, 20, list(DAYS)
    return min_h, max_h, active


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────


def _merge_sessions(sessions: list[dict], quantum_minutes: int = 15) -> list[dict]:
    """Merge consecutive sessions with same label on the same day."""
    sessions.sort(key=lambda x: (x["day"], x["start"]))
    merged: list[dict] = []
    i = 0
    while i < len(sessions):
        cur = dict(sessions[i])
        j = i + 1
        while j < len(sessions):
            nxt = sessions[j]
            if (
                nxt["label"] == cur["label"]
                and nxt["day"] == cur["day"]
                and abs(nxt["start"] - cur["end"]) < (quantum_minutes / 60.0) + 1e-6
            ):
                cur["end"] = nxt["end"]
                j += 1
            else:
                break
        merged.append(cur)
        i = j
    return merged


def _draw_calendar(
    ax: plt.Axes,
    sessions: list[dict],
    start_hour: int,
    end_hour: int,
    color_map: dict[str, str],
    quantum_minutes: int = 15,
    *,
    active_days: list[str] | None = None,
) -> None:
    """Draw session rectangles on an axes with a polished weekly grid."""
    sessions = _merge_sessions(sessions, quantum_minutes)

    if active_days is None:
        active_days = list(DAYS)
    day_idx = {d: i for i, d in enumerate(active_days)}
    num_days = len(active_days)

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
            fontsize=10,
            fontweight="bold",
            color=_DAY_HEADER_FG,
            clip_on=False,
        )
    ax.set_xticks([])

    # ── Y-axis ──
    ax.set_yticks(range(start_hour, end_hour + 1))
    ax.set_yticklabels(
        [f"{h:02d}:00" for h in range(start_hour, end_hour + 1)],
        fontsize=9,
        fontweight="medium",
        color="#333333",
    )
    ax.tick_params(axis="y", length=0, pad=8)

    # ── Grid lines ──
    for h in range(start_hour, end_hour + 1):
        ax.axhline(y=h, color=_GRID_LINE_COLOR, linewidth=0.6, zorder=1)
    for d in range(num_days + 1):
        ax.axvline(x=d, color=_GRID_LINE_COLOR, linewidth=0.6, zorder=1)

    # ── Session blocks ──
    for sess in sessions:
        day = sess["day"]
        if day not in day_idx:
            continue
        x = day_idx[day]
        y = sess["start"]
        h = sess["end"] - sess["start"]
        label = sess["label"]
        course_base = sess.get("course_base", label)
        color = color_map.get(course_base, "#CCCCCC")

        block = mpatches.FancyBboxPatch(
            (x + 0.06, y + 0.03),
            0.88,
            h - 0.06,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="#444444",
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(block)

        # Multi-line text
        if ", " in label:
            parts = label.split(", ", 1)
            wrapped = textwrap.fill(parts[0], width=18, break_long_words=False)
            display = f"{wrapped}\n{parts[1]}"
        else:
            display = textwrap.fill(label, width=18, break_long_words=False)

        if h >= 2.0:
            fs = 8.5
        elif h >= 1.0:
            fs = 7.5
        else:
            fs = 6.5

        ax.text(
            x + 0.5,
            y + h / 2,
            display,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="semibold",
            color="#1a1a1a",
            multialignment="center",
            zorder=4,
        )

    # ── Border ──
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER_COLOR)
        spine.set_linewidth(1.0)


def _draw_availability_bands(
    ax: plt.Axes,
    avail_ranges: dict[str, list[tuple[int, int]]],
    qts: QuantumTimeSystem,
    *,
    active_days: list[str] | None = None,
) -> None:
    """Draw light-green availability bands behind the calendar."""
    if active_days is None:
        active_days = list(DAYS)
    day_idx = {d: i for i, d in enumerate(active_days)}
    for day_name, ranges in avail_ranges.items():
        if day_name not in day_idx:
            continue
        x = day_idx[day_name]
        for start_q, end_q in ranges:
            _, start_time = qts.quanta_to_time(start_q)
            # end_q is exclusive — back up by 1 for the last quantum end
            _, end_time = qts.quanta_to_time(min(end_q - 1, qts.total_quanta - 1))
            y0 = _to_float(start_time)
            y1 = _to_float(end_time) + qts.QUANTUM_MINUTES / 60.0
            rect = plt.Rectangle(
                (x, y0),
                1.0,
                y1 - y0,
                facecolor="#8CD98C",
                edgecolor="none",
                alpha=0.45,
                zorder=0.5,
            )
            ax.add_patch(rect)


# ──────────────────────────────────────────────────────────────────────
# Violation helpers
# ──────────────────────────────────────────────────────────────────────


def _compute_instructor_violations(
    sessions: list[CourseSession],
    instructors: dict[str, Instructor],
    qts: QuantumTimeSystem,
) -> dict[str, dict[str, int]]:
    """Per-instructor violation counts.

    Returns ``{instructor_id: {"double_booking": n, "outside_avail": m}}``.
    """
    result: dict[str, dict[str, int]] = {}

    # Double-booking: >=2 sessions at same quantum
    occ: dict[str, Counter] = defaultdict(Counter)
    for s in sessions:
        for q in s.session_quanta:
            occ[s.instructor_id][q] += 1

    for iid, inst in instructors.items():
        dbl = sum(max(0, c - 1) for c in occ.get(iid, {}).values())
        # Availability
        avail_v = 0
        if not inst.is_full_time:
            for s in sessions:
                if s.instructor_id != iid:
                    continue
                for q in s.session_quanta:
                    if q not in inst.available_quanta:
                        avail_v += 1
        result[iid] = {"double_booking": dbl, "outside_avail": avail_v}

    return result


def _compute_room_violations(
    sessions: list[CourseSession],
    rooms: dict[str, Room],
    courses: dict[tuple[str, str], Any] | None,
    groups: dict[str, Any] | None = None,
) -> dict[str, dict[str, int]]:
    """Per-room violation counts.

    Returns ``{room_id: {"double_booking": n, "wrong_type": m, "over_capacity": k}}``.
    """
    from src.utils.room_compatibility import is_room_suitable_for_course

    result: dict[str, dict[str, int]] = {}

    occ: dict[str, Counter] = defaultdict(Counter)
    for s in sessions:
        for q in s.session_quanta:
            occ[s.room_id][q] += 1

    for rid, room in rooms.items():
        dbl = sum(max(0, c - 1) for c in occ.get(rid, {}).values())

        # Type/capacity violations for sessions actually in this room
        wrong_type = 0
        over_cap = 0
        for s in sessions:
            if s.room_id != rid:
                continue

            # Capacity — resolve each group from the groups dict
            total_students = 0
            for gid in s.group_ids:
                grp = (groups or {}).get(gid)
                if (
                    grp is None
                    and s.group
                    and getattr(s.group, "group_id", None) == gid
                ):
                    grp = s.group
                if grp:
                    total_students += getattr(grp, "student_count", 0)
            if total_students > room.capacity:
                over_cap += 1

            # Room type suitability
            if courses:
                course = courses.get((s.course_id, s.course_type))
                if course:
                    req = getattr(course, "required_room_features", "lecture")
                    req_str = req.lower().strip() if isinstance(req, str) else "lecture"
                    lab_feats = getattr(course, "specific_lab_features", None) or []
                    room_str = (
                        room.room_features.lower().strip()
                        if isinstance(room.room_features, str)
                        else "lecture"
                    )
                    room_spec = getattr(room, "specific_features", None) or []
                    if not is_room_suitable_for_course(
                        req_str, room_str, lab_feats, room_spec
                    ):
                        wrong_type += 1

        result[rid] = {
            "double_booking": dbl,
            "wrong_type": wrong_type,
            "over_capacity": over_cap,
        }

    return result


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def generate_instructor_schedules_pdf(
    sessions: list[CourseSession],
    instructors: dict[str, Instructor],
    courses: dict[tuple[str, str], Course] | None,
    qts: QuantumTimeSystem,
    output_path: str,
    *,
    start_hour: int = 10,
    end_hour: int = 17,
    filename: str = "instructor_schedules.pdf",
) -> str:
    """Generate a multi-page PDF with one calendar page per instructor.

    Each page shows:
      - Instructor name + ID
      - Availability: "Full-Time" or day-wise time ranges
      - Violation summary (double-booking, outside-availability)
      - Weekly calendar with green availability bands + session blocks
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = str(out / filename)

    # Auto-detect operating bounds from QTS
    auto_start, auto_end, active_days = _get_operating_bounds(qts)
    start_hour, end_hour = auto_start, auto_end
    num_days = len(active_days)
    num_hours = end_hour - start_hour
    # Collect sessions per instructor
    inst_sessions: dict[str, list[CourseSession]] = defaultdict(list)
    for s in sessions:
        inst_sessions[s.instructor_id].append(s)

    # Build color map
    course_ids: set[tuple[str, str]] = set()
    for s in sessions:
        display = _course_display(s, courses)
        course_ids.add((display, s.course_type))
    color_map = _build_color_map(course_ids)

    viol = _compute_instructor_violations(sessions, instructors, qts)

    with PdfPages(pdf_path) as pdf:
        # Sort by name for readability
        for iid in sorted(instructors, key=lambda k: instructors[k].name):
            inst = instructors[iid]
            my_sessions = inst_sessions.get(iid, [])
            my_viol = viol.get(iid, {})

            fig_w = max(11, num_days * 2.4)
            fig_h = max(7, num_hours * 0.95 + 3.5)

            # Two-row layout: header text row (fixed height) + calendar row.
            # gridspec height_ratios ensures the header never overlaps.
            fig = plt.figure(figsize=(fig_w, fig_h))
            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[1.8, num_hours],
                hspace=0.08,
                left=0.07,
                right=0.97,
                top=0.97,
                bottom=0.04,
            )
            ax_hdr = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")

            # ── Header (drawn in its own axes — structurally above calendar) ──
            avail_text = _format_instructor_availability(inst, qts)
            total_v = sum(my_viol.values())
            viol_detail = ", ".join(
                f"{k.replace('_', ' ')}={v}" for k, v in my_viol.items() if v > 0
            )
            viol_line = (
                f"Violations: {total_v} ({viol_detail})"
                if total_v > 0
                else "Violations: 0  ✓"
            )
            n_sessions = len(my_sessions)
            total_quanta = sum(len(s.session_quanta) for s in my_sessions)

            # Render header lines in the dedicated header axes (2 lines only)
            # Line 1: Name + ID + session stats + violations (merged)
            # Line 2: Availability
            ax_hdr.set_xlim(0, 1)
            ax_hdr.set_ylim(0, 1)
            ax_hdr.axis("off")

            line1 = (
                f"{inst.name}  ({iid})    |    "
                f"Sessions: {n_sessions}   Quanta: {total_quanta}   {viol_line}"
            )
            ax_hdr.text(
                0.0,
                0.85,
                line1,
                fontsize=11,
                fontweight="bold",
                family="monospace",
                color="#2C3E6B",
                va="top",
                ha="left",
                transform=ax_hdr.transAxes,
            )
            ax_hdr.text(
                0.0,
                0.35,
                f"Availability: {avail_text}",
                fontsize=9,
                family="monospace",
                color="#555555",
                va="top",
                ha="left",
                transform=ax_hdr.transAxes,
            )

            # ── Availability bands ──
            avail_ranges = inst.get_available_quanta_ranges(qts)
            _draw_availability_bands(ax, avail_ranges, qts, active_days=active_days)

            # ── Sessions ──
            cal_sessions = _sessions_to_cal(
                my_sessions, qts, courses, label_mode="instructor"
            )
            _draw_calendar(
                ax,
                cal_sessions,
                start_hour,
                end_hour,
                color_map,
                active_days=active_days,
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    logging.getLogger(__name__).info("Instructor schedules PDF saved: %s", pdf_path)
    return pdf_path


def generate_room_schedules_pdf(
    sessions: list[CourseSession],
    rooms: dict[str, Room],
    courses: dict[tuple[str, str], Course] | None,
    qts: QuantumTimeSystem,
    output_path: str,
    *,
    groups: dict[str, Any] | None = None,
    start_hour: int = 10,
    end_hour: int = 17,
    filename: str = "room_schedules.pdf",
) -> str:
    """Generate a multi-page PDF with one calendar page per room.

    Each page shows:
      - Room name + ID, capacity, features
      - Violation summary (double-booking, wrong type, over capacity)
      - Weekly calendar with session blocks
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = str(out / filename)

    # Auto-detect operating bounds from QTS
    auto_start, auto_end, active_days = _get_operating_bounds(qts)
    start_hour, end_hour = auto_start, auto_end
    num_days = len(active_days)
    num_hours = end_hour - start_hour

    # Collect sessions per room
    room_sessions: dict[str, list[CourseSession]] = defaultdict(list)
    for s in sessions:
        room_sessions[s.room_id].append(s)

    # Build color map
    course_ids: set[tuple[str, str]] = set()
    for s in sessions:
        display = _course_display(s, courses)
        course_ids.add((display, s.course_type))
    color_map = _build_color_map(course_ids)

    viol = _compute_room_violations(sessions, rooms, courses, groups)

    with PdfPages(pdf_path) as pdf:
        for rid in sorted(rooms, key=lambda k: rooms[k].name):
            room = rooms[rid]
            my_sessions = room_sessions.get(rid, [])
            my_viol = viol.get(rid, {})

            fig_w = max(11, num_days * 2.4)
            fig_h = max(6, num_hours * 0.95 + 2.5)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")

            # ── Header ──
            features_str = room.room_features or "—"
            spec_feats = (
                ", ".join(room.specific_features) if room.specific_features else "—"
            )
            total_v = sum(my_viol.values())
            viol_detail = ", ".join(
                f"{k.replace('_', ' ')}={v}" for k, v in my_viol.items() if v > 0
            )
            viol_line = (
                f"Violations: {total_v} ({viol_detail})"
                if total_v > 0
                else "Violations: 0  ✓"
            )
            n_sessions = len(my_sessions)
            total_quanta = sum(len(s.session_quanta) for s in my_sessions)

            # Utilization: quanta used vs total available per week
            total_avail = qts.total_quanta
            util_pct = (total_quanta / total_avail * 100) if total_avail else 0

            title = (
                f"{room.name}  ({rid})    Capacity: {room.capacity}\n"
                f"Type: {features_str}    Specific: {spec_feats}\n"
                f"Sessions: {n_sessions}   Quanta: {total_quanta}/"
                f"{total_avail} ({util_pct:.0f}%)   {viol_line}"
            )
            ax.set_title(
                title,
                fontsize=12,
                fontweight="bold",
                pad=28,
                loc="left",
                family="monospace",
                color="#2C3E6B",
            )

            # ── Sessions ──
            cal_sessions = _sessions_to_cal(
                my_sessions, qts, courses, label_mode="room"
            )
            _draw_calendar(
                ax,
                cal_sessions,
                start_hour,
                end_hour,
                color_map,
                active_days=active_days,
            )

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    logging.getLogger(__name__).info("Room schedules PDF saved: %s", pdf_path)
    return pdf_path


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


def _course_display(
    session: CourseSession,
    courses: dict[tuple[str, str], Any] | None,
) -> str:
    """Return a human-readable course display name."""
    tag = "PR" if session.course_type == "practical" else "TH"
    if courses:
        course = courses.get((session.course_id, session.course_type))
        if course:
            return f"{course.name} ({tag})"
    return f"{session.course_id} ({tag})"


def _build_color_map(
    course_ids: set[tuple[str, str]],
) -> dict[str, str]:
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


def _format_instructor_availability(
    inst: Any,
    qts: QuantumTimeSystem,
) -> str:
    """Return a compact availability string for an instructor."""
    if inst.is_full_time:
        return "Full-Time (all operating hours)"

    if not inst.available_quanta:
        return "No availability declared"

    ranges = inst.get_available_quanta_ranges(qts)
    parts = []
    for day in DAYS:
        if day not in ranges:
            continue
        time_strs = []
        for start_q, end_q in ranges[day]:
            _, st = qts.quanta_to_time(start_q)
            _, et = qts.quanta_to_time(min(end_q - 1, qts.total_quanta - 1))
            # Add one quantum to get end time
            et_min = _to_float(et) + qts.QUANTUM_MINUTES / 60.0
            eh = int(et_min)
            em = int((et_min - eh) * 60)
            time_strs.append(f"{st}-{eh:02d}:{em:02d}")
        parts.append(f"{day[:3]}: {', '.join(time_strs)}")

    return "  |  ".join(parts) if parts else "No availability"


def _sessions_to_cal(
    sessions: list[CourseSession],
    qts: QuantumTimeSystem,
    courses: dict[tuple[str, str], Any] | None,
    *,
    label_mode: str = "instructor",
) -> list[dict]:
    """Convert CourseSession list to calendar-plottable dicts.

    label_mode:
        "instructor" — label shows group IDs + course (for instructor view)
        "room"       — label shows instructor + group IDs + course
    """
    result: list[dict] = []
    for s in sessions:
        course_label = _course_display(s, courses)
        groups_str = ",".join(s.group_ids[:3])
        if len(s.group_ids) > 3:
            groups_str += f"+{len(s.group_ids) - 3}"

        inst_name = ""
        if s.instructor and s.instructor.name:
            inst_name = s.instructor.name
        else:
            inst_name = s.instructor_id

        if label_mode == "instructor":
            label = f"{course_label}, {groups_str}"
        else:
            label = f"{course_label}, {inst_name}\n{groups_str}"

        # Convert quanta to day/time
        schedule = qts.decode_schedule(set(s.session_quanta))
        for day_name, slots in schedule.items():
            result.extend(
                {
                    "day": day_name,
                    "start": _to_float(slot["start"]),
                    "end": _to_float(slot["end"]),
                    "label": label,
                    "course_base": course_label,
                    "course_type": s.course_type,
                }
                for slot in slots
            )

    return result
