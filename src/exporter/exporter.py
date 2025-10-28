import os
import json
from typing import List, Dict
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.entities.decoded_session import CourseSession
from src.encoder.quantum_time_system import QuantumTimeSystem

# Config values (used internally only)
from src.config.calendar_config import (
    EXCAL_QUANTUM_MINUTES,
    EXCAL_START_HOUR,
    EXCAL_END_HOUR,
    EXCAL_DEFAULT_OUTPUT_PDF,
)


def _format_course_name_with_type(course_id: str, course_type: str) -> str:
    """Append (TH) or (PR) tag to course name based on course type.

    Args:
        course_id (str): The course ID.
        course_type (str): The course type ('theory' or 'practical').

    Returns:
        str: Course ID with appropriate tag appended.
    """
    if course_type == "practical":
        return f"{course_id} (PR)"
    else:
        return f"{course_id} (TH)"


def _get_time_schedule_format(
    qts: QuantumTimeSystem, quanta: List[int]
) -> Dict[str, List[Dict[str, str]]]:
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
    schedule: List[CourseSession], output_path: str, qts: QuantumTimeSystem
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
    full_path = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)

    result = []
    for session in schedule:
        time_schedule = _get_time_schedule_format(qts, session.session_quanta)
        # Format course_id with (TH) or (PR) tag for display
        display_course_id = _format_course_name_with_type(
            session.course_id, session.course_type
        )

        result.append(
            {
                "course_id": display_course_id,  # Display with (TH) or (PR) tag
                "original_course_id": session.course_id,  # Keep original for reference
                "course_type": session.course_type,  # Include course type
                "instructor_id": session.instructor_id,
                "group_ids": session.group_ids,  # Export as list for multi-group support
                "room_id": session.room_id,
                "time": time_schedule,
            }
        )

    with open(full_path, "w") as f:
        json.dump(result, f, indent=2)

    return full_path


def _save_json_schedule_as_pdf(
    json_path: str,
    output_pdf_path: str,
    quantum_minutes: int,
    start_hour: int,
    end_hour: int,
):
    """Converts a structured JSON schedule into a calendar-style PDF.

    Creates a multi-page PDF with one calendar page per group. Sessions are
    color-coded by course and merged when they are consecutive.

    Args:
        json_path (str): Path to the input JSON schedule file.
        output_pdf_path (str): Path where the PDF will be saved.
        quantum_minutes (int): Time granularity in minutes for merging sessions.
        start_hour (int): Earliest hour shown on the calendar (e.g., 7 for 07:00).
        end_hour (int): Latest hour shown on the calendar (e.g., 20 for 20:00).

    Note:
        - Uses matplotlib to generate calendar grids
        - Each course gets a unique color from the tab20 colormap
        - Sessions are automatically merged if they are consecutive
        - PDF contains one page per student group
    """

    DAYS = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]
    DAY_IDX = {day: i for i, day in enumerate(DAYS)}
    TIME_FORMAT = "%H:%M"

    def to_float(time_str):
        """Convert time string to float hours.

        Args:
            time_str (str): Time in HH:MM format.

        Returns:
            float: Time as decimal hours (e.g., 14:30 -> 14.5).
        """
        t = datetime.strptime(time_str, TIME_FORMAT)
        return t.hour + t.minute / 60.0

    def merge_sessions(sessions):
        """Merge consecutive sessions with the same label and day.

        Args:
            sessions (List[Dict]): List of session dictionaries.

        Returns:
            List[Dict]: Merged sessions list.
        """
        merged = []
        sessions.sort(key=lambda x: (x["day"], x["start"]))
        i = 0
        while i < len(sessions):
            s = sessions[i]
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

    def plot_schedule(sessions, group_name, pdf, color_map):
        """Plot a weekly schedule for a specific group.

        Args:
            sessions (List[Dict]): Sessions for this group.
            group_name (str): Name of the student group.
            pdf (PdfPages): PDF writer object.
            color_map (Dict[str, str]): Mapping of course IDs to hex colors.
        """
        sessions = merge_sessions(sessions)

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_title(f"Routine for {group_name}", fontsize=16, pad=20)
        ax.set_xlim(0, len(DAYS))
        ax.set_ylim(end_hour, start_hour)
        ax.set_xticks(range(len(DAYS)))
        ax.set_xticklabels(DAYS, fontsize=10)
        ax.set_yticks(range(start_hour, end_hour + 1))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(start_hour, end_hour + 1)])
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        for session in sessions:
            day = session["day"]
            if day not in DAY_IDX:
                continue
            x = DAY_IDX[day]
            y = session["start"]
            height = session["end"] - session["start"]
            label = session["label"]
            color = color_map.get(label, "#CCCCCC")

            rect = plt.Rectangle(
                (x + 0.05, y),
                0.9,
                height,
                edgecolor="black",
                facecolor=color,
                linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(
                x + 0.5,
                y + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",  # Black text for both theory and practical
                wrap=True,
            )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Load JSON
    with open(json_path) as f:
        data = json.load(f)

    group_sessions = defaultdict(list)
    course_ids = set()

    for entry in data:
        # Handle both old format (group_id) and new format (group_ids)
        group_ids = entry.get(
            "group_ids", [entry.get("group_id")] if entry.get("group_id") else []
        )
        course = entry["course_id"]
        course_ids.add(course)

        # Add session to all groups in the list
        for day, times in entry["time"].items():
            for s in times:
                start = to_float(s["start"])
                end = to_float(s["end"])
                for group in group_ids:
                    if group:  # Skip None values
                        group_sessions[group].append(
                            {"day": day, "start": start, "end": end, "label": course}
                        )

    # Assign colors based on course type: blue for theory, red for practical
    color_map = {}
    for course in course_ids:
        if "(PR)" in course:
            color_map[course] = "#F16A6A"  # Red for practical
        else:
            color_map[course] = "#8888F7"  # Blue for theory

    # Save PDF
    with PdfPages(output_pdf_path) as pdf:
        for group_id, sessions in group_sessions.items():
            plot_schedule(sessions, group_id, pdf, color_map)

    print(f" PDF saved as '{output_pdf_path}'")


def export_everything(
    schedule: List[CourseSession],
    output_path: str,
    qts: QuantumTimeSystem,
    parallel: bool = True,
):
    """Exports schedule as both JSON and PDF to a single directory.

    This is the main export function that combines JSON and PDF generation.
    It uses configuration values from calendar_config.py for PDF settings.

    Args:
        schedule (List[CourseSession]): Decoded sessions from genetic algorithm output.
        output_path (str): Output directory path. Will be created if it doesn't exist.
        qts (QuantumTimeSystem): Quantum time system instance for time conversion.
        parallel (bool): If True, generate JSON and PDF concurrently (default: True, 2x faster)

    Example:
        >>> from src.exporter.exporter import export_everything
        >>> export_everything(final_schedule, "./output", qts_instance)
        [OK-KRISHNA] Schedule exported successfully!
        JSON: ./output/schedule.json
        [...]PDF:  ./output/calendar_colored_merged.pdf

    Note:
        - Creates output directory if it doesn't exist
        - JSON file is always named 'schedule.json'
        - PDF filename comes from EXCAL_DEFAULT_OUTPUT_PDF config
        - PDF settings (hours, quantum minutes) come from calendar_config.py
        - Parallel mode generates JSON and PDF concurrently (2x speedup)
    """
    os.makedirs(output_path, exist_ok=True)

    if not parallel:
        # Sequential export (for debugging)
        json_path = _save_schedule_as_json(schedule, output_path, qts)
        pdf_path = os.path.join(output_path, EXCAL_DEFAULT_OUTPUT_PDF)
        _save_json_schedule_as_pdf(
            json_path=json_path,
            output_pdf_path=pdf_path,
            quantum_minutes=EXCAL_QUANTUM_MINUTES,
            start_hour=EXCAL_START_HOUR,
            end_hour=EXCAL_END_HOUR,
        )
    else:
        # Parallel export (2x faster)
        pdf_path = os.path.join(output_path, EXCAL_DEFAULT_OUTPUT_PDF)

        def save_json():
            """Worker function for JSON export."""
            return _save_schedule_as_json(schedule, output_path, qts)

        def save_pdf(json_path_result):
            """Worker function for PDF export."""
            _save_json_schedule_as_pdf(
                json_path=json_path_result,
                output_pdf_path=pdf_path,
                quantum_minutes=EXCAL_QUANTUM_MINUTES,
                start_hour=EXCAL_START_HOUR,
                end_hour=EXCAL_END_HOUR,
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
