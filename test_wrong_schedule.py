"""Debug helper to locate invalid (start > end) schedule entries.

Run this script from the folder that also contains `schedule.json` or provide a
custom path via CLI:

    uv run python test_wrong_schedule.py [path/to/schedule.json]

It will scan each session/day block and report cases where the start time is not
strictly earlier than the end time.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TimeBlock:
    course_id: str
    instructor_id: str
    room_id: str
    groups: list[str]
    day: str
    start: str
    end: str

    def is_invalid(self) -> bool:
        """Return True if start >= end after converting to minutes."""

        return _time_to_minutes(self.start) >= _time_to_minutes(self.end)


def _time_to_minutes(value: str) -> int:
    try:
        hour_str, minute_str = value.split(":", maxsplit=1)
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError as exc:  # pragma: no cover - simple utility
        raise ValueError(f"Invalid HH:MM time value '{value}'") from exc

    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"Time '{value}' outside 24-hour bounds")

    return hour * 60 + minute


def _load_schedule_blocks(schedule_path: Path) -> Iterable[TimeBlock]:
    with schedule_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    for session in payload:
        course_id = session.get("course_id", "<unknown>")
        instructor_id = session.get("instructor_id", "<unknown>")
        room_id = session.get("room_id", "<unknown>")
        groups = session.get("group_ids", [])
        time_map = session.get("time", {}) or {}

        for day, blocks in time_map.items():
            for block in blocks:
                yield TimeBlock(
                    course_id=course_id,
                    instructor_id=instructor_id,
                    room_id=room_id,
                    groups=list(groups),
                    day=day,
                    start=block.get("start", "00:00"),
                    end=block.get("end", "00:00"),
                )


def main() -> int:
    default_path = Path(__file__).with_name("schedule.json")
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not target.exists():
        print(f"[error] schedule file not found: {target}")
        return 1

    invalid_blocks = [
        block for block in _load_schedule_blocks(target) if block.is_invalid()
    ]

    if not invalid_blocks:
        print(f"[ok] All time blocks in {target} have start < end")
        return 0

    print("[warn] Found time blocks with start >= end:\n")
    for block in invalid_blocks:
        print(
            f"- {block.day:>9}  {block.start}->{block.end}  "
            f"course={block.course_id}  instructor={block.instructor_id}  "
            f"room={block.room_id}  groups={','.join(block.groups)}"
        )

    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
