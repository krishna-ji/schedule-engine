"""
CONTINUOUS QUANTUM TIME SYSTEM

This module implements a quantum-based time system where quantum indices are CONTINUOUS
and only cover operating hours. Non-operating times receive NO quantum indices.

CONSTANTS:
QUANTUM_MINUTES: Duration of a Single Quantum in Minutes (also the Unit Course Duration)
QUANTA_PER_HOUR: Number of Quanta in One Hour
UNIT_SESSION_DURATION_QUANTA: Duration of a Single Session in Quanta
DAY_NAMES: List of Days in a Week (Sunday-first order)
DEFAULT_OPERATING_HOURS: Default Operating Hours for Each Day

KEY FEATURES:
- Continuous indexing: Only operational times get quantum indices
- No gaps: Quantum 0, 1, 2, ... N-1 represent all operational time slots
- Day-aware: Each operational day contributes its operating hours only
- Efficient: No wasted indices on non-operational times (nights, weekends, etc.)

EXAMPLE:
If operating hours are:
  Sunday: 08:00-20:00 (12 hours = 12 quanta with 60-min quantum)
  Monday: 08:00-20:00 (12 hours = 12 quanta)

Then quantum indices are:
  0-11: Sunday 08:00-19:00
  12-23: Monday 08:00-19:00
  (No indices for Sunday 00:00-07:59, 20:00-23:59, etc.)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

from src.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class QuantumTimeSystem:
    """
    A quantum-based time system for scheduling with CONTINUOUS quantum indices.

    - Time is represented in quantum units (default: 60-minute units).
    - Operating hours are configured per day.
    - Quantum indices are CONTINUOUS and ONLY cover operating hours.
    - No quantum indices are assigned to non-operating times.
    - Supports schedule encoding/decoding into quantum sets.

    Public Methods:
        __init__(self)
        set_operating_hours(self, day: str, start_time: Optional[str], end_time: Optional[str])
        is_operational(self, day: str) -> bool
        encode_schedule(self, schedule_json: Dict) -> Set[int]
        decode_schedule(self, quanta_set: Set[int]) -> Dict[str, List[Dict]]
        get_all_operating_quanta(self) -> Set[int]
        time_to_quanta(self, day: str, time_str: str) -> int
        quanta_to_time(self, quantum: int) -> Tuple[str, str]

    Example:
        qts = QuantumTimeSystem()
        q = qts.time_to_quanta("Monday", "10:00")  # Returns continuous index
        day, time = qts.quanta_to_time(q)
    """

    # Constants
    QUANTUM_MINUTES: ClassVar[int] = (
        60  # Quantum duration in minutes (also unit course duration)
    )

    # Derived constants
    QUANTA_PER_HOUR: ClassVar[int] = 60 // QUANTUM_MINUTES
    UNIT_SESSION_DURATION_QUANTA: ClassVar[int] = 1  # One quantum per session

    # Day configuration
    DAY_NAMES: ClassVar[list[str]] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]

    DEFAULT_OPERATING_HOURS: ClassVar[dict[str, tuple[str, str] | None]] = {
        "Sunday": ("10:00", "17:00"),
        "Monday": ("10:00", "17:00"),
        "Tuesday": ("10:00", "17:00"),
        "Wednesday": ("10:00", "17:00"),
        "Thursday": ("10:00", "17:00"),
        "Friday": ("10:00", "17:00"),
        "Saturday": None,
    }

    def __init__(
        self, operating_hours: dict[str, tuple[str, str] | None] | None = None
    ):
        """
        Initializes the QuantumTimeSystem with default operating hours.
        Precomputes continuous quantum mappings for each operational day.

        Example:
            qts = QuantumTimeSystem()
        """
        resolved_hours = (
            operating_hours
            or self._resolve_operating_hours_from_config()
            or self.DEFAULT_OPERATING_HOURS
        )
        self.operating_hours = resolved_hours.copy()
        self._build_quanta_map()

    def _resolve_operating_hours_from_config(
        self,
    ) -> dict[str, tuple[str, str] | None] | None:
        """Derive operating hours from the active configuration if present."""

        try:
            cfg = get_config()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "QuantumTimeSystem using defaults (config unavailable): %s", exc
            )
            return None

        time_cfg = getattr(cfg, "time", None)
        if time_cfg is None:
            return None

        base_start = getattr(time_cfg, "opening_time", None)
        base_end = getattr(time_cfg, "closing_time", None)
        if not base_start or not base_end:
            return None

        closed_days = {
            day.capitalize() for day in (getattr(time_cfg, "closed_days", []) or [])
        }
        overrides = getattr(time_cfg, "day_overrides", {}) or {}

        resolved: dict[str, tuple[str, str] | None] = {}
        for day in self.DAY_NAMES:
            if day in closed_days:
                resolved[day] = None
                continue

            override = self._get_override_for_day(overrides, day)
            if override is None:
                resolved[day] = (base_start, base_end)
                continue

            resolved[day] = self._extract_override_hours(override)

        return resolved

    @staticmethod
    def _get_override_for_day(overrides: dict, day: str):
        """Fetch override entry ignoring case mismatches in config keys."""

        for key in (day, day.lower(), day.upper()):
            if key in overrides:
                return overrides[key]
        return None

    @staticmethod
    def _extract_override_hours(override) -> tuple[str, str] | None:
        """Normalize override payloads to (start, end) tuples."""

        if override is None:
            return None

        start = getattr(override, "start", None)
        end = getattr(override, "end", None)

        if start is None and isinstance(override, dict):
            start = override.get("start")
            end = override.get("end")

        if not start or not end:
            return None

        return (start, end)

    def _build_quanta_map(self) -> None:
        """
        Builds continuous quantum mappings for operational days only.

        Creates:
        - day_quanta_offset: Starting quantum index for each day
        - day_start_time: Operating start time for each day (in minutes from midnight)
        - day_quanta_count: Number of quanta available for each day
        - total_quanta: Total continuous quanta across all operational days

        Example:
            If Sun 08:00-20:00 (12 quanta) and Mon 08:00-20:00 (12 quanta):
            day_quanta_offset = {'Sunday': 0, 'Monday': 12, ...}
            day_start_time = {'Sunday': 600, 'Monday': 600, ...}  # 10*60 = 600
            day_quanta_count = {'Sunday': 7, 'Monday': 7, ...}
        """
        self.day_quanta_offset = {}
        self.day_start_time = {}
        self.day_quanta_count = {}

        current_offset = 0

        for day in self.DAY_NAMES:
            hours = self.operating_hours.get(day)

            if hours:
                # Parse start and end times
                start_hour, start_min = map(int, hours[0].split(":"))
                end_hour, end_min = map(int, hours[1].split(":"))

                # Convert to minutes from midnight
                start_minutes = start_hour * 60 + start_min
                end_minutes = end_hour * 60 + end_min

                # Calculate number of quanta for this day
                duration_minutes = end_minutes - start_minutes
                quanta_count = duration_minutes // self.QUANTUM_MINUTES

                self.day_quanta_offset[day] = current_offset
                self.day_start_time[day] = start_minutes
                self.day_quanta_count[day] = quanta_count

                current_offset += quanta_count
            else:
                # Non-operational day
                self.day_quanta_offset[day] = None
                self.day_start_time[day] = None
                self.day_quanta_count[day] = 0

        self.total_quanta = current_offset

    def time_to_quanta(self, day: str, time_str: str) -> int:
        """
        Convert (day, time) to CONTINUOUS quantum index.
        Only returns indices for times within operating hours.

        Args:
            day: Day of the week
            time_str: Time string in HH:MM format

        Returns:
            int: Continuous Quantum Index

        Raises:
            ValueError: If day is non-operational or time is outside operating hours

        Example:
            If Sunday operates 08:00-20:00 and Monday operates 08:00-20:00:
            Sunday 08:00 -> 0
            Sunday 09:00 -> 1
            Monday 08:00 -> 12
            Monday 09:00 -> 13
        """
        day = day.capitalize()

        # Check if day is operational
        if not self.is_operational(day):
            raise ValueError(f"{day} is not an operational day")

        # Parse time
        hour, minute = map(int, time_str.split(":"))
        time_minutes = hour * 60 + minute

        # Get day's operating parameters
        start_minutes = self.day_start_time[day]
        quanta_offset = self.day_quanta_offset[day]

        # Check if time is within operating hours
        operating_hours = self.operating_hours[day]
        end_hour, end_minute = map(int, operating_hours[1].split(":"))
        end_minutes = end_hour * 60 + end_minute

        if time_minutes < start_minutes or time_minutes >= end_minutes:
            raise ValueError(
                f"Time {time_str} on {day} is outside operating hours "
                f"({operating_hours[0]}-{operating_hours[1]})"
            )

        # Calculate quantum index within the day
        minutes_from_start = time_minutes - start_minutes
        quantum_in_day = minutes_from_start // self.QUANTUM_MINUTES

        # Return continuous quantum index
        return quanta_offset + quantum_in_day

    def quanta_to_time(self, quantum: int) -> tuple[str, str]:
        """
        Convert CONTINUOUS quantum index back to (day, HH:MM) format.

        Args:
            quantum: Continuous Quantum Index (0 to total_quanta-1)

        Returns:
            Tuple of (day_name, time_string in HH:MM format)

        Raises:
            ValueError: If quantum is out of valid range

        Example:
            If Sunday operates 08:00-20:00 (12 quanta), Monday 08:00-20:00 (12 quanta):
            quantum 0 -> ('Sunday', '08:00')
            quantum 11 -> ('Sunday', '19:00')
            quantum 12 -> ('Monday', '08:00')
            quantum 23 -> ('Monday', '19:00')
        """
        if not 0 <= quantum < self.total_quanta:
            raise ValueError(
                f"Quantum {quantum} is out of range (0 to {self.total_quanta-1})"
            )

        # Find which day this quantum belongs to
        for day in self.DAY_NAMES:
            if self.day_quanta_offset[day] is None:
                continue

            day_offset = self.day_quanta_offset[day]
            day_count = self.day_quanta_count[day]

            # Check if quantum falls within this day's range
            if day_offset <= quantum < day_offset + day_count:
                # Calculate quantum position within the day
                quantum_in_day = quantum - day_offset

                # Convert to time
                minutes_from_start = quantum_in_day * self.QUANTUM_MINUTES
                start_minutes = self.day_start_time[day]
                total_minutes = start_minutes + minutes_from_start

                hour = total_minutes // 60
                minute = total_minutes % 60

                return day, f"{hour:02d}:{minute:02d}"

        # Should never reach here if quantum is valid
        raise ValueError(f"Could not decode quantum {quantum}")

    def set_operating_hours(
        self, day: str, start_time: str | None, end_time: str | None
    ) -> None:
        """

        Set or Override the operating hours for a specific day.

        Args:
            day: Day name
            start_time: Start time string (HH:MM) format
            end_time: End time string (HH:MM) format

        Example:
            qts.set_operating_hours("Monday", "10:00", "17:00")

        """
        day = day.capitalize()
        self._validate_day(day)

        self.operating_hours[day] = (
            (start_time, end_time) if start_time and end_time else None
        )
        self._build_quanta_map()

    def _validate_day(self, day: str) -> None:
        """Validate that day exists in system"""
        if day not in self.DAY_NAMES:
            raise ValueError(f"Invalid day: {day}")

    def is_operational(self, day: str) -> bool:
        """Check if a day has operating hours"""
        return self.operating_hours.get(day.capitalize()) is not None

    def encode_schedule(self, schedule_json: dict) -> set[int]:
        """
        Convert JSON schedule to quantum set

        Args:
            schedule_json: { day: [{"start": "HH:MM", "end": "HH:MM"}] }

        Returns:
            Set of quantum indices
        """
        occupied_quanta = set()

        for day, periods in schedule_json.items():
            if not self.is_operational(day):
                continue
            for period in periods:
                occupied_quanta.update(self._get_period_quanta(day, period))

        return occupied_quanta

    def _get_period_quanta(self, day: str, period: dict) -> range:
        """
        Get quantum index range for a single period

        Args:
            day: Day of the week
            period: {"start": "HH:MM", "end": "HH:MM"}

        Returns:
            range(start, end) of quantum indices
        """
        start = self.time_to_quanta(day, period["start"])
        end = self.time_to_quanta(day, period["end"])  # Exclusive
        return range(start, end)

    def decode_schedule(self, quanta_set: set[int]) -> dict[str, list[dict]]:
        """
        Converts a set of continuous quantum indices back to readable JSON schedule.

        Args:
            quanta_set: Set of continuous quantum indices

        Returns:
            { day: [ {"start": "HH:MM", "end": "HH:MM"}, ... ] }
        """
        schedule = {day: [] for day in self.DAY_NAMES}
        day_groups = self._group_quanta_by_day(quanta_set)

        for day, quanta_list in day_groups.items():
            if quanta_list:
                schedule[day] = self._merge_consecutive_quanta(quanta_list)

        return {day: periods for day, periods in schedule.items() if periods}

    def _group_quanta_by_day(self, quanta_set: set[int]) -> dict[str, list[int]]:
        """
        Group continuous quanta by their corresponding days.

        Args:
            quanta_set: Set of continuous quantum indices

        Returns:
            Dict mapping day names to lists of quantum indices within that day
        """
        day_groups = defaultdict(list)

        for quantum in sorted(quanta_set):
            # Find which day this quantum belongs to
            for day in self.DAY_NAMES:
                if self.day_quanta_offset[day] is None:
                    continue

                day_offset = self.day_quanta_offset[day]
                day_count = self.day_quanta_count[day]

                if day_offset <= quantum < day_offset + day_count:
                    # This quantum belongs to this day
                    day_groups[day].append(quantum)
                    break

        return day_groups

    def _merge_consecutive_quanta(self, quanta_list: list[int]) -> list[dict]:
        """
        Merge consecutive continuous quanta into time periods.

        Args:
            quanta_list: List of continuous quantum indices (should be sorted)

        Returns:
            List of period dictionaries with start/end times
        """
        periods = []
        if not quanta_list:
            return periods

        sorted_quanta = sorted(quanta_list)
        current_start = sorted_quanta[0]
        current_end = current_start + 1

        for q in sorted_quanta[1:]:
            if q == current_end:
                current_end += 1
            else:
                periods.append(self._create_period(current_start, current_end))
                current_start = q
                current_end = q + 1

        periods.append(self._create_period(current_start, current_end))
        return periods

    def get_all_operating_quanta(self) -> set[int]:
        """
        Get all quantum time slots during operating hours across all days.

        Returns:
            Set[int]: Set of all continuous operating quantum indices (0 to total_quanta-1)
        """
        all_quanta = set()

        for day in self.DAY_NAMES:
            if self.day_quanta_offset[day] is not None:
                day_offset = self.day_quanta_offset[day]
                day_count = self.day_quanta_count[day]
                all_quanta.update(range(day_offset, day_offset + day_count))

        return all_quanta

    def _create_period(self, start: int, end: int) -> dict:
        """
        Converts start and end continuous quantum indices into a period dictionary.

        Args:
            start: Starting continuous quantum index (inclusive)
            end: Ending continuous quantum index (exclusive)

        Returns:
            Dict with 'start' and 'end' time strings
        """
        start_day, start_time = self.quanta_to_time(start)

        day_offset = self.day_quanta_offset[start_day] or 0
        day_quanta = self.day_quanta_count[start_day]
        day_end_quantum = day_offset + day_quanta

        if end >= day_end_quantum:
            end_time = self._get_day_end_time(start)
        else:
            _, end_time = self.quanta_to_time(end)

        return {
            "start": start_time,
            "end": end_time,
        }

    def _get_day_end_time(self, quantum: int) -> str:
        """
        Get the end time for the day containing the given quantum.
        Used when a period extends to the end of operating hours.
        """
        day, _ = self.quanta_to_time(quantum)
        operating_hours = self.operating_hours[day]
        return operating_hours[1]
