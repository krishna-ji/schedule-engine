"""Unified data loading and storage.

DataStore is the single source of truth for scheduling entities.
It replaces the copy-paste loaders in run_helpers, standard_run,
and parallel_worker with one ``from_json()`` class method.

Usage::

    store = DataStore.from_json("data")
    store = DataStore.from_json("data", opening_time="08:00")
    context = store.to_context()      # backward-compat SchedulingContext
    d = store.to_dict()               # for multiprocessing serialization
    store2 = DataStore.from_dict(d)    # reconstruct in worker process
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schedule_engine.domain.course import Course
from schedule_engine.domain.group import Group
from schedule_engine.domain.instructor import Instructor
from schedule_engine.domain.room import Room
from schedule_engine.domain.types import SchedulingContext
from schedule_engine.io.data_loader import (
    derive_cohort_pairs_from_groups,
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from schedule_engine.io.time_system import QuantumTimeSystem

__all__ = ["DataStore"]


@dataclass
class DataStore:
    """Immutable container for all scheduling data.

    Attributes:
        courses: Only courses enrolled by at least one group.
        groups: All groups from Groups.json.
        instructors: All instructors from Instructors.json.
        rooms: All rooms from Rooms.json.
        qts: The QuantumTimeSystem created during loading.
        cohort_pairs: Auto-derived (+ manual override) cohort pairs.
    """

    courses: dict[tuple[str, str], Course]
    groups: dict[str, Group]
    instructors: dict[str, Instructor]
    rooms: dict[str, Room]
    qts: QuantumTimeSystem
    cohort_pairs: list[tuple[str, str]] = field(default_factory=list)

    # Factory

    @classmethod
    def from_json(
        cls,
        data_dir: str | Path,
        *,
        opening_time: str = "10:00",
        closing_time: str = "17:00",
        closed_days: list[str] | None = None,
        operating_hours: dict[str, tuple[str, str] | None] | None = None,
        extra_cohort_pairs: list[tuple[str, str]] | None = None,
    ) -> DataStore:
        """Load all scheduling data from a directory of JSON files.

        Parameters:
            data_dir: Directory containing Course.json, Groups.json,
                      Instructors.json, Rooms.json.
            opening_time: Default day start (ignored if *operating_hours* given).
            closing_time: Default day end (ignored if *operating_hours* given).
            closed_days: Days with no classes (default ``["Saturday"]``).
            operating_hours: Full per-day override; takes precedence over
                             opening_time / closing_time / closed_days.
            extra_cohort_pairs: Manually configured pairs to merge with
                                auto-derived ones.
        """
        data_dir = Path(data_dir)
        closed_days = closed_days or ["Saturday"]

        # ---- QTS ----
        if operating_hours is None:
            all_days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            operating_hours = {
                day: None if day in closed_days else (opening_time, closing_time)
                for day in all_days
            }

        qts = QuantumTimeSystem(operating_hours=operating_hours)

        # Note: QTS is now passed via Timetable context to constraints

        # ---- entities ----
        groups_path = str(data_dir / "Groups.json")
        groups = load_groups(groups_path, qts)

        # Only keep courses enrolled by at least one group.
        enrolled_codes: set[str] = set()
        for g in groups.values():
            enrolled_codes.update(g.enrolled_courses)

        all_courses = load_courses(str(data_dir / "Course.json"))
        courses = {k: c for k, c in all_courses.items() if k[0] in enrolled_codes}

        instructors = load_instructors(str(data_dir / "Instructors.json"), qts)
        rooms = load_rooms(str(data_dir / "Rooms.json"), qts)

        # ---- relationships ----
        link_courses_and_groups(courses, groups)
        link_courses_and_instructors(courses, instructors)

        # ---- cohort pairs ----
        derived_pairs = derive_cohort_pairs_from_groups(groups_path)
        cohort_pairs = _merge_cohort_pairs(
            derived_pairs,
            extra_cohort_pairs or [],
        )

        # Note: Cohort pairs are now accessed via config or passed to build_constraints()

        return cls(
            courses=courses,
            groups=groups,
            instructors=instructors,
            rooms=rooms,
            qts=qts,
            cohort_pairs=cohort_pairs,
        )

    # Convenience

    @property
    def available_quanta(self) -> list[int]:
        """All valid quantum indices for this time system."""
        return list(self.qts.get_all_operating_quanta())

    def to_context(self) -> SchedulingContext:
        """Create a ``SchedulingContext`` for backward-compat callers."""
        # Build family_map once here instead of caching in 3 globals
        family_map = self._build_family_map()
        return SchedulingContext(
            courses=self.courses,
            groups=self.groups,
            instructors=self.instructors,
            rooms=self.rooms,
            available_quanta=self.available_quanta,
            cohort_pairs=self.cohort_pairs,
            family_map=family_map,
        )

    def summary(self) -> str:
        return (
            f"Courses: {len(self.courses)}, Groups: {len(self.groups)}, "
            f"Instructors: {len(self.instructors)}, Rooms: {len(self.rooms)}, "
            f"Quanta: {self.qts.total_quanta}"
        )

    # Serialization  (for multiprocessing / pickling)

    def _build_family_map(self) -> dict[str, set[str]]:
        """Build a family map from the loaded groups.

        Centralises the logic duplicated in repair.py, repair_engine.py,
        and group_hierarchy.py.  If the hierarchy module isn't available,
        returns an empty dict (each group only maps to itself).
        """
        try:
            from schedule_engine.ga.group_hierarchy import (
                analyze_group_hierarchy,
                build_group_family_map,
            )

            hierarchy = analyze_group_hierarchy(self.groups)
            return build_group_family_map(hierarchy)
        except Exception:
            return {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for ``init_worker()``."""
        return {
            "data_dir": None,  # not needed once loaded
            "courses": self.courses,
            "groups": self.groups,
            "instructors": self.instructors,
            "rooms": self.rooms,
            "qts": self.qts,
            "cohort_pairs": self.cohort_pairs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataStore:
        """Reconstruct a DataStore from a serialized dict."""
        return cls(
            courses=d["courses"],
            groups=d["groups"],
            instructors=d["instructors"],
            rooms=d["rooms"],
            qts=d["qts"],
            cohort_pairs=d.get("cohort_pairs", []),
        )


def load_input_data(
    data_dir: str,
    config: Any | None = None,
) -> tuple["QuantumTimeSystem", "SchedulingContext"]:
    """Load and link all input entities via :class:`DataStore`.

    Only includes courses enrolled by at least one group.

    Args:
        data_dir: Directory containing input JSON files.
        config: Config object (cohort_pairs will be read from it).

    Returns:
        Tuple of (QuantumTimeSystem, SchedulingContext).

    Raises:
        ValueError: If config is None.
    """
    import time

    start_time = time.time()

    extra_pairs = list(getattr(config, "cohort_pairs", [])) if config else []
    store = DataStore.from_json(data_dir, extra_cohort_pairs=extra_pairs)

    elapsed = time.time() - start_time
    print(f"[!info] Filtered {len(store.courses)} courses, loading took {elapsed:.2f}s")

    if config is None:
        raise ValueError("Config must be provided")

    context = store.to_context()
    # Attach config to context for callers that expect it.
    context.config = config

    return store.qts, context


# Helpers


def _merge_cohort_pairs(
    derived: list[tuple[str, str]],
    configured: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Merge auto-derived cohort pairs with manual overrides (deduped)."""
    merged: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for pair in (*derived, *configured):
        left, right = pair
        left_c, right_c = left.strip(), right.strip()
        if not left_c or not right_c:
            continue
        canonical = tuple(sorted((left_c.lower(), right_c.lower())))
        if canonical in seen:
            continue
        seen.add(canonical)  # type: ignore[arg-type]
        merged.append((left_c, right_c))

    return merged
