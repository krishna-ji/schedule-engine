"""Population initialization for experiment notebooks.

Provides random and smart individual creation functions.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.entities.room import Room
from src.ga.sessiongene import SessionGene

if TYPE_CHECKING:
    from src.notebooks.data_loader import ScheduleData


def get_subsession_durations(quanta_per_week: int, course_type: str) -> list[int]:
    """Break course duration into subsessions.

    Theory courses: Break into 2-quanta blocks (with 1-quanta remainder if odd)
    Practical courses: Single continuous session

    Args:
        quanta_per_week: Total quanta required per week
        course_type: "theory" or "practical"

    Returns:
        List of subsession durations in quanta
    """
    if course_type == "practical":
        return [quanta_per_week]
    else:
        if quanta_per_week % 2 == 0:
            return [2] * (quanta_per_week // 2)
        else:
            return [2] * (quanta_per_week // 2) + [1]


def _room_suitable(room: Room, required: str) -> bool:
    """Check if room is suitable for course type."""
    rf = room.room_features.lower() if room.room_features else ""
    if required == "lecture":
        return "lecture" in rf or "seminar" in rf or rf == ""
    elif required == "practical":
        return any(f in rf for f in ["lab", "studio", "practical", "workshop"])
    return True


def create_random_individual(data: ScheduleData) -> list[SessionGene]:
    """Create one individual with random assignments.

    Args:
        data: Loaded schedule data

    Returns:
        List of SessionGene representing one schedule
    """
    genes: list[SessionGene] = []
    room_ids = list(data.rooms.keys())

    for course_key, group_ids, session_type, num_quanta in data.course_group_pairs:
        course = data.courses.get(course_key)
        if not course:
            continue

        # Get qualified instructors
        qualified = course.qualified_instructor_ids or list(data.instructors.keys())

        # Get suitable rooms
        suitable_rooms = [
            r
            for r in room_ids
            if _room_suitable(data.rooms[r], course.required_room_features)
        ]
        if not suitable_rooms:
            suitable_rooms = room_ids

        # Create genes for each subsession
        for duration in get_subsession_durations(num_quanta, course.course_type):
            genes.append(
                SessionGene(
                    course_id=course_key[0],
                    course_type=course_key[1],
                    instructor_id=random.choice(qualified),
                    group_ids=list(group_ids),
                    room_id=random.choice(suitable_rooms),
                    start_quanta=random.randint(
                        0, max(0, data.qts.total_quanta - duration)
                    ),
                    num_quanta=duration,
                )
            )
    return genes


def create_smart_individual(data: ScheduleData) -> list[SessionGene]:
    """Create individual with conflict-avoiding heuristics.

    Uses simple scheduling heuristics:
    - Tracks instructor schedules to avoid conflicts
    - Tracks room schedules to avoid conflicts
    - Prefers morning slots

    Args:
        data: Loaded schedule data

    Returns:
        List of SessionGene with reduced conflicts
    """
    genes: list[SessionGene] = []
    room_ids = list(data.rooms.keys())

    # Track schedules for conflict avoidance
    instructor_schedule: dict[str, set[int]] = {
        inst_id: set() for inst_id in data.instructors
    }
    room_schedule: dict[str, set[int]] = {room_id: set() for room_id in data.rooms}

    for course_key, group_ids, session_type, num_quanta in data.course_group_pairs:
        course = data.courses.get(course_key)
        if not course:
            continue

        qualified = course.qualified_instructor_ids or list(data.instructors.keys())
        suitable_rooms = [
            r
            for r in room_ids
            if _room_suitable(data.rooms[r], course.required_room_features)
        ]
        if not suitable_rooms:
            suitable_rooms = room_ids

        for duration in get_subsession_durations(num_quanta, course.course_type):
            # Try to find conflict-free assignment
            best_instructor = None
            best_room = None
            best_start = None
            min_conflicts = float("inf")

            # Try multiple random combinations
            for _ in range(10):
                inst = random.choice(qualified)
                room = random.choice(suitable_rooms)
                start = random.randint(0, max(0, data.qts.total_quanta - duration))

                # Count conflicts
                slots = set(range(start, start + duration))
                conflicts = len(slots & instructor_schedule[inst]) + len(
                    slots & room_schedule[room]
                )

                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_instructor = inst
                    best_room = room
                    best_start = start

                if conflicts == 0:
                    break

            # Use best found (or random if none found)
            instructor_id = best_instructor or random.choice(qualified)
            room_id = best_room or random.choice(suitable_rooms)
            start_quanta = (
                best_start
                if best_start is not None
                else random.randint(0, max(0, data.qts.total_quanta - duration))
            )

            # Update schedules
            slots = set(range(start_quanta, start_quanta + duration))
            instructor_schedule[instructor_id].update(slots)
            room_schedule[room_id].update(slots)

            genes.append(
                SessionGene(
                    course_id=course_key[0],
                    course_type=course_key[1],
                    instructor_id=instructor_id,
                    group_ids=list(group_ids),
                    room_id=room_id,
                    start_quanta=start_quanta,
                    num_quanta=duration,
                )
            )
    return genes
