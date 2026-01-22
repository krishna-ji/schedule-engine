"""
Centralized Room Type Compatibility Logic

SINGLE SOURCE OF TRUTH for room type matching across:
- Constraints (hard.py)
- Mutation operators (mutation.py)
- Repair operators (repair.py, repair_selective.py)
- Room entity methods (room.py)

This eliminates 4+ duplicate implementations and ensures consistent behavior.
"""

from __future__ import annotations

__all__ = ["is_room_type_compatible"]


def is_room_type_compatible(required: str, room_type: str) -> bool:
    """
    Check if room type satisfies requirement with flexible compatibility.

    Compatibility Rules:
    - Lecture courses → lecture, classroom, auditorium, seminar, tutorial
    - Practical courses → practical, lab, laboratory, computer_lab, science_lab
    - Exact matches always work

    Args:
        required: Required room type (e.g., "lecture", "practical")
            Should be lowercase and stripped, but will normalize if not.
        room_type: Actual room type (e.g., "lecture", "practical")
            Should be lowercase and stripped, but will normalize if not.

    Returns:
        True if compatible, False otherwise

    Examples:
        >>> is_room_type_compatible("lecture", "classroom")
        True
        >>> is_room_type_compatible("lecture", "auditorium")
        True
        >>> is_room_type_compatible("practical", "lab")
        True
        >>> is_room_type_compatible("practical", "lecture")
        False
        >>> is_room_type_compatible("LECTURE", "Classroom")  # Case insensitive
        True

    Note:
        This function is intentionally lenient to handle real-world scheduling
        flexibility where similar room types can often substitute for each other.
    """
    # Normalize inputs (defensive - callers should already normalize)
    req = required.lower().strip()
    room = room_type.lower().strip()

    # Exact match
    if req == room:
        return True

    # Lecture/theory courses: Accept lecture, classroom, auditorium, seminar, tutorial
    if req in ["lecture", "classroom", "theory"] and room in [
        "lecture",
        "classroom",
        "auditorium",
        "seminar",
        "tutorial",
    ]:
        return True

    # Practical/lab courses: Accept practical, lab variants
    return req in ["practical", "lab", "laboratory"] and room in [
        "practical",
        "lab",
        "laboratory",
        "computer_lab",
        "science_lab",
    ]
