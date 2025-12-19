"""Room entity model for the timetabling system."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Room"]


@dataclass(slots=True)
class Room:
    """
    Represents a room in the university timetabling system.

    Attributes:
        room_id: Unique identifier for the room
        name: Display name of the room
        capacity: Maximum number of students the room can accommodate
        room_features: Type of room (e.g., 'lecture', 'lab', 'seminar', 'auditorium')
        available_quanta: Set of available quantum time slots
    """

    room_id: str
    name: str
    capacity: int
    room_features: str
    available_quanta: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate room data after initialization."""
        if self.capacity <= 0:
            raise ValueError(f"Room {self.room_id}: capacity must be positive")
        # Note: available_quanta can be empty if room has no specific restrictions

    def can_accommodate_group_size(self, group_size: int) -> bool:
        """Check if room can accommodate a given group size."""
        return self.capacity >= group_size

    def is_suitable_for_course_type(self, required_room_features: str) -> bool:
        """Check if room type matches the required room type for a course."""
        from src.utils.room_compatibility import is_room_type_compatible

        return is_room_type_compatible(required_room_features, self.room_features)
