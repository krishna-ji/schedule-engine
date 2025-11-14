"""
Room entity model for the timetabling system.
Represents a room with its capacity, type, and availability.
"""

from typing import Set
from dataclasses import dataclass


@dataclass
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
    available_quanta: Set[int]

    def __post_init__(self):
        """Validate room data after initialization."""
        if self.capacity <= 0:
            raise ValueError(f"Room {self.room_id}: capacity must be positive")
        # Note: available_quanta can be empty if room has no specific restrictions

    def can_accommodate_group_size(self, group_size: int) -> bool:
        """Check if room can accommodate a given group size."""
        return self.capacity >= group_size

    def is_suitable_for_course_type(self, required_room_features: str) -> bool:
        """Check if room type matches the required room type for a course."""
        # Exact match
        if self.room_features == required_room_features:
            return True
        # Allow flexibility for certain room types
        flexibility_map = {
            "lecture": ["auditorium", "seminar"],
            "seminar": ["lecture"],
            "lab": ["computer_lab", "science_lab"],
            "computer_lab": ["lab"],
            "science_lab": ["lab"],
        }
        if required_room_features in flexibility_map:
            return self.room_features in flexibility_map[required_room_features]
        return False
