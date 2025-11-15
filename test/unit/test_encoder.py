"""Unit tests for input encoding and data loading."""

import pytest
from unittest.mock import patch, MagicMock

from src.encoder.input_encoder import (
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from src.encoder.quantum_time_system import QuantumTimeSystem


class TestQuantumTimeSystem:
    """Test suite for QuantumTimeSystem."""

    def test_qts_initialization_with_default_quantum(self):
        """Test QTS initializes with 60-minute quanta."""
        qts = QuantumTimeSystem(quantum_minutes=60)
        assert qts.quantum_minutes == 60

    def test_qts_time_to_quantum_conversion(self):
        """Test wall-clock time to quantum conversion."""
        qts = QuantumTimeSystem(quantum_minutes=60)

        # 10:00 AM on Sunday should map to quantum 0
        quantum = qts.time_to_quantum(day=0, hour=10, minute=0)
        assert quantum >= 0

    def test_qts_quantum_to_time_conversion(self):
        """Test quantum to wall-clock time conversion."""
        qts = QuantumTimeSystem(quantum_minutes=60)

        # Convert quantum back to time
        day, hour, minute = qts.quantum_to_time(quantum=0)
        assert 0 <= day < 7
        assert 0 <= hour < 24
        assert 0 <= minute < 60

    def test_qts_roundtrip_conversion(self):
        """Test that time -> quantum -> time is reversible."""
        qts = QuantumTimeSystem(quantum_minutes=60)

        original_day, original_hour = 1, 14  # Monday 2:00 PM
        quantum = qts.time_to_quantum(day=original_day, hour=original_hour, minute=0)
        day, hour, minute = qts.quantum_to_time(quantum)

        assert day == original_day
        assert hour == original_hour
        assert minute == 0

    def test_qts_validates_day_range(self):
        """Test QTS rejects invalid day values."""
        qts = QuantumTimeSystem(quantum_minutes=60)

        with pytest.raises((ValueError, AssertionError)):
            qts.time_to_quantum(day=7, hour=10, minute=0)  # Day 7 invalid

    def test_qts_validates_hour_range(self):
        """Test QTS rejects invalid hour values."""
        qts = QuantumTimeSystem(quantum_minutes=60)

        with pytest.raises((ValueError, AssertionError)):
            qts.time_to_quantum(day=0, hour=24, minute=0)  # Hour 24 invalid


class TestDataLoading:
    """Test suite for JSON data loading functions."""

    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_load_courses_returns_dict(self, mock_json_load, mock_open):
        """Test load_courses returns dictionary keyed by (code, type)."""
        # Mock JSON data
        mock_json_load.return_value = [
            {
                "course_id": "1",
                "course_code": "CS101",
                "course_type": "Theory",
                "hours_per_week": 3,
                "requires_split": False,
            }
        ]

        courses = load_courses("dummy_path.json")

        assert isinstance(courses, dict)
        assert ("CS101", "Theory") in courses
        assert courses[("CS101", "Theory")].course_code == "CS101"

    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_load_groups_returns_list(self, mock_json_load, mock_open):
        """Test load_groups returns list of Group objects."""
        mock_json_load.return_value = [
            {
                "group_id": "G1",
                "group_name": "BEI 077",
                "size": 30,
                "enrolled_courses": ["CS101"],
            }
        ]

        groups = load_groups("dummy_path.json")

        assert isinstance(groups, list)
        assert len(groups) == 1
        assert groups[0].group_id == "G1"

    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_load_instructors_returns_list(self, mock_json_load, mock_open):
        """Test load_instructors returns list of Instructor objects."""
        mock_json_load.return_value = [
            {
                "instructor_id": "I1",
                "instructor_name": "Dr. Smith",
                "assigned_courses": ["CS101"],
            }
        ]

        instructors = load_instructors("dummy_path.json")

        assert isinstance(instructors, list)
        assert len(instructors) == 1
        assert instructors[0].instructor_id == "I1"

    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_load_rooms_returns_list(self, mock_json_load, mock_open):
        """Test load_rooms returns list of Room objects."""
        mock_json_load.return_value = [
            {
                "room_id": "R1",
                "room_name": "Lab 101",
                "capacity": 40,
                "room_type": "Lab",
            }
        ]

        rooms = load_rooms("dummy_path.json")

        assert isinstance(rooms, list)
        assert len(rooms) == 1
        assert rooms[0].room_id == "R1"

    def test_load_courses_with_missing_file_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_courses("nonexistent_file.json")
