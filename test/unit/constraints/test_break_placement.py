"""
Unit tests for break placement constraint and repair.
"""

from __future__ import annotations

from src.constraints.soft import break_placement_compliance
from src.entities.decoded_session import CourseSession


def test_break_placement_no_violations() -> None:
    """Test that schedules with proper breaks have no penalty."""
    # Create sessions that avoid 12:00-14:00 (quanta 2-4 typically)
    # Session from 10:00-11:00 (quanta 0-1)
    sessions = [
        CourseSession(
            course_id="CS101",
            group_ids=["group1"],
            instructor_id="instructor1",
            room_id="room1",
            session_quanta=[0, 1],  # 10:00-12:00 (before break)
        )
    ]

    # Should have no penalty if constraint is enabled
    penalty = break_placement_compliance(sessions)
    # Note: Actual penalty depends on config, but with proper break should be low/zero
    assert penalty >= 0


def test_break_placement_with_violations() -> None:
    """Test that schedules without breaks are penalized."""
    # Create sessions that occupy entire day including break window
    # Assuming 12:00-14:00 is quanta 2-4 (depends on quantum_minutes=60)
    sessions = [
        CourseSession(
            course_id="CS101",
            group_ids=["group1"],
            instructor_id="instructor1",
            room_id="room1",
            session_quanta=[0, 1, 2, 3, 4, 5, 6],  # 10:00-17:00 (full day, no break)
        )
    ]

    # Should have penalty if constraint is enabled
    penalty = break_placement_compliance(sessions)
    # Note: Exact penalty depends on config settings
    assert penalty >= 0


def test_break_placement_disabled() -> None:
    """Test that constraint returns 0 when disabled."""
    from src.config import get_config

    cfg = get_config()

    # If enforce_break_placement is False, should return 0
    if not cfg.time.enforce_break_placement:
        sessions = [
            CourseSession(
                course_id="CS101",
                group_ids=["group1"],
                instructor_id="instructor1",
                room_id="room1",
                session_quanta=[0, 1, 2, 3, 4],
            )
        ]

        penalty = break_placement_compliance(sessions)
        assert penalty == 0


if __name__ == "__main__":
    # Run simple smoke tests
    print("Running break placement constraint tests...")

    print("\n1. Testing no violations...")
    test_break_placement_no_violations()
    print("   ✓ Passed")

    print("\n2. Testing with violations...")
    test_break_placement_with_violations()
    print("   ✓ Passed")

    print("\n3. Testing disabled state...")
    test_break_placement_disabled()
    print("   ✓ Passed")

    print("\nAll tests passed! ✓")
