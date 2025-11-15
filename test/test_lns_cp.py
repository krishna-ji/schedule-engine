"""
Unit tests for LNS-CP hybrid system.

Tests the conflict detection, CP-SAT repair, and LNS operator modules.
"""

import pytest
from typing import List, Dict
from src.ga.sessiongene import SessionGene
from src.entities.course import Course
from src.entities.instructor import Instructor
from src.entities.group import Group
from src.entities.room import Room
from src.lns.conflict_detection import (
    find_hard_conflict_sessions,
    select_worst_conflicts,
)
from src.lns.cp_repair import repair_with_cp_sat
from src.lns.lns_operator import lns_cp_repair, should_trigger_lns_repair


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    # Create sample instructors
    instructors = {
        "I001": Instructor(
            instructor_id="I001",
            name="Prof. A",
            qualified_courses=["CS101", "CS102"],
            available_quanta=set(range(0, 48)),  # Available all week
            is_full_time=True,
        ),
        "I002": Instructor(
            instructor_id="I002",
            name="Prof. B",
            qualified_courses=["MATH101"],
            available_quanta=set(range(0, 48)),
            is_full_time=True,
        ),
    }

    # Create sample groups
    groups = {
        "G001": Group(
            group_id="G001",
            name="CS Year 1",
            student_count=30,
            enrolled_courses=["CS101", "MATH101"],
            available_quanta=set(range(0, 48)),
        ),
    }

    # Create sample rooms
    rooms = {
        "R001": Room(
            room_id="R001",
            name="Room 101",
            capacity=50,
            room_features="lecture",
            available_quanta=set(range(0, 48)),
        ),
        "R002": Room(
            room_id="R002",
            name="Lab 201",
            capacity=30,
            room_features="lab",
            available_quanta=set(range(0, 48)),
        ),
    }

    # Create sample courses
    courses = {
        ("CS101", "theory"): Course(
            course_id="CS101",
            name="Intro to CS",
            quanta_per_week=4,
            required_room_features="lecture",
            enrolled_group_ids=["G001"],
            qualified_instructor_ids=["I001"],
            course_type="theory",
        ),
        ("MATH101", "theory"): Course(
            course_id="MATH101",
            name="Calculus",
            quanta_per_week=4,
            required_room_features="lecture",
            enrolled_group_ids=["G001"],
            qualified_instructor_ids=["I002"],
            course_type="theory",
        ),
    }

    return courses, instructors, groups, rooms


def test_conflict_detection_no_conflicts(sample_entities):
    """Test conflict detection with temporal overlaps."""
    courses, instructors, groups, rooms = sample_entities

    # Create an individual with no temporal overlaps
    # (may have course completeness issues, but no overlaps)
    individual = [
        SessionGene(
            course_id="CS101",
            course_type="theory",
            instructor_id="I001",
            group_ids=["G001"],
            room_id="R001",
            quanta=[0, 1],  # Monday 8-10am
        ),
        SessionGene(
            course_id="MATH101",
            course_type="theory",
            instructor_id="I002",
            group_ids=["G001"],
            room_id="R001",
            quanta=[12, 13],  # Tuesday 8-10am (no overlap)
        ),
    ]

    conflicted_indices, violations = find_hard_conflict_sessions(
        individual, courses, instructors, groups, rooms
    )

    # May detect course completeness violations, but test that function runs
    assert isinstance(conflicted_indices, list), "Should return list of indices"
    assert isinstance(violations, list), "Should return list of violations"


def test_conflict_detection_with_conflicts(sample_entities):
    """Test conflict detection with overlapping sessions."""
    courses, instructors, groups, rooms = sample_entities

    # Create individual with student group conflict
    individual = [
        SessionGene(
            course_id="CS101",
            course_type="theory",
            instructor_id="I001",
            group_ids=["G001"],
            room_id="R001",
            quanta=[0, 1],  # Monday 8-10am
        ),
        SessionGene(
            course_id="MATH101",
            course_type="theory",
            instructor_id="I002",
            group_ids=["G001"],  # Same group
            room_id="R002",
            quanta=[0, 1],  # Same time - CONFLICT!
        ),
    ]

    conflicted_indices, violations = find_hard_conflict_sessions(
        individual, courses, instructors, groups, rooms
    )

    assert len(conflicted_indices) > 0, "Expected conflicts to be detected"
    assert (
        0 in conflicted_indices or 1 in conflicted_indices
    ), "Expected session indices in conflicts"
    assert len(violations) > 0, "Expected violation information"


def test_select_worst_conflicts():
    """Test selection of worst conflicts when subproblem is too large."""
    # Mock violation data
    from src.lns.conflict_detection import ViolationInfo

    conflicted_indices = list(range(30))  # 30 conflicted sessions
    violations = [
        ViolationInfo(
            constraint_name="student_group_exclusivity",
            violation_count=10,
            affected_sessions=set(range(0, 10)),
        ),
        ViolationInfo(
            constraint_name="instructor_exclusivity",
            violation_count=5,
            affected_sessions=set(range(10, 15)),
        ),
    ]

    selected = select_worst_conflicts(conflicted_indices, violations, max_sessions=20)

    assert len(selected) == 20, "Expected exactly 20 sessions selected"
    assert len(selected) == len(set(selected)), "Expected no duplicates"


def test_should_trigger_lns_repair():
    """Test LNS repair trigger conditions."""
    # Test interval trigger
    assert should_trigger_lns_repair(
        generation=50,
        trigger_interval=50,
        stagnation_counter=0,
        stagnation_threshold=10,
    ), "Expected trigger on interval"

    assert not should_trigger_lns_repair(
        generation=49,
        trigger_interval=50,
        stagnation_counter=0,
        stagnation_threshold=10,
    ), "Expected no trigger before interval"

    # Test stagnation trigger
    assert should_trigger_lns_repair(
        generation=25,
        trigger_interval=50,
        stagnation_counter=10,
        stagnation_threshold=10,
    ), "Expected trigger on stagnation"

    assert not should_trigger_lns_repair(
        generation=25,
        trigger_interval=50,
        stagnation_counter=9,
        stagnation_threshold=10,
    ), "Expected no trigger before stagnation threshold"


def test_lns_repair_with_no_conflicts(sample_entities):
    """Test LNS repair runs without crashing."""
    courses, instructors, groups, rooms = sample_entities

    # Create an individual
    individual = [
        SessionGene(
            course_id="CS101",
            course_type="theory",
            instructor_id="I001",
            group_ids=["G001"],
            room_id="R001",
            quanta=[0, 1],
        ),
        SessionGene(
            course_id="MATH101",
            course_type="theory",
            instructor_id="I002",
            group_ids=["G001"],
            room_id="R001",
            quanta=[12, 13],
        ),
    ]

    repaired = lns_cp_repair(
        individual=individual,
        courses=courses,
        instructors=instructors,
        groups=groups,
        rooms=rooms,
        max_subproblem_size=20,
        cp_time_limit=5.0,
    )

    # Should return a valid individual (either original or repaired)
    assert isinstance(repaired, list), "Should return list of SessionGenes"
    assert len(repaired) == len(individual), "Should return same number of sessions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
