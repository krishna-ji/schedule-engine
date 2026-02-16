"""Tests for the Timetable class — the pre-indexed schedule view.

Tests cover:
- Construction & basic access
- All 6 index types (occupancy, daily, completeness, course_daily, practical)
- Conflict detection methods
- Edge cases (empty, single gene, overlapping genes)
- Lookup helpers (course_for_gene, instructor_for_gene, etc.)
"""

from __future__ import annotations

import pytest

from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.domain.timetable import ConflictPair, Timetable
from src.domain.types import SchedulingContext

# Fixtures


def _make_course(
    course_id: str = "CS101",
    course_type: str = "theory",
    quanta: int = 2,
    room_feat: str = "lecture",
    groups: list[str] | None = None,
    instructors: list[str] | None = None,
) -> Course:
    return Course(
        course_id=course_id,
        name=f"Course {course_id}",
        quanta_per_week=quanta,
        required_room_features=room_feat,
        enrolled_group_ids=groups or ["G1"],
        qualified_instructor_ids=instructors or ["I1"],
        course_type=course_type,
    )


def _make_instructor(
    instructor_id: str = "I1",
    courses: list | None = None,
) -> Instructor:
    return Instructor(
        instructor_id=instructor_id,
        name=f"Instructor {instructor_id}",
        qualified_courses=courses or [],
    )


def _make_group(
    group_id: str = "G1",
    students: int = 30,
    courses: list[str] | None = None,
) -> Group:
    return Group(
        group_id=group_id,
        name=f"Group {group_id}",
        student_count=students,
        enrolled_courses=courses or ["CS101"],
    )


def _make_room(
    room_id: str = "R1",
    capacity: int = 50,
    features: str = "lecture",
) -> Room:
    return Room(
        room_id=room_id,
        name=f"Room {room_id}",
        capacity=capacity,
        room_features=features,
    )


def _make_gene(
    course_id: str = "CS101",
    course_type: str = "theory",
    instructor_id: str = "I1",
    group_ids: list[str] | None = None,
    room_id: str = "R1",
    start: int = 0,
    duration: int = 2,
) -> SessionGene:
    return SessionGene(
        course_id=course_id,
        course_type=course_type,
        instructor_id=instructor_id,
        group_ids=group_ids or ["G1"],
        room_id=room_id,
        start_quanta=start,
        num_quanta=duration,
    )


def _make_context(
    courses: dict[tuple[str, str], Course] | None = None,
    groups: dict[str, Group] | None = None,
    instructors: dict[str, Instructor] | None = None,
    rooms: dict[str, Room] | None = None,
) -> SchedulingContext:
    if courses is None:
        c = _make_course()
        courses = {(c.course_id, c.course_type): c}
    if groups is None:
        groups = {"G1": _make_group()}
    if instructors is None:
        instructors = {"I1": _make_instructor()}
    if rooms is None:
        rooms = {"R1": _make_room()}
    return SchedulingContext(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        available_quanta=list(range(70)),
    )


# Tests: Construction & basic access


class TestTimetableConstruction:
    def test_empty_timetable(self):
        tt = Timetable([], _make_context())
        assert len(tt) == 0
        assert list(tt) == []
        assert tt.genes == []

    def test_single_gene(self):
        gene = _make_gene(start=0, duration=2)
        tt = Timetable([gene], _make_context())
        assert len(tt) == 1
        assert tt[0] is gene
        assert list(tt) == [gene]

    def test_context_preserved(self):
        ctx = _make_context()
        tt = Timetable([], ctx)
        assert tt.context is ctx

    def test_from_individual_factory(self):
        gene = _make_gene()
        ctx = _make_context()
        tt = Timetable.from_individual([gene], ctx)
        assert len(tt) == 1
        assert tt.context is ctx


# Tests: Per-entity gene lists


class TestGenesByEntity:
    def test_genes_for_group(self):
        g1 = _make_gene(group_ids=["G1"], start=0)
        g2 = _make_gene(group_ids=["G2"], start=2)
        g3 = _make_gene(group_ids=["G1"], start=4)
        ctx = _make_context(groups={"G1": _make_group("G1"), "G2": _make_group("G2")})
        tt = Timetable([g1, g2, g3], ctx)

        assert tt.genes_for_group("G1") == [0, 2]
        assert tt.genes_for_group("G2") == [1]
        assert tt.genes_for_group("G_NONEXISTENT") == []

    def test_genes_for_instructor(self):
        g1 = _make_gene(instructor_id="I1", start=0)
        g2 = _make_gene(instructor_id="I2", start=2)
        ctx = _make_context(
            instructors={"I1": _make_instructor("I1"), "I2": _make_instructor("I2")}
        )
        tt = Timetable([g1, g2], ctx)

        assert tt.genes_for_instructor("I1") == [0]
        assert tt.genes_for_instructor("I2") == [1]

    def test_genes_for_room(self):
        g1 = _make_gene(room_id="R1", start=0)
        g2 = _make_gene(room_id="R2", start=2)
        ctx = _make_context(rooms={"R1": _make_room("R1"), "R2": _make_room("R2")})
        tt = Timetable([g1, g2], ctx)

        assert tt.genes_for_room("R1") == [0]
        assert tt.genes_for_room("R2") == [1]

    def test_genes_at_quantum(self):
        g1 = _make_gene(start=0, duration=3)  # quanta 0, 1, 2
        g2 = _make_gene(
            start=2, duration=2, instructor_id="I2", group_ids=["G2"]
        )  # quanta 2, 3
        ctx = _make_context(
            groups={"G1": _make_group("G1"), "G2": _make_group("G2")},
            instructors={"I1": _make_instructor("I1"), "I2": _make_instructor("I2")},
        )
        tt = Timetable([g1, g2], ctx)

        assert tt.genes_at_quantum(0) == [0]
        assert tt.genes_at_quantum(1) == [0]
        assert set(tt.genes_at_quantum(2)) == {0, 1}  # Both active at q=2
        assert tt.genes_at_quantum(3) == [1]
        assert tt.genes_at_quantum(99) == []  # No gene active


# Tests: Occupancy indexes


class TestOccupancyIndexes:
    def test_group_occupancy_no_conflict(self):
        g1 = _make_gene(group_ids=["G1"], start=0, duration=2)  # q: 0, 1
        g2 = _make_gene(group_ids=["G1"], start=2, duration=2)  # q: 2, 3
        tt = Timetable([g1, g2], _make_context())

        # Each (group, quantum) should have exactly 1 gene
        assert tt.group_occupancy[("G1", 0)] == [0]
        assert tt.group_occupancy[("G1", 1)] == [0]
        assert tt.group_occupancy[("G1", 2)] == [1]
        assert tt.group_occupancy[("G1", 3)] == [1]

    def test_group_occupancy_with_conflict(self):
        g1 = _make_gene(group_ids=["G1"], start=0, duration=2, instructor_id="I1")
        g2 = _make_gene(group_ids=["G1"], start=1, duration=2, instructor_id="I2")
        ctx = _make_context(
            instructors={"I1": _make_instructor("I1"), "I2": _make_instructor("I2")}
        )
        tt = Timetable([g1, g2], ctx)

        # quantum 1 is double-booked for G1
        assert len(tt.group_occupancy[("G1", 1)]) == 2

    def test_instructor_occupancy(self):
        g1 = _make_gene(instructor_id="I1", start=0, duration=2)
        tt = Timetable([g1], _make_context())

        assert tt.instructor_occupancy[("I1", 0)] == [0]
        assert tt.instructor_occupancy[("I1", 1)] == [0]

    def test_room_occupancy(self):
        g1 = _make_gene(room_id="R1", start=0, duration=2)
        tt = Timetable([g1], _make_context())

        assert tt.room_occupancy[("R1", 0)] == [0]
        assert tt.room_occupancy[("R1", 1)] == [0]

    def test_multi_group_gene(self):
        """A gene with group_ids=["G1", "G2"] should appear in both groups' occupancy."""
        g = _make_gene(group_ids=["G1", "G2"], start=0, duration=1)
        ctx = _make_context(groups={"G1": _make_group("G1"), "G2": _make_group("G2")})
        tt = Timetable([g], ctx)

        assert tt.group_occupancy[("G1", 0)] == [0]
        assert tt.group_occupancy[("G2", 0)] == [0]


# Tests: Conflict detection


class TestConflictDetection:
    def test_no_conflicts_empty(self):
        tt = Timetable([], _make_context())
        assert tt.group_conflicts() == []
        assert tt.instructor_conflicts() == []
        assert tt.room_conflicts() == []
        assert tt.all_conflicts() == []

    def test_no_conflicts_non_overlapping(self):
        g1 = _make_gene(start=0, duration=2)
        g2 = _make_gene(
            start=2, duration=2, instructor_id="I2", group_ids=["G2"], room_id="R2"
        )
        ctx = _make_context(
            groups={"G1": _make_group("G1"), "G2": _make_group("G2")},
            instructors={"I1": _make_instructor("I1"), "I2": _make_instructor("I2")},
            rooms={"R1": _make_room("R1"), "R2": _make_room("R2")},
        )
        tt = Timetable([g1, g2], ctx)
        assert tt.all_conflicts() == []

    def test_group_conflict_detected(self):
        g1 = _make_gene(
            group_ids=["G1"], start=0, duration=2, instructor_id="I1", room_id="R1"
        )
        g2 = _make_gene(
            group_ids=["G1"], start=1, duration=2, instructor_id="I2", room_id="R2"
        )
        ctx = _make_context(
            instructors={"I1": _make_instructor("I1"), "I2": _make_instructor("I2")},
            rooms={"R1": _make_room("R1"), "R2": _make_room("R2")},
        )
        tt = Timetable([g1, g2], ctx)

        conflicts = tt.group_conflicts()
        assert len(conflicts) >= 1
        assert all(c.resource_type == "group" for c in conflicts)
        # quantum 1 is the conflict point
        assert any(c.quantum == 1 for c in conflicts)

    def test_instructor_conflict_detected(self):
        g1 = _make_gene(
            instructor_id="I1", start=0, duration=2, group_ids=["G1"], room_id="R1"
        )
        g2 = _make_gene(
            instructor_id="I1", start=1, duration=2, group_ids=["G2"], room_id="R2"
        )
        ctx = _make_context(
            groups={"G1": _make_group("G1"), "G2": _make_group("G2")},
            rooms={"R1": _make_room("R1"), "R2": _make_room("R2")},
        )
        tt = Timetable([g1, g2], ctx)

        conflicts = tt.instructor_conflicts()
        assert len(conflicts) >= 1
        assert all(c.resource_type == "instructor" for c in conflicts)

    def test_room_conflict_detected(self):
        g1 = _make_gene(
            room_id="R1", start=0, duration=2, group_ids=["G1"], instructor_id="I1"
        )
        g2 = _make_gene(
            room_id="R1", start=1, duration=2, group_ids=["G2"], instructor_id="I2"
        )
        ctx = _make_context(
            groups={"G1": _make_group("G1"), "G2": _make_group("G2")},
            instructors={"I1": _make_instructor("I1"), "I2": _make_instructor("I2")},
        )
        tt = Timetable([g1, g2], ctx)

        conflicts = tt.room_conflicts()
        assert len(conflicts) >= 1
        assert all(c.resource_type == "room" for c in conflicts)

    def test_violation_counts(self):
        # Two genes sharing G1 and overlapping at quantum 1
        g1 = _make_gene(
            group_ids=["G1"], start=0, duration=2, instructor_id="I1", room_id="R1"
        )
        g2 = _make_gene(
            group_ids=["G1"], start=1, duration=2, instructor_id="I2", room_id="R2"
        )
        ctx = _make_context(
            instructors={"I1": _make_instructor("I1"), "I2": _make_instructor("I2")},
            rooms={"R1": _make_room("R1"), "R2": _make_room("R2")},
        )
        tt = Timetable([g1, g2], ctx)

        # 1 quantum of overlap for group G1
        assert tt.count_group_violations() == 1
        # No instructor or room overlap
        assert tt.count_instructor_violations() == 0
        assert tt.count_room_violations() == 0


# Tests: Completeness map


class TestCompletenessMap:
    def test_single_gene_quanta_counted(self):
        gene = _make_gene(
            course_id="CS101",
            course_type="theory",
            group_ids=["G1"],
            start=0,
            duration=3,
        )
        tt = Timetable([gene], _make_context())

        assert tt.course_group_quanta[("CS101", "theory", "G1")] == 3

    def test_multiple_genes_accumulated(self):
        g1 = _make_gene(
            course_id="CS101",
            course_type="theory",
            group_ids=["G1"],
            start=0,
            duration=2,
        )
        g2 = _make_gene(
            course_id="CS101",
            course_type="theory",
            group_ids=["G1"],
            start=4,
            duration=3,
        )
        tt = Timetable([g1, g2], _make_context())

        assert tt.course_group_quanta[("CS101", "theory", "G1")] == 5

    def test_multi_group_gene(self):
        gene = _make_gene(group_ids=["G1", "G2"], start=0, duration=2)
        ctx = _make_context(groups={"G1": _make_group("G1"), "G2": _make_group("G2")})
        tt = Timetable([gene], ctx)

        assert tt.course_group_quanta[("CS101", "theory", "G1")] == 2
        assert tt.course_group_quanta[("CS101", "theory", "G2")] == 2

    def test_missing_key_not_in_map(self):
        tt = Timetable([], _make_context())
        assert ("CS999", "theory", "G1") not in tt.course_group_quanta


# Tests: Practical quanta


class TestPracticalQuanta:
    def test_theory_gene_not_tracked(self):
        gene = _make_gene(course_type="theory", start=0, duration=2)
        tt = Timetable([gene], _make_context())
        assert len(tt.practical_quanta) == 0

    def test_practical_gene_tracked(self):
        course = _make_course(
            course_id="CS101", course_type="practical", room_feat="lab"
        )
        gene = _make_gene(
            course_id="CS101",
            course_type="practical",
            group_ids=["G1"],
            start=0,
            duration=3,
        )
        ctx = _make_context(courses={("CS101", "practical"): course})
        tt = Timetable([gene], ctx)

        key = ("CS101", "practical", "G1")
        assert key in tt.practical_quanta
        assert tt.practical_quanta[key] == {0, 1, 2}


# Tests: Lookup helpers


class TestLookupHelpers:
    def test_course_for_gene(self):
        course = _make_course(course_id="CS101", course_type="theory")
        gene = _make_gene(course_id="CS101", course_type="theory")
        ctx = _make_context(courses={("CS101", "theory"): course})
        tt = Timetable([gene], ctx)

        assert tt.course_for_gene(gene) is course

    def test_instructor_for_gene(self):
        inst = _make_instructor("I1")
        gene = _make_gene(instructor_id="I1")
        ctx = _make_context(instructors={"I1": inst})
        tt = Timetable([gene], ctx)

        assert tt.instructor_for_gene(gene) is inst

    def test_room_for_gene(self):
        room = _make_room("R1")
        gene = _make_gene(room_id="R1")
        ctx = _make_context(rooms={"R1": room})
        tt = Timetable([gene], ctx)

        assert tt.room_for_gene(gene) is room

    def test_groups_for_gene(self):
        g1 = _make_group("G1")
        g2 = _make_group("G2")
        gene = _make_gene(group_ids=["G1", "G2"])
        ctx = _make_context(groups={"G1": g1, "G2": g2})
        tt = Timetable([gene], ctx)

        result = tt.groups_for_gene(gene)
        assert len(result) == 2
        assert result[0] is g1
        assert result[1] is g2


# Tests: Daily indexes (without QTS — should be empty)


class TestDailyWithoutQTS:
    """When no QTS is provided, daily indexes should be empty."""

    def test_group_daily_empty_without_qts(self):
        gene = _make_gene(start=0, duration=2)
        tt = Timetable([gene], _make_context(), qts=None)
        assert tt.group_daily == {}

    def test_instructor_daily_empty_without_qts(self):
        gene = _make_gene(start=0, duration=2)
        tt = Timetable([gene], _make_context(), qts=None)
        assert tt.instructor_daily == {}

    def test_course_daily_empty_without_qts(self):
        gene = _make_gene(start=0, duration=2)
        tt = Timetable([gene], _make_context(), qts=None)
        assert tt.course_daily == {}


# Tests: Daily indexes (with QTS)


class TestDailyWithQTS:
    """When QTS is provided, daily indexes should be populated."""

    @pytest.fixture()
    def qts(self):
        from src.io.time_system import QuantumTimeSystem

        return QuantumTimeSystem()

    def test_group_daily_populated(self, qts):
        # quantum 0 should be day "Sun", within_day=0 (with default QTS)
        gene = _make_gene(group_ids=["G1"], start=0, duration=2)
        tt = Timetable([gene], _make_context(), qts=qts)

        assert "G1" in tt.group_daily
        # At least one day should have entries
        total_entries = sum(len(v) for v in tt.group_daily["G1"].values())
        assert total_entries == 2  # 2 quanta → 2 within-day entries

    def test_instructor_daily_populated(self, qts):
        gene = _make_gene(instructor_id="I1", start=0, duration=3)
        tt = Timetable([gene], _make_context(), qts=qts)

        assert "I1" in tt.instructor_daily
        total_entries = sum(len(v) for v in tt.instructor_daily["I1"].values())
        assert total_entries == 3

    def test_course_daily_populated(self, qts):
        gene = _make_gene(course_id="CS101", course_type="theory", start=0, duration=2)
        tt = Timetable([gene], _make_context(), qts=qts)

        key = ("CS101", "theory")
        assert key in tt.course_daily
        total_entries = sum(len(v) for v in tt.course_daily[key].values())
        assert total_entries == 2


# Tests: ConflictPair dataclass


class TestConflictPair:
    def test_frozen(self):
        cp = ConflictPair(
            gene_a_idx=0,
            gene_b_idx=1,
            resource_type="group",
            resource_id="G1",
            quantum=5,
        )
        with pytest.raises(AttributeError):
            cp.gene_a_idx = 99  # type: ignore[misc]

    def test_equality(self):
        cp1 = ConflictPair(0, 1, "group", "G1", 5)
        cp2 = ConflictPair(0, 1, "group", "G1", 5)
        assert cp1 == cp2

    def test_hashable(self):
        cp = ConflictPair(0, 1, "group", "G1", 5)
        s = {cp}
        assert len(s) == 1


# Tests: Edge cases


class TestEdgeCases:
    def test_gene_with_zero_duration_gets_clamped(self):
        """SessionGene.__post_init__ clamps num_quanta to 1 if <= 0."""
        gene = _make_gene(start=0, duration=0)
        # SessionGene should clamp to 1
        assert gene.num_quanta >= 1

        tt = Timetable([gene], _make_context())
        assert len(tt) == 1

    def test_many_genes_same_slot(self):
        """Stress: 10 genes all at quantum 0."""
        genes = [
            _make_gene(
                instructor_id=f"I{i}",
                group_ids=[f"G{i}"],
                room_id=f"R{i}",
                start=0,
                duration=1,
            )
            for i in range(10)
        ]
        ctx = _make_context(
            groups={f"G{i}": _make_group(f"G{i}") for i in range(10)},
            instructors={f"I{i}": _make_instructor(f"I{i}") for i in range(10)},
            rooms={f"R{i}": _make_room(f"R{i}") for i in range(10)},
        )
        tt = Timetable(genes, ctx)

        # No conflicts since all use different resources
        assert tt.count_group_violations() == 0
        assert tt.count_instructor_violations() == 0
        assert tt.count_room_violations() == 0

        # But all 10 genes are active at quantum 0
        assert len(tt.genes_at_quantum(0)) == 10
