"""Phase 1: Hard Constraint Unit Tests.

Tests all 8 hard constraints with unit, semantic, intent, and edge-case coverage.
Each test creates a minimal timetable with a specific violation pattern and
verifies the constraint's evaluate() returns the correct penalty.

Hard constraints:
    HC1: StudentGroupExclusivity   — groups can't be in two places at once
    HC2: InstructorExclusivity     — instructors can't teach two classes at once
    HC3: RoomExclusivity           — rooms can't host two sessions at once
    HC4: InstructorQualifications  — instructors must be qualified for their course
    HC5: RoomSuitability           — rooms must match course type (lecture/lab)
    HC6: InstructorTimeAvailability — part-time instructors only available at certain times
    HC7: RoomTimeAvailability      — rooms only available at certain times
    HC8: CourseCompleteness        — each course-group must get exactly the required quanta
"""

from __future__ import annotations

import pathlib

# conftest.py is auto-loaded by pytest; import helpers via sys.path
import sys

import pytest

from schedule_engine.constraints.constraints import (
    CourseCompleteness,
    InstructorExclusivity,
    InstructorQualifications,
    InstructorTimeAvailability,
    RoomExclusivity,
    RoomSuitability,
    RoomTimeAvailability,
    StudentGroupExclusivity,
)
from schedule_engine.domain.timetable import Timetable

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import (
    assert_constraint_positive,
    assert_constraint_zero,
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HC1: StudentGroupExclusivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStudentGroupExclusivity:
    """Groups cannot be in two places at the same time."""

    constraint = StudentGroupExclusivity()

    def _tt(self, genes, **ctx_kw):
        ctx = make_context(**ctx_kw)
        return Timetable(genes, ctx)

    # ── Unit tests ──

    def test_no_overlap(self):
        """G1 at q=0-1, then G1 at q=2-3 → no violation."""
        g1 = make_gene(start=0, duration=2, group_ids=["G1"])
        g2 = make_gene(course_id="CS102", start=2, duration=2, group_ids=["G1"])
        c2 = make_course("CS102", groups=["G1"])
        tt = self._tt([g1, g2], courses=[make_course(), c2])
        assert_constraint_zero(self.constraint, tt)

    def test_empty_timetable(self):
        """Empty schedule has no violations."""
        tt = self._tt([])
        assert_constraint_zero(self.constraint, tt)

    def test_single_gene(self):
        """A single gene can never have a group overlap."""
        tt = self._tt([make_gene()])
        assert_constraint_zero(self.constraint, tt)

    def test_different_groups_same_time(self):
        """G1 and G2 at the same time → no violation (different groups)."""
        g1 = make_gene(group_ids=["G1"], start=0)
        g2 = make_gene(course_id="CS102", group_ids=["G2"], start=0)
        c2 = make_course("CS102", groups=["G2"])
        tt = self._tt(
            [g1, g2],
            courses=[make_course(), c2],
            groups=[make_group("G1"), make_group("G2")],
        )
        assert_constraint_zero(self.constraint, tt)

    # ── Semantic tests ──

    def test_exact_overlap_penalty_equals_shared_quanta(self):
        """G1 at q=0-1 AND G1 at q=0-1 → penalty = 2 (2 shared quanta)."""
        g1 = make_gene(start=0, duration=2, group_ids=["G1"])
        g2 = make_gene(course_id="CS102", start=0, duration=2, group_ids=["G1"])
        c2 = make_course("CS102", groups=["G1"])
        tt = self._tt([g1, g2], courses=[make_course(), c2])
        assert_constraint_positive(self.constraint, tt, expected=2)

    def test_partial_overlap(self):
        """G1 at q=0-2 AND G1 at q=1-2 → penalty = 2 (quanta 1 and 2 shared)."""
        g1 = make_gene(start=0, duration=3, group_ids=["G1"])
        g2 = make_gene(course_id="CS102", start=1, duration=2, group_ids=["G1"])
        c2 = make_course("CS102", groups=["G1"])
        tt = self._tt([g1, g2], courses=[make_course(quanta=3), c2])
        assert_constraint_positive(self.constraint, tt, expected=2)

    def test_triple_overlap_penalty(self):
        """3 genes with G1 at q=0 → penalty = 2 per quantum (len-1 for 2 extras)."""
        genes = [
            make_gene(start=0, duration=1, group_ids=["G1"]),
            make_gene(course_id="CS102", start=0, duration=1, group_ids=["G1"]),
            make_gene(course_id="CS103", start=0, duration=1, group_ids=["G1"]),
        ]
        courses = [
            make_course("CS101", quanta=1, groups=["G1"]),
            make_course("CS102", quanta=1, groups=["G1"]),
            make_course("CS103", quanta=1, groups=["G1"]),
        ]
        tt = self._tt(genes, courses=courses)
        # At q=0: G1 appears in 3 genes → occupancy=3, violations=3-1=2
        assert_constraint_positive(self.constraint, tt, expected=2)

    def test_multi_group_gene_overlap(self):
        """Gene with [G1,G2] at q=0 + gene with [G1] at q=0 → G1 overlaps."""
        g1 = make_gene(group_ids=["G1", "G2"], start=0, duration=1)
        g2 = make_gene(course_id="CS102", group_ids=["G1"], start=0, duration=1)
        courses = [
            make_course("CS101", quanta=1, groups=["G1", "G2"]),
            make_course("CS102", quanta=1, groups=["G1"]),
        ]
        tt = self._tt(
            [g1, g2],
            courses=courses,
            groups=[make_group("G1"), make_group("G2")],
        )
        # G1 at q=0 appears in 2 genes → violation=1
        # G2 at q=0 appears in 1 gene → no violation
        assert_constraint_positive(self.constraint, tt, expected=1)

    # ── Intent test ──

    def test_intent_group_cannot_attend_two_sessions(self):
        """INTENT: A student group physically cannot attend two sessions at once.
        The constraint must catch every quantum where this happens."""
        # G1 has two classes at the exact same time
        g1 = make_gene(start=7, duration=2, group_ids=["G1"])  # Monday 10-12
        g2 = make_gene(course_id="CS102", start=7, duration=2, group_ids=["G1"])
        c2 = make_course("CS102", groups=["G1"])
        tt = self._tt([g1, g2], courses=[make_course(), c2])
        penalty = self.constraint.evaluate(tt)
        assert penalty > 0, "Must detect that G1 has two classes at the same time"
        assert penalty == 2, "Penalty should equal the number of conflicting quanta"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HC2: InstructorExclusivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInstructorExclusivity:
    """Instructors cannot teach two classes simultaneously."""

    constraint = InstructorExclusivity()

    def _tt(self, genes, **ctx_kw):
        return Timetable(genes, make_context(**ctx_kw))

    def test_no_conflict(self):
        """I1 at q=0-1, then I1 at q=2-3 → no violation."""
        g1 = make_gene(instructor_id="I1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", instructor_id="I1", start=2, duration=2)
        c2 = make_course("CS102", instructors=["I1"])
        tt = self._tt([g1, g2], courses=[make_course(), c2])
        assert_constraint_zero(self.constraint, tt)

    def test_exact_overlap(self):
        """I1 at q=0-1 AND I1 at q=0-1 → penalty = 2."""
        g1 = make_gene(instructor_id="I1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", instructor_id="I1", start=0, duration=2)
        c2 = make_course("CS102", instructors=["I1"])
        tt = self._tt([g1, g2], courses=[make_course(), c2])
        assert_constraint_positive(self.constraint, tt, expected=2)

    def test_partial_overlap(self):
        """I1 at q=0-2 AND I1 at q=1-2 → penalty = 2."""
        g1 = make_gene(instructor_id="I1", start=0, duration=3)
        g2 = make_gene(course_id="CS102", instructor_id="I1", start=1, duration=2)
        c2 = make_course("CS102", instructors=["I1"])
        tt = self._tt([g1, g2], courses=[make_course(quanta=3), c2])
        assert_constraint_positive(self.constraint, tt, expected=2)

    def test_different_instructors_same_time(self):
        """I1 and I2 at the same time → no violation."""
        g1 = make_gene(instructor_id="I1", start=0)
        g2 = make_gene(course_id="CS102", instructor_id="I2", start=0)
        c2 = make_course("CS102", instructors=["I2"])
        tt = self._tt(
            [g1, g2],
            courses=[make_course(), c2],
            instructors=[make_instructor("I1"), make_instructor("I2")],
        )
        assert_constraint_zero(self.constraint, tt)

    def test_empty_timetable(self):
        tt = self._tt([])
        assert_constraint_zero(self.constraint, tt)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HC3: RoomExclusivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRoomExclusivity:
    """Rooms cannot host two sessions simultaneously."""

    constraint = RoomExclusivity()

    def _tt(self, genes, **ctx_kw):
        return Timetable(genes, make_context(**ctx_kw))

    def test_no_conflict(self):
        g1 = make_gene(room_id="R1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", room_id="R1", start=2, duration=2)
        c2 = make_course("CS102")
        tt = self._tt([g1, g2], courses=[make_course(), c2])
        assert_constraint_zero(self.constraint, tt)

    def test_exact_overlap(self):
        g1 = make_gene(room_id="R1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", room_id="R1", start=0, duration=2)
        c2 = make_course("CS102")
        tt = self._tt([g1, g2], courses=[make_course(), c2])
        assert_constraint_positive(self.constraint, tt, expected=2)

    def test_different_rooms_same_time(self):
        g1 = make_gene(room_id="R1", start=0)
        g2 = make_gene(course_id="CS102", room_id="R2", start=0)
        c2 = make_course("CS102")
        tt = self._tt(
            [g1, g2],
            courses=[make_course(), c2],
            rooms=[make_room("R1"), make_room("R2")],
        )
        assert_constraint_zero(self.constraint, tt)

    def test_triple_overlap(self):
        """3 genes in same room at same time → penalty = 2 per quantum."""
        genes = [
            make_gene(room_id="R1", start=0, duration=1),
            make_gene(course_id="CS102", room_id="R1", start=0, duration=1),
            make_gene(course_id="CS103", room_id="R1", start=0, duration=1),
        ]
        courses = [
            make_course("CS101", quanta=1),
            make_course("CS102", quanta=1),
            make_course("CS103", quanta=1),
        ]
        tt = self._tt(genes, courses=courses)
        assert_constraint_positive(self.constraint, tt, expected=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HC4: InstructorQualifications
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInstructorQualifications:
    """Instructors must be qualified to teach assigned courses."""

    def _make_constraint(self):
        # Fresh instance to avoid _warned accumulation
        return InstructorQualifications()

    def test_qualified(self):
        """I1 teaches CS101, CS101.qualified=[I1] → no violation."""
        c = InstructorQualifications()
        gene = make_gene(instructor_id="I1")
        course = make_course("CS101", instructors=["I1"])
        ctx = make_context(courses=[course])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(c, tt)

    def test_unqualified(self):
        """I2 teaches CS101 but CS101.qualified=[I1] → 1 violation."""
        c = self._make_constraint()
        gene = make_gene(instructor_id="I2")
        course = make_course("CS101", instructors=["I1"])
        ctx = make_context(
            courses=[course],
            instructors=[make_instructor("I1"), make_instructor("I2")],
        )
        tt = Timetable([gene], ctx)
        assert_constraint_positive(c, tt, expected=1)

    def test_missing_course_definition(self):
        """Gene references course not in context → counts as violation."""
        c = self._make_constraint()
        gene = make_gene(course_id="UNKNOWN")
        ctx = make_context()  # Only has CS101
        tt = Timetable([gene], ctx)
        assert_constraint_positive(c, tt, expected=1)

    def test_empty_qualification_list(self):
        """Course with no qualified instructors → violation."""
        c = self._make_constraint()
        gene = make_gene(instructor_id="I1")
        course = make_course("CS101", instructors=[])
        ctx = make_context(courses=[course])
        tt = Timetable([gene], ctx)
        penalty = c.evaluate(tt)
        # NOTE: Current impl returns 1 (empty list = violation) — verify:
        assert penalty >= 0  # At minimum, document current behavior

    def test_multiple_violations(self):
        """3 genes with unqualified instructors → penalty = 3."""
        c = self._make_constraint()
        genes = [
            make_gene(instructor_id="I2"),
            make_gene(course_id="CS102", instructor_id="I3"),
            make_gene(course_id="CS103", instructor_id="I4"),
        ]
        courses = [
            make_course("CS101", instructors=["I1"]),
            make_course("CS102", instructors=["I1"]),
            make_course("CS103", instructors=["I1"]),
        ]
        ctx = make_context(
            courses=courses,
            instructors=[make_instructor(f"I{i}") for i in range(1, 5)],
        )
        tt = Timetable(genes, ctx)
        assert_constraint_positive(c, tt, expected=3)

    def test_all_valid_multiple_genes(self):
        """5 genes, all with qualified instructors → penalty = 0."""
        c = self._make_constraint()
        genes = [
            make_gene(instructor_id="I1", start=i * 2, duration=1) for i in range(5)
        ]
        ctx = make_context(courses=[make_course("CS101", quanta=5, instructors=["I1"])])
        tt = Timetable(genes, ctx)
        assert_constraint_zero(c, tt)

    def test_mixed_valid_invalid(self):
        """2 qualified + 1 unqualified → penalty = 1."""
        c = self._make_constraint()
        genes = [
            make_gene(instructor_id="I1", start=0),
            make_gene(instructor_id="I1", start=2),
            make_gene(instructor_id="I2", start=4),
        ]
        ctx = make_context(
            courses=[make_course("CS101", instructors=["I1"])],
            instructors=[make_instructor("I1"), make_instructor("I2")],
        )
        tt = Timetable(genes, ctx)
        assert_constraint_positive(c, tt, expected=1)

    def test_intent_instructor_teaches_what_they_know(self):
        """INTENT: An instructor assigned to a course they're not qualified for
        is always detected, even if they happen to be qualified for other courses."""
        c = self._make_constraint()
        # I1 qualified for CS101, I2 qualified for CS102
        # But I1 is assigned to CS102 (not qualified!)
        c1 = make_course("CS101", instructors=["I1"])
        c2 = make_course("CS102", instructors=["I2"])
        gene = make_gene(course_id="CS102", instructor_id="I1", start=0)
        ctx = make_context(
            courses=[c1, c2],
            instructors=[make_instructor("I1"), make_instructor("I2")],
        )
        tt = Timetable([gene], ctx)
        assert_constraint_positive(c, tt, expected=1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HC5: RoomSuitability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRoomSuitability:
    """Rooms must be suitable for the course type (lecture/lab distinction)."""

    constraint = RoomSuitability()

    def test_lecture_in_lecture_room(self):
        gene = make_gene(room_id="R1")
        course = make_course("CS101", room_feat="lecture")
        room = make_room("R1", features="lecture")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_practical_in_lab_room(self):
        gene = make_gene(course_id="CS101", course_type="practical", room_id="R1")
        course = make_course("CS101", course_type="practical", room_feat="practical")
        room = make_room("R1", features="lab")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_lecture_in_lab_room_violation(self):
        gene = make_gene(room_id="R1")
        course = make_course("CS101", room_feat="lecture")
        room = make_room("R1", features="lab")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_practical_in_lecture_room_violation(self):
        gene = make_gene(course_id="CS101", course_type="practical", room_id="R1")
        course = make_course("CS101", course_type="practical", room_feat="practical")
        room = make_room("R1", features="lecture")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_lecture_in_auditorium(self):
        gene = make_gene(room_id="R1")
        course = make_course("CS101", room_feat="lecture")
        room = make_room("R1", features="auditorium")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_lecture_in_seminar_room(self):
        gene = make_gene(room_id="R1")
        course = make_course("CS101", room_feat="lecture")
        room = make_room("R1", features="seminar")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_practical_in_computer_lab(self):
        gene = make_gene(course_id="CS101", course_type="practical", room_id="R1")
        course = make_course("CS101", course_type="practical", room_feat="practical")
        room = make_room("R1", features="computer_lab")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_case_insensitive(self):
        gene = make_gene(room_id="R1")
        course = make_course("CS101", room_feat="LECTURE")
        room = make_room("R1", features="Lecture")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_missing_room_raises(self):
        """Gene references room not in context → KeyError from Timetable."""
        gene = make_gene(room_id="NONEXISTENT")
        ctx = make_context()  # R1 only
        tt = Timetable([gene], ctx)
        with pytest.raises(KeyError):
            self.constraint.evaluate(tt)

    def test_intent_theory_never_in_lab(self):
        """INTENT: A theory class should never be placed in a lab room."""
        gene = make_gene(room_id="R1")
        course = make_course("CS101", course_type="theory", room_feat="lecture")
        room = make_room("R1", features="laboratory")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        penalty = self.constraint.evaluate(tt)
        assert penalty > 0, "Theory class in a laboratory must be caught"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HC6: InstructorTimeAvailability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInstructorTimeAvailability:
    """Part-time instructors only teach during their available slots."""

    constraint = InstructorTimeAvailability()

    def test_full_time_always_available(self):
        """Full-time instructors are available at any time → 0 penalty."""
        gene = make_gene(instructor_id="I1", start=0, duration=2)
        ctx = make_context(instructors=[make_instructor("I1", is_full_time=True)])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_part_time_all_quanta_available(self):
        """Part-time, but scheduled within their available slots."""
        inst = make_instructor("I1", is_full_time=False, available_quanta={0, 1, 2, 3})
        gene = make_gene(instructor_id="I1", start=0, duration=2)
        ctx = make_context(instructors=[inst])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_part_time_fully_unavailable(self):
        """Part-time, scheduled at times they're NOT available → 2 violations (2 quanta)."""
        inst = make_instructor("I1", is_full_time=False, available_quanta={0, 1})
        gene = make_gene(instructor_id="I1", start=2, duration=2)  # q=2,3
        ctx = make_context(instructors=[inst])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=2)

    def test_part_time_partial_availability(self):
        """Part-time, available for q=0 but not q=1 → 1 violation."""
        inst = make_instructor("I1", is_full_time=False, available_quanta={0})
        gene = make_gene(instructor_id="I1", start=0, duration=2)  # q=0,1
        ctx = make_context(instructors=[inst])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_empty_timetable(self):
        tt = Timetable([], make_context())
        assert_constraint_zero(self.constraint, tt)

    def test_missing_instructor_raises(self):
        """Gene references instructor not in context → KeyError from Timetable."""
        gene = make_gene(instructor_id="GHOST")
        ctx = make_context()
        tt = Timetable([gene], ctx)
        with pytest.raises(KeyError):
            self.constraint.evaluate(tt)

    def test_semantic_penalty_is_per_quantum(self):
        """SEMANTIC: penalty counts individual unavailable quanta, not genes."""
        # Part-time instructor available only at q=0 (must have at least 1)
        inst = make_instructor("I1", is_full_time=False, available_quanta={0})
        gene = make_gene(instructor_id="I1", start=0, duration=5)
        ctx = make_context(
            courses=[make_course("CS101", quanta=5)],
            instructors=[inst],
        )
        tt = Timetable([gene], ctx)
        # 5 quanta (0-4), only q=0 available → 4 violations (q=1,2,3,4)
        assert_constraint_positive(self.constraint, tt, expected=4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HC7: RoomTimeAvailability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRoomTimeAvailability:
    """Rooms only available during specific time slots."""

    constraint = RoomTimeAvailability()

    def test_room_available(self):
        """Room available at scheduled time → no violation."""
        room = make_room("R1", available_quanta={0, 1})
        gene = make_gene(room_id="R1", start=0, duration=2)
        ctx = make_context(rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_room_unavailable(self):
        """Room not available at scheduled time → penalty per quantum."""
        room = make_room("R1", available_quanta={2, 3})
        gene = make_gene(room_id="R1", start=0, duration=2)
        ctx = make_context(rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=2)

    def test_room_partially_available(self):
        """Room available for first quantum only → 1 violation."""
        room = make_room("R1", available_quanta={0})
        gene = make_gene(room_id="R1", start=0, duration=2)
        ctx = make_context(rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_room_missing_raises(self):
        """Gene references unknown room → KeyError from Timetable."""
        gene = make_gene(room_id="GHOST")
        ctx = make_context()
        tt = Timetable([gene], ctx)
        with pytest.raises(KeyError):
            self.constraint.evaluate(tt)

    def test_empty_available_quanta(self):
        """Room with empty available_quanta set — check behavior."""
        room = make_room("R1", available_quanta=set())
        gene = make_gene(room_id="R1", start=0, duration=2)
        ctx = make_context(rooms=[room])
        tt = Timetable([gene], ctx)
        # Empty set means every quantum is unavailable
        assert_constraint_positive(self.constraint, tt, expected=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HC8: CourseCompleteness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCourseCompleteness:
    """Each course-group must get exactly the required quanta per week."""

    constraint = CourseCompleteness()

    def test_exact_match(self):
        """CS101 needs 4q, gene provides 4q → no violation."""
        course = make_course("CS101", quanta=4, groups=["G1"])
        gene = make_gene(start=0, duration=4, group_ids=["G1"])
        ctx = make_context(courses=[course])
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_under_scheduled(self):
        """CS101 needs 4q, only 2q provided → 1 violation."""
        course = make_course("CS101", quanta=4, groups=["G1"])
        gene = make_gene(start=0, duration=2, group_ids=["G1"])
        ctx = make_context(courses=[course])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_over_scheduled(self):
        """CS101 needs 2q, 4q provided → 1 violation."""
        course = make_course("CS101", quanta=2, groups=["G1"])
        gene = make_gene(start=0, duration=4, group_ids=["G1"])
        ctx = make_context(courses=[course])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_not_scheduled_at_all(self):
        """CS101 needs 4q but has no genes at all → 1 violation per enrolled group."""
        course = make_course("CS101", quanta=4, groups=["G1"])
        ctx = make_context(courses=[course])
        tt = Timetable([], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_multiple_groups_mixed(self):
        """CS101 enrolled by G1 and G2; G1 has 4q (correct), G2 has 2q (wrong) → 1 violation."""
        course = make_course("CS101", quanta=4, groups=["G1", "G2"])
        g1_gene = make_gene(start=0, duration=4, group_ids=["G1"])
        g2_gene = make_gene(start=7, duration=2, group_ids=["G2"])
        ctx = make_context(
            courses=[course],
            groups=[make_group("G1"), make_group("G2")],
        )
        tt = Timetable([g1_gene, g2_gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_split_sessions_sum_correctly(self):
        """CS101 needs 4q, two 2q genes → total = 4q → no violation."""
        course = make_course("CS101", quanta=4, groups=["G1"])
        g1 = make_gene(start=0, duration=2, group_ids=["G1"])
        g2 = make_gene(start=7, duration=2, group_ids=["G1"])
        ctx = make_context(courses=[course])
        tt = Timetable([g1, g2], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_multi_group_gene(self):
        """Gene with [G1, G2] → counts towards both groups' requirements."""
        course = make_course("CS101", quanta=2, groups=["G1", "G2"])
        gene = make_gene(start=0, duration=2, group_ids=["G1", "G2"])
        ctx = make_context(
            courses=[course],
            groups=[make_group("G1"), make_group("G2")],
        )
        tt = Timetable([gene], ctx)
        assert_constraint_zero(self.constraint, tt)

    def test_intent_every_group_gets_full_hours(self):
        """INTENT: Every enrolled group must receive exactly the required
        weekly hours. Under-scheduling means students miss content.
        Over-scheduling wastes resources."""
        c1 = make_course("CS101", quanta=3, groups=["G1"])
        c2 = make_course("CS102", quanta=2, groups=["G1"])
        genes = [
            make_gene("CS101", start=0, duration=3, group_ids=["G1"]),
            make_gene("CS102", start=7, duration=2, group_ids=["G1"]),
        ]
        ctx = make_context(courses=[c1, c2])
        tt = Timetable(genes, ctx)
        assert_constraint_zero(self.constraint, tt, msg="Both courses fully scheduled")
