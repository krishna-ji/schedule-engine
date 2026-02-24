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
    HC7: CourseCompleteness        — each course-group must get exactly the required quanta
    HC8: SiblingSameDay            — sub-sessions must not be on the same day
"""

from __future__ import annotations

import pytest
from conftest import (
    assert_constraint_positive,
    assert_constraint_zero,
    count_hard_violations,
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
    make_violation_free_timetable,
)

from src.constraints.constraints import (
    CourseCompleteness,
    InstructorExclusivity,
    InstructorQualifications,
    InstructorTimeAvailability,
    RoomExclusivity,
    RoomSuitability,
    StudentGroupExclusivity,
)
from src.domain.timetable import Timetable

# Cross-constraint sanity check


class TestHardConstraintSanity:
    """Verify the shared test infrastructure produces correct results across all constraints."""

    def test_violation_free_timetable_zero_all_hard(self):
        """make_violation_free_timetable() must have 0 penalty for EVERY hard constraint.
        This catches infrastructure bugs (e.g., rooms with wrong default availability).
        """
        from src.constraints.constraints import HARD_CONSTRAINT_CLASSES

        tt, _ctx = make_violation_free_timetable()
        for c in HARD_CONSTRAINT_CLASSES:
            penalty = c.evaluate(tt)
            assert penalty == 0, (
                f"{c.name} returned {penalty} on violation-free timetable! "
                "Likely a conftest.py infrastructure bug."
            )

    def test_aggregate_hard_count_is_zero(self):
        """count_hard_violations() must return 0 for the violation-free timetable."""
        tt, _ = make_violation_free_timetable()
        total = count_hard_violations(tt)
        assert total == 0, f"Expected 0, got {total}"


# HC1: StudentGroupExclusivity


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


# HC2: InstructorExclusivity


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


# HC3: RoomExclusivity


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


# HC4: InstructorQualifications


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
        assert_constraint_positive(c, tt, expected=1)

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

    def test_course_type_mismatch_same_course_id(self):
        """I1 qualified for CS101-theory but gene is CS101-practical → violation.
        This verifies the lookup is by (course_id, course_type) not just course_id."""
        c = self._make_constraint()
        # Only CS101-theory is defined with I1 qualified
        course_theory = make_course("CS101", course_type="theory", instructors=["I1"])
        # Gene is CS101-practical → course_key=("CS101","practical") not in context
        gene = make_gene(course_id="CS101", course_type="practical", instructor_id="I1")
        ctx = make_context(courses=[course_theory])
        tt = Timetable([gene], ctx)
        # Missing course definition = violation (not just checking instructor)
        assert_constraint_positive(c, tt, expected=1)

    def test_qualified_for_wrong_course_type(self):
        """I1 qualified for CS101-theory but NOT for CS101-practical.
        Both course types exist — verifies per-type qualification."""
        c = self._make_constraint()
        c_theory = make_course("CS101", course_type="theory", instructors=["I1"])
        c_prac = make_course("CS101", course_type="practical", instructors=["I2"])
        gene = make_gene(course_id="CS101", course_type="practical", instructor_id="I1")
        ctx = make_context(
            courses=[c_theory, c_prac],
            instructors=[make_instructor("I1"), make_instructor("I2")],
        )
        tt = Timetable([gene], ctx)
        # I1 is NOT in CS101-practical's qualified list ["I2"]
        assert_constraint_positive(c, tt, expected=1)


# HC5: RoomSuitability


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

    def test_constraint_uses_course_required_features_not_gene_type(self):
        """Verify the constraint checks course.required_room_features,
        not gene.course_type. A practical course requiring 'lab' room in a
        lecture room must be caught."""
        gene = make_gene(course_id="CS101", course_type="practical", room_id="R1")
        course = make_course("CS101", course_type="practical", room_feat="practical")
        room = make_room("R1", features="lecture")
        ctx = make_context(courses=[course], rooms=[room])
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)

    def test_multiple_genes_mixed_compatibility(self):
        """2 compatible + 1 incompatible → exactly 1 violation."""
        genes = [
            make_gene(room_id="R1", start=0, duration=1),
            make_gene(course_id="CS102", room_id="R2", start=0, duration=1),
            make_gene(
                course_id="CS103",
                course_type="practical",
                room_id="R1",
                start=2,
                duration=1,
            ),
        ]
        courses = [
            make_course("CS101", room_feat="lecture", quanta=1),
            make_course("CS102", room_feat="lecture", quanta=1),
            make_course(
                "CS103", course_type="practical", room_feat="practical", quanta=1
            ),
        ]
        rooms = [
            make_room("R1", features="lecture"),
            make_room("R2", features="lecture"),
        ]
        ctx = make_context(courses=courses, rooms=rooms)
        tt = Timetable(genes, ctx)
        # CS101 in lecture room → ok, CS102 in lecture room → ok
        # CS103(practical) in lecture room → violation
        assert_constraint_positive(self.constraint, tt, expected=1)


# HC6: InstructorTimeAvailability


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

    def test_part_time_empty_available_quanta_rejected(self):
        """Part-time with empty available_quanta → Instructor model rejects it.
        This edge case is prevented at the data model level, not the constraint."""
        with pytest.raises(ValueError, match="must have available time slots"):
            make_instructor("I1", is_full_time=False, available_quanta=set())

    def test_boundary_exactly_at_available_edge(self):
        """Gene spans q=2-4, available={2,3} → q=4 is the only violation."""
        inst = make_instructor("I1", is_full_time=False, available_quanta={2, 3})
        gene = make_gene(instructor_id="I1", start=2, duration=3)
        ctx = make_context(
            courses=[make_course("CS101", quanta=3)],
            instructors=[inst],
        )
        tt = Timetable([gene], ctx)
        assert_constraint_positive(self.constraint, tt, expected=1)


# HC8: CourseCompleteness


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

    def test_theory_and_practical_tracked_independently(self):
        """CS101-theory (2q) and CS101-practical (3q) are independent.
        Verifies course_key is (course_id, course_type) not just course_id."""
        c_theory = make_course("CS101", course_type="theory", quanta=2, groups=["G1"])
        c_prac = make_course("CS101", course_type="practical", quanta=3, groups=["G1"])
        genes = [
            make_gene(
                "CS101", course_type="theory", start=0, duration=2, group_ids=["G1"]
            ),
            make_gene(
                "CS101", course_type="practical", start=7, duration=3, group_ids=["G1"]
            ),
        ]
        ctx = make_context(courses=[c_theory, c_prac])
        tt = Timetable(genes, ctx)
        # Both satisfied independently
        assert_constraint_zero(self.constraint, tt)

    def test_theory_satisfied_practical_missing(self):
        """CS101-theory met but CS101-practical under-scheduled → 1 violation."""
        c_theory = make_course("CS101", course_type="theory", quanta=2, groups=["G1"])
        c_prac = make_course("CS101", course_type="practical", quanta=3, groups=["G1"])
        genes = [
            make_gene(
                "CS101", course_type="theory", start=0, duration=2, group_ids=["G1"]
            ),
            make_gene(
                "CS101", course_type="practical", start=7, duration=1, group_ids=["G1"]
            ),
        ]
        ctx = make_context(courses=[c_theory, c_prac])
        tt = Timetable(genes, ctx)
        # theory: 2/2 = ok, practical: 1/3 = violation
        assert_constraint_positive(self.constraint, tt, expected=1)
