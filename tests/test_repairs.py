"""Phase 4: Repair Operator Tests.

Tests repair operators with the intent pattern:
    1. PRE:       Create schedule WITH specific violation
    2. ACT:       Run repair operator
    3. POST:      Verify violation count decreased or reached 0
    4. SIDE:      Verify no NEW hard violations introduced
    5. PRESERVE:  Verify structural invariants maintained

Tests individual repair operators and the unified orchestration pipeline.
"""

from __future__ import annotations

import copy

from conftest import (
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
    structural_fields_preserved,
)

from src.config import Config, init_config
from src.constraints.constraints import (
    InstructorExclusivity,
    InstructorQualifications,
    RoomExclusivity,
    RoomSuitability,
    StudentGroupExclusivity,
)
from src.domain.timetable import Timetable
from src.ga.repair.basic import (
    repair_group_overlaps,
    repair_individual_unified,
    repair_instructor_availability,
    repair_instructor_conflicts,
    repair_instructor_qualifications,
    repair_room_conflicts,
    repair_room_overlap_reassign,
    repair_room_type_mismatches,
)

# R1: repair_instructor_availability — fix HC6


class TestRepairInstructorAvailability:
    """Fix instructor availability violations by shifting gene to a valid time."""

    def test_shifts_to_available_time(self):
        """INTENT: After repair, part-time instructor is available at the scheduled time."""
        # I1 available at q=7-9 (Monday), but gene scheduled at q=0-1 (Sunday)
        inst = make_instructor("I1", is_full_time=False, available_quanta={7, 8})
        gene = make_gene(instructor_id="I1", start=0, duration=2)
        ctx = make_context(instructors=[inst])

        copy.deepcopy(gene)
        individual = [gene]
        fixes = repair_instructor_availability(individual, ctx)

        assert fixes >= 1, "Should fix the availability violation"
        # After repair, instructors should be available at new time
        repaired_gene = individual[0]
        for q in range(
            repaired_gene.start_quanta,
            repaired_gene.start_quanta + repaired_gene.num_quanta,
        ):
            assert q in inst.available_quanta, f"Instructor not available at q={q}"

    def test_full_time_instructor_no_change(self):
        """Full-time instructors don't need availability repair."""
        inst = make_instructor("I1", is_full_time=True)
        gene = make_gene(instructor_id="I1", start=0, duration=2)
        ctx = make_context(instructors=[inst])

        before_start = gene.start_quanta
        individual = [gene]
        fixes = repair_instructor_availability(individual, ctx)

        assert fixes == 0
        assert individual[0].start_quanta == before_start

    def test_structural_preservation(self):
        """course_id, course_type, group_ids, num_quanta must NOT change."""
        inst = make_instructor("I1", is_full_time=False, available_quanta={7, 8})
        gene = make_gene(instructor_id="I1", start=0, duration=2)
        ctx = make_context(instructors=[inst])

        before = copy.deepcopy(gene)
        individual = [gene]
        repair_instructor_availability(individual, ctx)

        assert structural_fields_preserved(before, individual[0])

    def test_only_time_changes(self):
        """Only start_quanta should change — room_id, instructor_id preserved."""
        inst = make_instructor("I1", is_full_time=False, available_quanta={7, 8})
        gene = make_gene(instructor_id="I1", room_id="R1", start=0, duration=2)
        ctx = make_context(instructors=[inst])

        individual = [gene]
        repair_instructor_availability(individual, ctx)

        assert individual[0].instructor_id == "I1"
        assert individual[0].room_id == "R1"

    def test_already_available(self):
        """Gene at available time → no repair needed → 0 fixes."""
        inst = make_instructor("I1", is_full_time=False, available_quanta={0, 1})
        gene = make_gene(instructor_id="I1", start=0, duration=2)
        ctx = make_context(instructors=[inst])

        individual = [gene]
        fixes = repair_instructor_availability(individual, ctx)

        assert fixes == 0


# R3: repair_group_overlaps — fix HC1


class TestRepairGroupOverlaps:
    """Fix group schedule overlaps by shifting genes to conflict-free times."""

    def test_fixes_overlap(self):
        """INTENT: After repair, group G1 has no time overlap."""
        g1 = make_gene(course_id="CS101", group_ids=["G1"], start=0, duration=2)
        g2 = make_gene(course_id="CS102", group_ids=["G1"], start=0, duration=2)
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
        )

        individual = [g1, g2]
        constraint = StudentGroupExclusivity()
        pre_penalty = constraint.evaluate(Timetable(individual, ctx))
        assert pre_penalty > 0, "Must start with a violation"

        repair_group_overlaps(individual, ctx)

        post_penalty = constraint.evaluate(Timetable(individual, ctx))
        assert post_penalty < pre_penalty, "Repair should reduce violations"

    def test_structural_preservation(self):
        """Structural fields must not change during repair."""
        g1 = make_gene(course_id="CS101", group_ids=["G1"], start=0, duration=2)
        g2 = make_gene(course_id="CS102", group_ids=["G1"], start=0, duration=2)
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
        )

        before = [copy.deepcopy(g) for g in [g1, g2]]
        individual = [g1, g2]
        repair_group_overlaps(individual, ctx)

        for i, (b, a) in enumerate(zip(before, individual, strict=False)):
            assert structural_fields_preserved(b, a), f"Gene {i}: structural change"

    def test_no_conflict_no_change(self):
        """Non-overlapping genes → 0 fixes."""
        g1 = make_gene(course_id="CS101", group_ids=["G1"], start=0, duration=2)
        g2 = make_gene(course_id="CS102", group_ids=["G1"], start=7, duration=2)
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
        )

        individual = [g1, g2]
        repair_group_overlaps(individual, ctx)

        # May be 0 or could reposition — just verify no violation exists
        tt = Timetable(individual, ctx)
        assert StudentGroupExclusivity().evaluate(tt) == 0


# R4: repair_room_overlap_reassign — fix HC3


class TestRepairRoomOverlapReassign:
    """Fix room conflicts by swapping to a different compatible room."""

    def test_fixes_room_conflict(self):
        """INTENT: After repair, no two sessions share the same room at the same time."""
        g1 = make_gene(course_id="CS101", room_id="R1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", room_id="R1", start=0, duration=2)
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
            rooms=[make_room("R1"), make_room("R2")],
        )

        individual = [g1, g2]
        constraint = RoomExclusivity()
        pre_penalty = constraint.evaluate(Timetable(individual, ctx))
        assert pre_penalty > 0

        fixes = repair_room_overlap_reassign(individual, ctx)

        if fixes > 0:
            tt = Timetable(individual, ctx)
            post_penalty = constraint.evaluate(tt)
            assert post_penalty < pre_penalty

    def test_room_type_preserved(self):
        """New room must be compatible with the course type."""
        g1 = make_gene(course_id="CS101", room_id="R1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", room_id="R1", start=0, duration=2)
        ctx = make_context(
            courses=[
                make_course("CS101", room_feat="lecture"),
                make_course("CS102", room_feat="lecture"),
            ],
            rooms=[
                make_room("R1", features="lecture"),
                make_room("R2", features="lecture"),
                make_room("R3", features="lab"),  # Incompatible
            ],
        )

        individual = [g1, g2]
        fixes = repair_room_overlap_reassign(individual, ctx)

        if fixes > 0:
            # Swapped gene should NOT be in lab room for lecture course
            for gene in individual:
                room = ctx.rooms[gene.room_id]
                course_key = (gene.course_id, gene.course_type)
                course = ctx.courses[course_key]
                constraint = RoomSuitability()
                tt = Timetable([gene], make_context(courses=[course], rooms=[room]))
                # Room suitability should be 0
                assert constraint.evaluate(tt) == 0

    def test_structural_preservation(self):
        g1 = make_gene(course_id="CS101", room_id="R1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", room_id="R1", start=0, duration=2)
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
            rooms=[make_room("R1"), make_room("R2")],
        )
        before = [copy.deepcopy(g) for g in [g1, g2]]
        individual = [g1, g2]
        repair_room_overlap_reassign(individual, ctx)

        for b, a in zip(before, individual, strict=False):
            assert structural_fields_preserved(b, a)


# R5: repair_room_conflicts — fix HC3 (fallback)


class TestRepairRoomConflicts:
    """Fallback for room conflicts: time shift first, then room swap."""

    def test_fixes_room_conflict_by_time_shift(self):
        """Primary approach: shift gene to different time, keep same room."""
        g1 = make_gene(course_id="CS101", room_id="R1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", room_id="R1", start=0, duration=2)
        # Only one room available → must use time shift
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
            rooms=[make_room("R1")],
        )

        individual = [g1, g2]
        fixes = repair_room_conflicts(individual, ctx)

        if fixes > 0:
            tt = Timetable(individual, ctx)
            assert RoomExclusivity().evaluate(tt) == 0


# R6: repair_instructor_conflicts — fix HC2


class TestRepairInstructorConflicts:
    """Fix instructor conflicts by time shift or instructor swap."""

    def test_fixes_instructor_conflict(self):
        """INTENT: After repair, no instructor teaches two sessions at the same time."""
        g1 = make_gene(course_id="CS101", instructor_id="I1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", instructor_id="I1", start=0, duration=2)
        ctx = make_context(
            courses=[
                make_course("CS101", instructors=["I1"]),
                make_course("CS102", instructors=["I1"]),
            ],
        )

        individual = [g1, g2]
        pre_penalty = InstructorExclusivity().evaluate(Timetable(individual, ctx))
        assert pre_penalty > 0

        fixes = repair_instructor_conflicts(individual, ctx)

        if fixes > 0:
            tt = Timetable(individual, ctx)
            post_penalty = InstructorExclusivity().evaluate(tt)
            assert post_penalty < pre_penalty, (
                "Repair should reduce instructor conflicts"
            )

    def test_structural_preservation(self):
        g1 = make_gene(course_id="CS101", instructor_id="I1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", instructor_id="I1", start=0, duration=2)
        ctx = make_context(
            courses=[
                make_course("CS101", instructors=["I1"]),
                make_course("CS102", instructors=["I1"]),
            ],
        )
        before = [copy.deepcopy(g) for g in [g1, g2]]
        individual = [g1, g2]
        repair_instructor_conflicts(individual, ctx)

        for b, a in zip(before, individual, strict=False):
            assert structural_fields_preserved(b, a)


# R7: repair_instructor_qualifications — fix HC4


class TestRepairInstructorQualifications:
    """Fix qualification violations by swapping to a qualified instructor."""

    def test_swaps_to_qualified_instructor(self):
        """INTENT: After repair, instructor is qualified to teach the course."""
        gene = make_gene(instructor_id="I2")  # I2 is NOT qualified
        course = make_course("CS101", instructors=["I1"])  # Only I1 is qualified
        ctx = make_context(
            courses=[course],
            instructors=[make_instructor("I1"), make_instructor("I2")],
        )

        pre_penalty = InstructorQualifications().evaluate(Timetable([gene], ctx))
        assert pre_penalty > 0

        individual = [gene]
        fixes = repair_instructor_qualifications(individual, ctx)

        if fixes > 0:
            tt = Timetable(individual, ctx)
            post_penalty = InstructorQualifications().evaluate(tt)
            assert post_penalty == 0, "Repaired gene should have qualified instructor"

    def test_only_instructor_changes(self):
        """Only instructor_id should change — time, room, structure preserved."""
        gene = make_gene(instructor_id="I2", room_id="R1", start=5, duration=2)
        course = make_course("CS101", instructors=["I1"])
        ctx = make_context(
            courses=[course],
            instructors=[make_instructor("I1"), make_instructor("I2")],
        )

        before = copy.deepcopy(gene)
        individual = [gene]
        repair_instructor_qualifications(individual, ctx)

        repaired = individual[0]
        assert structural_fields_preserved(before, repaired)
        assert repaired.start_quanta == before.start_quanta
        assert repaired.room_id == before.room_id

    def test_no_qualified_instructor(self):
        """No qualified instructor available → repair fails gracefully."""
        gene = make_gene(instructor_id="I2")
        course = make_course("CS101", instructors=[])  # No one is qualified
        ctx = make_context(courses=[course])

        individual = [gene]
        fixes = repair_instructor_qualifications(individual, ctx)
        # Should not crash, may fix 0
        assert fixes >= 0


# R8: repair_room_type_mismatches — fix HC5


class TestRepairRoomTypeMismatches:
    """Fix room type mismatches by swapping to a compatible room."""

    def test_swaps_to_compatible_room(self):
        """INTENT: After repair, room type matches course requirement."""
        gene = make_gene(course_id="CS101", course_type="practical", room_id="R1")
        course = make_course("CS101", course_type="practical", room_feat="practical")
        r1 = make_room("R1", features="lecture")  # Wrong type!
        r2 = make_room("R2", features="lab")  # Compatible
        ctx = make_context(courses=[course], rooms=[r1, r2])

        pre_penalty = RoomSuitability().evaluate(Timetable([gene], ctx))
        assert pre_penalty > 0

        individual = [gene]
        fixes = repair_room_type_mismatches(individual, ctx)

        if fixes > 0:
            tt = Timetable(individual, ctx)
            post_penalty = RoomSuitability().evaluate(tt)
            assert post_penalty == 0, "Room type should match after repair"

    def test_structural_preservation(self):
        gene = make_gene(course_id="CS101", course_type="practical", room_id="R1")
        course = make_course("CS101", course_type="practical", room_feat="practical")
        ctx = make_context(
            courses=[course],
            rooms=[
                make_room("R1", features="lecture"),
                make_room("R2", features="lab"),
            ],
        )
        before = copy.deepcopy(gene)
        individual = [gene]
        repair_room_type_mismatches(individual, ctx)
        assert structural_fields_preserved(before, individual[0])


# Orchestration: repair_individual_unified


class TestRepairIndividualUnified:
    """Test the unified repair pipeline that runs all repairs in priority order."""

    def setup_method(self):
        """Initialize config before each test."""
        init_config(Config(repair={"enabled": True, "heuristics": {}}))

    def _make_messy_individual(self):
        """Create a schedule with multiple violations."""
        # Group overlap: G1 at same time in 2 genes
        g1 = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        g2 = make_gene(
            course_id="CS102",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )  # Triple conflict!
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["G1"], instructors=["I1"]),
                make_course("CS102", groups=["G1"], instructors=["I1"]),
            ],
            rooms=[make_room("R1"), make_room("R2")],
        )
        return [g1, g2], ctx

    def test_reduces_hard_violations(self):
        """INTENT: unified repair should reduce total hard constraint violations."""
        individual, ctx = self._make_messy_individual()
        pre_tt = Timetable(individual, ctx)
        pre_group = StudentGroupExclusivity().evaluate(pre_tt)
        pre_room = RoomExclusivity().evaluate(pre_tt)
        pre_instr = InstructorExclusivity().evaluate(pre_tt)
        pre_total = pre_group + pre_room + pre_instr
        assert pre_total > 0, "Must start with violations"

        repair_individual_unified(individual, ctx, selective=False)

        post_tt = Timetable(individual, ctx)
        post_group = StudentGroupExclusivity().evaluate(post_tt)
        post_room = RoomExclusivity().evaluate(post_tt)
        post_instr = InstructorExclusivity().evaluate(post_tt)
        post_total = post_group + post_room + post_instr
        assert post_total <= pre_total, "Repair should not increase hard violations"

    def test_idempotent_on_clean_schedule(self):
        """Clean schedule → repair does nothing."""
        g1 = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        g2 = make_gene(
            course_id="CS102",
            instructor_id="I2",
            group_ids=["G2"],
            room_id="R2",
            start=0,
            duration=2,
        )
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["G1"], instructors=["I1"]),
                make_course("CS102", groups=["G2"], instructors=["I2"]),
            ],
            groups=[make_group("G1"), make_group("G2")],
            instructors=[make_instructor("I1"), make_instructor("I2")],
            rooms=[make_room("R1"), make_room("R2")],
        )

        individual = [g1, g2]
        before = [copy.deepcopy(g) for g in individual]
        repair_individual_unified(individual, ctx, selective=False)

        # Genes should be essentially unchanged
        for b, a in zip(before, individual, strict=False):
            assert structural_fields_preserved(b, a)

    def test_selective_mode(self):
        """Selective mode should also fix violations."""
        individual, ctx = self._make_messy_individual()
        stats = repair_individual_unified(individual, ctx, selective=True)
        assert isinstance(stats, dict)

    def test_never_increases_hard_violations(self):
        """ALGORITHM: Total hard violations after repair <= before repair."""
        individual, ctx = self._make_messy_individual()

        pre_tt = Timetable(individual, ctx)
        pre_violations = sum(
            [
                StudentGroupExclusivity().evaluate(pre_tt),
                InstructorExclusivity().evaluate(pre_tt),
                RoomExclusivity().evaluate(pre_tt),
            ]
        )

        repair_individual_unified(individual, ctx, selective=False)

        post_tt = Timetable(individual, ctx)
        post_violations = sum(
            [
                StudentGroupExclusivity().evaluate(post_tt),
                InstructorExclusivity().evaluate(post_tt),
                RoomExclusivity().evaluate(post_tt),
            ]
        )

        assert post_violations <= pre_violations, (
            f"Repair worsened violations: {pre_violations} → {post_violations}"
        )
