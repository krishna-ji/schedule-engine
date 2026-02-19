"""Phase 6: GA Operator Tests.

Tests crossover and mutation operators:
    1. CROSSOVER:  structural invariants (course/group/duration preserved)
    2. MUTATION:   never mutates course/group/duration, qualification-aware
"""

from __future__ import annotations

import copy
import random

import pytest
from conftest import (
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
)

from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import (
    find_suitable_rooms_for_course,
    mutate_gene,
    mutate_individual,
    mutate_time_quanta,
)

# Crossover Tests


class TestCrossover:
    """Test crossover_course_group_aware operator."""

    def _make_pair(self):
        """Create two individuals with same course-group structure."""
        ind1 = [
            make_gene(
                course_id="CS101",
                instructor_id="I1",
                group_ids=["G1"],
                room_id="R1",
                start=0,
                duration=2,
            ),
            make_gene(
                course_id="CS102",
                instructor_id="I2",
                group_ids=["G2"],
                room_id="R2",
                start=7,
                duration=3,
            ),
            make_gene(
                course_id="CS103",
                instructor_id="I1",
                group_ids=["G1", "G2"],
                room_id="R3",
                start=14,
                duration=1,
            ),
        ]
        ind2 = [
            make_gene(
                course_id="CS101",
                instructor_id="I3",
                group_ids=["G1"],
                room_id="R4",
                start=21,
                duration=2,
            ),
            make_gene(
                course_id="CS102",
                instructor_id="I4",
                group_ids=["G2"],
                room_id="R5",
                start=28,
                duration=3,
            ),
            make_gene(
                course_id="CS103",
                instructor_id="I3",
                group_ids=["G1", "G2"],
                room_id="R6",
                start=35,
                duration=1,
            ),
        ]
        return ind1, ind2

    def test_preserves_course_ids(self):
        """INVARIANT: course_id never changes during crossover."""
        ind1, ind2 = self._make_pair()
        before1 = {g.course_id for g in ind1}
        before2 = {g.course_id for g in ind2}

        random.seed(42)
        crossover_course_group_aware(ind1, ind2, cx_prob=1.0)

        after1 = {g.course_id for g in ind1}
        after2 = {g.course_id for g in ind2}
        assert before1 == after1
        assert before2 == after2

    def test_preserves_group_ids(self):
        """INVARIANT: group_ids never change during crossover."""
        ind1, ind2 = self._make_pair()
        before1 = [tuple(sorted(g.group_ids)) for g in ind1]
        before2 = [tuple(sorted(g.group_ids)) for g in ind2]

        random.seed(42)
        crossover_course_group_aware(ind1, ind2, cx_prob=1.0)

        after1 = [tuple(sorted(g.group_ids)) for g in ind1]
        after2 = [tuple(sorted(g.group_ids)) for g in ind2]
        assert sorted(before1) == sorted(after1)
        assert sorted(before2) == sorted(after2)

    def test_preserves_duration(self):
        """INVARIANT: num_quanta (duration) never changes during crossover."""
        ind1, ind2 = self._make_pair()
        before1 = sorted([g.num_quanta for g in ind1])
        before2 = sorted([g.num_quanta for g in ind2])

        random.seed(42)
        crossover_course_group_aware(ind1, ind2, cx_prob=1.0)

        after1 = sorted([g.num_quanta for g in ind1])
        after2 = sorted([g.num_quanta for g in ind2])
        assert before1 == after1
        assert before2 == after2

    def test_swaps_mutable_attributes(self):
        """With cx_prob=1.0, all mutable attributes should swap."""
        ind1, ind2 = self._make_pair()
        before1_instructors = {g.course_id: g.instructor_id for g in ind1}
        before2_instructors = {g.course_id: g.instructor_id for g in ind2}

        random.seed(0)  # ensure deterministic
        crossover_course_group_aware(ind1, ind2, cx_prob=1.0)

        after1_instructors = {g.course_id: g.instructor_id for g in ind1}
        # After swap: ind1's instructors should be ind2's original ones
        for cid in before1_instructors:
            assert after1_instructors[cid] == before2_instructors[cid]

    def test_cx_prob_zero_no_change(self):
        """With cx_prob=0.0, no attributes should swap."""
        ind1, ind2 = self._make_pair()
        before1 = [copy.deepcopy(g) for g in ind1]
        [copy.deepcopy(g) for g in ind2]

        crossover_course_group_aware(ind1, ind2, cx_prob=0.0)

        for b, a in zip(before1, ind1, strict=False):
            assert b.instructor_id == a.instructor_id
            assert b.room_id == a.room_id
            assert b.start_quanta == a.start_quanta

    def test_mismatched_structure_raises(self):
        """Validation should reject individuals with different course-group pairs."""
        ind1 = [make_gene(course_id="CS101", group_ids=["G1"])]
        ind2 = [make_gene(course_id="CS999", group_ids=["G2"])]

        with pytest.raises(ValueError, match="mismatched"):
            crossover_course_group_aware(ind1, ind2, validate=True)

    def test_mismatched_no_validate(self):
        """Without validation, mismatched structures handled gracefully."""
        ind1 = [make_gene(course_id="CS101", group_ids=["G1"])]
        ind2 = [make_gene(course_id="CS999", group_ids=["G2"])]

        # Should not raise
        result = crossover_course_group_aware(ind1, ind2, validate=False)
        assert len(result) == 2

    def test_returns_tuple_of_two(self):
        ind1, ind2 = self._make_pair()
        result = crossover_course_group_aware(ind1, ind2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_time_bounds_clipped(self):
        """Start quanta should be clipped to valid range after swap."""
        from src.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        # Gene at very end of range
        ind1 = [make_gene(course_id="CS101", group_ids=["G1"], start=0, duration=2)]
        ind2 = [
            make_gene(
                course_id="CS101",
                group_ids=["G1"],
                start=qts.total_quanta - 1,
                duration=2,
            )
        ]

        crossover_course_group_aware(ind1, ind2, cx_prob=1.0)

        # After swap, start_quanta should be clipped so session fits
        for g in ind1:
            assert g.start_quanta + g.num_quanta - 1 < qts.total_quanta
        for g in ind2:
            assert g.start_quanta + g.num_quanta - 1 < qts.total_quanta


# Mutation Tests


class TestMutation:
    """Test mutation operators: course, group, duration NEVER mutated."""

    def _make_ctx(self):
        return make_context(
            courses=[
                make_course("CS101", groups=["G1"], instructors=["I1", "I2"]),
                make_course("CS102", groups=["G2"], instructors=["I2", "I3"]),
            ],
            groups=[make_group("G1", students=30), make_group("G2", students=25)],
            instructors=[
                make_instructor("I1"),
                make_instructor("I2"),
                make_instructor("I3"),
            ],
            rooms=[
                make_room("R1", capacity=50),
                make_room("R2", capacity=40),
                make_room("R3", capacity=60),
            ],
        )

    def test_mutate_gene_preserves_course_id(self):
        """INVARIANT: course_id NEVER mutated."""
        ctx = self._make_ctx()
        g = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        mutated = mutate_gene(g, ctx)
        assert mutated.course_id == "CS101"

    def test_mutate_gene_preserves_group_ids(self):
        """INVARIANT: group_ids NEVER mutated."""
        ctx = self._make_ctx()
        g = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        mutated = mutate_gene(g, ctx)
        assert mutated.group_ids == ["G1"]

    def test_mutate_gene_preserves_duration(self):
        """INVARIANT: num_quanta (duration) NEVER mutated."""
        ctx = self._make_ctx()
        g = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        for _ in range(10):
            mutated = mutate_gene(g, ctx)
            assert mutated.num_quanta == 2

    def test_mutate_gene_preserves_course_type(self):
        """INVARIANT: course_type NEVER mutated."""
        ctx = self._make_ctx()
        g = make_gene(
            course_id="CS101",
            course_type="theory",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        mutated = mutate_gene(g, ctx)
        assert mutated.course_type == "theory"

    def test_mutate_gene_selects_qualified_instructor(self):
        """ALGORITHM: mutated instructor should be qualified for the course."""
        ctx = self._make_ctx()
        g = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        qualified = {"I1", "I2"}
        for _ in range(20):
            mutated = mutate_gene(g, ctx)
            assert mutated.instructor_id in qualified, (
                f"Instructor {mutated.instructor_id} not qualified for CS101"
            )

    def test_mutate_time_quanta_preserves_count(self):
        """INVARIANT: number of quanta never changes during time mutation."""
        ctx = self._make_ctx()
        g = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=3,
        )
        course = ctx.courses.get(("CS101", "theory"))
        for _ in range(20):
            new_quanta = mutate_time_quanta(g, course, ctx)
            assert len(new_quanta) == 3, f"Expected 3 quanta, got {len(new_quanta)}"

    def test_find_suitable_rooms_capacity_filter(self):
        """Rooms below group size should be excluded."""
        ctx = make_context(
            courses=[make_course("CS101", groups=["G1"])],
            groups=[make_group("G1", students=45)],
            rooms=[
                make_room("R_small", capacity=20),
                make_room("R_big", capacity=50),
            ],
        )
        suitable = find_suitable_rooms_for_course("CS101", "theory", "G1", ctx)
        assert "R_big" in suitable
        assert "R_small" not in suitable

    def test_find_suitable_rooms_type_filter(self):
        """Practical courses need lab-type rooms."""
        ctx = make_context(
            courses=[
                make_course("CS101", course_type="practical", room_feat="practical")
            ],
            groups=[make_group("G1", students=20)],
            rooms=[
                make_room("R_lecture", features="lecture", capacity=50),
                make_room("R_lab", features="lab", capacity=50),
            ],
        )
        suitable = find_suitable_rooms_for_course("CS101", "practical", "G1", ctx)
        assert "R_lab" in suitable
        assert "R_lecture" not in suitable

    def test_mutate_individual_returns_tuple(self):
        """DEAP compatibility: must return tuple."""
        ctx = self._make_ctx()
        individual = [
            make_gene(
                course_id="CS101",
                instructor_id="I1",
                group_ids=["G1"],
                room_id="R1",
                start=0,
                duration=2,
            ),
        ]
        result = mutate_individual(individual, ctx, guided=False)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_mutate_individual_preserves_length(self):
        """Mutation should never add or remove genes."""
        ctx = self._make_ctx()
        individual = [
            make_gene(
                course_id="CS101",
                instructor_id="I1",
                group_ids=["G1"],
                room_id="R1",
                start=0,
                duration=2,
            ),
            make_gene(
                course_id="CS102",
                instructor_id="I2",
                group_ids=["G2"],
                room_id="R2",
                start=7,
                duration=2,
            ),
        ]
        before_len = len(individual)
        mutate_individual(individual, ctx, guided=False)
        assert len(individual) == before_len
