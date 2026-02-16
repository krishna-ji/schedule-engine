"""Phase 6: GA Operator Tests.

Tests crossover, mutation, and NSGA-II selection operators:
    1. CROSSOVER:  structural invariants (course/group/duration preserved)
    2. MUTATION:   never mutates course/group/duration, qualification-aware
    3. NSGA-II:    dominance, non-dominated sorting, crowding distance, selection
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
from src.ga.operators.fast_nsga2 import (
    assign_crowding_distance,
    compare_nsga2,
    dominates,
    fast_nondominated_sort,
    sel_nsga2_fast,
)
from src.ga.operators.mutation import (
    find_suitable_rooms_for_course,
    mutate_gene,
    mutate_individual,
    mutate_time_quanta,
)

# Helper: Fake fitness for NSGA-II tests


class FakeFitness:
    """Minimal fitness object for NSGA-II tests."""

    def __init__(self, values):
        self.values = values
        self.crowding_dist = 0.0
        self.rank = 0


class FakeIndividual:
    """Minimal individual wrapper for NSGA-II tests."""

    def __init__(self, values):
        self.fitness = FakeFitness(values)


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
            assert (
                mutated.instructor_id in qualified
            ), f"Instructor {mutated.instructor_id} not qualified for CS101"

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


# NSGA-II Tests


class TestDominance:
    """Test Pareto dominance (minimization)."""

    def test_strict_dominance(self):
        """(1,1) dominates (2,2)."""
        a = FakeIndividual((1.0, 1.0))
        b = FakeIndividual((2.0, 2.0))
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_no_dominance_tradeoff(self):
        """(1,3) does NOT dominate (2,2) — tradeoff."""
        a = FakeIndividual((1.0, 3.0))
        b = FakeIndividual((2.0, 2.0))
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_equal_no_dominance(self):
        """(1,1) does NOT dominate (1,1) — must be STRICTLY better in at least one."""
        a = FakeIndividual((1.0, 1.0))
        b = FakeIndividual((1.0, 1.0))
        assert dominates(a, b) is False

    def test_dominance_one_better_one_equal(self):
        """(1,2) dominates (1,3) — better in one, equal in other."""
        a = FakeIndividual((1.0, 2.0))
        b = FakeIndividual((1.0, 3.0))
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_three_objectives(self):
        a = FakeIndividual((1.0, 1.0, 1.0))
        b = FakeIndividual((2.0, 2.0, 2.0))
        assert dominates(a, b) is True


class TestNonDominatedSorting:
    """Test fast_nondominated_sort correctness."""

    def test_empty_population(self):
        assert fast_nondominated_sort([]) == []

    def test_single_individual(self):
        pop = [FakeIndividual((1.0, 1.0))]
        fronts = fast_nondominated_sort(pop)
        assert len(fronts) == 1
        assert len(fronts[0]) == 1

    def test_all_nondominated(self):
        """All tradeoffs → single front."""
        pop = [
            FakeIndividual((1.0, 3.0)),
            FakeIndividual((2.0, 2.0)),
            FakeIndividual((3.0, 1.0)),
        ]
        fronts = fast_nondominated_sort(pop)
        assert len(fronts) == 1
        assert len(fronts[0]) == 3

    def test_two_fronts(self):
        """Clear dominated point → two fronts."""
        pop = [
            FakeIndividual((1.0, 1.0)),  # Front 0 (dominates c)
            FakeIndividual((3.0, 0.5)),  # Front 0 (tradeoff with a)
            FakeIndividual((2.0, 2.0)),  # Front 1 (dominated by a)
        ]
        fronts = fast_nondominated_sort(pop)
        assert len(fronts) == 2
        assert len(fronts[0]) == 2  # a and b
        assert len(fronts[1]) == 1  # c

    def test_rank_assignment(self):
        """Each individual should get correct rank."""
        a = FakeIndividual((1.0, 1.0))
        b = FakeIndividual((2.0, 2.0))
        c = FakeIndividual((3.0, 3.0))
        pop = [a, b, c]
        fast_nondominated_sort(pop)
        assert a.fitness.rank == 0
        assert b.fitness.rank == 1
        assert c.fitness.rank == 2

    def test_sorting_deterministic(self):
        """Same population should always produce same fronts."""
        pop1 = [
            FakeIndividual((1.0, 3.0)),
            FakeIndividual((2.0, 2.0)),
            FakeIndividual((3.0, 1.0)),
        ]
        pop2 = [
            FakeIndividual((1.0, 3.0)),
            FakeIndividual((2.0, 2.0)),
            FakeIndividual((3.0, 1.0)),
        ]
        f1 = fast_nondominated_sort(pop1)
        f2 = fast_nondominated_sort(pop2)
        assert len(f1) == len(f2)
        for front1, front2 in zip(f1, f2, strict=False):
            assert len(front1) == len(front2)


class TestCrowdingDistance:
    """Test crowding distance assignment."""

    def test_two_individuals_infinite(self):
        """With ≤2 individuals, all get infinite distance."""
        front = [FakeIndividual((1.0, 3.0)), FakeIndividual((3.0, 1.0))]
        assign_crowding_distance(front)
        assert front[0].fitness.crowding_dist == float("inf")
        assert front[1].fitness.crowding_dist == float("inf")

    def test_boundary_infinite(self):
        """Boundary individuals (min/max per objective) get infinite distance."""
        front = [
            FakeIndividual((1.0, 3.0)),
            FakeIndividual((2.0, 2.0)),
            FakeIndividual((3.0, 1.0)),
        ]
        assign_crowding_distance(front)
        # After sorting, boundary individuals should have inf
        inf_count = sum(1 for i in front if i.fitness.crowding_dist == float("inf"))
        assert inf_count >= 2

    def test_middle_individual_finite(self):
        """Middle individual should have finite positive distance."""
        front = [
            FakeIndividual((1.0, 4.0)),
            FakeIndividual((2.0, 2.0)),
            FakeIndividual((4.0, 1.0)),
        ]
        assign_crowding_distance(front)
        finite = [i for i in front if i.fitness.crowding_dist != float("inf")]
        assert len(finite) >= 1
        for i in finite:
            assert i.fitness.crowding_dist > 0

    def test_empty_front(self):
        """Empty front should not crash."""
        assign_crowding_distance([])  # Should not raise

    def test_identical_objectives_no_crash(self):
        """All equal objectives → range=0, should not crash."""
        front = [
            FakeIndividual((2.0, 2.0)),
            FakeIndividual((2.0, 2.0)),
            FakeIndividual((2.0, 2.0)),
        ]
        assign_crowding_distance(front)  # Should not divide by zero


class TestSelection:
    """Test sel_nsga2_fast selection operator."""

    def test_selects_k(self):
        pop = [FakeIndividual((float(i), float(10 - i))) for i in range(10)]
        selected = sel_nsga2_fast(pop, 5)
        assert len(selected) == 5

    def test_selects_all_if_k_ge_n(self):
        pop = [FakeIndividual((1.0, 1.0))]
        selected = sel_nsga2_fast(pop, 5)
        assert len(selected) == 1

    def test_prefers_front_zero(self):
        """Front 0 individuals should be selected over front 1."""
        a = FakeIndividual((1.0, 1.0))  # Front 0
        b = FakeIndividual((5.0, 5.0))  # Front 1
        pop = [a, b]
        selected = sel_nsga2_fast(pop, 1)
        assert selected[0] is a

    def test_diverse_selection(self):
        """When cutting a front, most diverse (highest crowding) should be kept."""
        pop = [
            FakeIndividual((1.0, 5.0)),  # boundary
            FakeIndividual((2.0, 4.5)),  # close to first
            FakeIndividual((5.0, 1.0)),  # boundary
        ]
        # All in front 0. Select 2 → should keep boundaries
        selected = sel_nsga2_fast(pop, 2)
        assert len(selected) == 2


class TestCompareNSGA2:
    """Test NSGA-II comparison function."""

    def test_lower_rank_preferred(self):
        a = FakeIndividual((1.0,))
        b = FakeIndividual((2.0,))
        a.fitness.rank = 0
        b.fitness.rank = 1
        assert compare_nsga2(a, b) == -1

    def test_higher_rank_worse(self):
        a = FakeIndividual((1.0,))
        b = FakeIndividual((2.0,))
        a.fitness.rank = 1
        b.fitness.rank = 0
        assert compare_nsga2(a, b) == 1

    def test_same_rank_higher_crowding_wins(self):
        a = FakeIndividual((1.0,))
        b = FakeIndividual((2.0,))
        a.fitness.rank = 0
        b.fitness.rank = 0
        a.fitness.crowding_dist = 5.0
        b.fitness.crowding_dist = 2.0
        assert compare_nsga2(a, b) == -1

    def test_equal_returns_zero(self):
        a = FakeIndividual((1.0,))
        b = FakeIndividual((1.0,))
        a.fitness.rank = 0
        b.fitness.rank = 0
        a.fitness.crowding_dist = 3.0
        b.fitness.crowding_dist = 3.0
        assert compare_nsga2(a, b) == 0
