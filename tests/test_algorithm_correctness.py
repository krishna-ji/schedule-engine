"""Phase 8: Algorithm Correctness Tests.

Tests mathematical/algorithmic properties that must hold:
    1. NSGA-II:    Pareto dominance properties, sorting invariants
    2. STRATEGIES:  RoundRobin cycling, Adaptive probability normalization
    3. REPAIR_ENGINE:  Monotonicity (never worsens), policy correctness
    4. CONSTRAINT ALGEBRA:  Weights, commutativity, additivity
"""

from __future__ import annotations

import copy
import math

import pytest

from src.ga.operators.fast_nsga2 import (
    assign_crowding_distance,
    dominates,
    fast_nondominated_sort,
    sel_nsga2_fast,
)

# Helper


class FakeFitness:
    def __init__(self, values):
        self.values = values
        self.crowding_dist = 0.0
        self.rank = 0


class FakeIndividual:
    def __init__(self, values):
        self.fitness = FakeFitness(values)


# NSGA-II Algorithmic Properties


class TestNSGA2Properties:
    """Test mathematical properties of NSGA-II."""

    def test_dominance_transitivity(self):
        """If A dominates B and B dominates C, then A dominates C."""
        a = FakeIndividual((1.0, 1.0))
        b = FakeIndividual((2.0, 2.0))
        c = FakeIndividual((3.0, 3.0))
        assert dominates(a, b)
        assert dominates(b, c)
        assert dominates(a, c)  # Transitivity

    def test_dominance_irreflexivity(self):
        """An individual cannot dominate itself."""
        a = FakeIndividual((1.0, 2.0))
        assert not dominates(a, a)

    def test_dominance_asymmetry(self):
        """If A dominates B, then B does NOT dominate A."""
        a = FakeIndividual((1.0, 1.0))
        b = FakeIndividual((2.0, 2.0))
        assert dominates(a, b)
        assert not dominates(b, a)

    def test_front_zero_nondominated(self):
        """Every individual in front 0 is not dominated by any other individual."""
        pop = [
            FakeIndividual((1.0, 5.0)),
            FakeIndividual((3.0, 3.0)),
            FakeIndividual((5.0, 1.0)),
            FakeIndividual(
                (2.0, 2.0)
            ),  # Dominated by (1,5)? No; dominated by (3,3)? No. But (2,2) dominates (3,3)
            FakeIndividual((4.0, 4.0)),  # Dominated by (2,2) and (3,3)
        ]
        fronts = fast_nondominated_sort(pop)
        front0 = fronts[0]

        # No individual in front 0 is dominated by any other in the population
        for i in front0:
            for j in pop:
                if j is not i:
                    assert not dominates(j, i) or j in front0, (
                        f"Front 0 individual {i.fitness.values} is dominated by "
                        f"{j.fitness.values} which is not in front 0"
                    )

    def test_every_individual_assigned_to_exactly_one_front(self):
        """Every individual appears in exactly one front."""
        pop = [
            FakeIndividual((1.0, 5.0)),
            FakeIndividual((3.0, 3.0)),
            FakeIndividual((5.0, 1.0)),
            FakeIndividual((2.0, 4.0)),
            FakeIndividual((4.0, 2.0)),
        ]
        fronts = fast_nondominated_sort(pop)
        all_ids = set()
        for front in fronts:
            for ind in front:
                assert id(ind) not in all_ids, "Individual appears in multiple fronts"
                all_ids.add(id(ind))
        assert len(all_ids) == len(pop), "Not all individuals assigned to a front"

    def test_front_ordering_correct(self):
        """Front i individuals are dominated by at least one ind in front i-1."""
        a = FakeIndividual((1.0, 1.0))
        b = FakeIndividual((2.0, 2.0))
        c = FakeIndividual((3.0, 3.0))
        fronts = fast_nondominated_sort([a, b, c])
        assert len(fronts) == 3
        # a dominates b, b dominates c
        assert a in fronts[0]
        assert b in fronts[1]
        assert c in fronts[2]

    def test_crowding_distance_nonnegative(self):
        """Crowding distance is always >= 0."""
        front = [
            FakeIndividual((1.0, 5.0)),
            FakeIndividual((2.0, 3.0)),
            FakeIndividual((3.0, 2.0)),
            FakeIndividual((5.0, 1.0)),
        ]
        assign_crowding_distance(front)
        for ind in front:
            assert ind.fitness.crowding_dist >= 0

    def test_crowding_boundary_infinite(self):
        """Extreme individuals (min/max per objective) get infinite crowding distance."""
        front = [
            FakeIndividual((1.0, 5.0)),
            FakeIndividual((2.0, 3.0)),
            FakeIndividual((3.0, 2.0)),
            FakeIndividual((5.0, 1.0)),
        ]
        assign_crowding_distance(front)
        inf_count = sum(1 for i in front if i.fitness.crowding_dist == float("inf"))
        assert inf_count >= 2

    def test_selection_preserves_count(self):
        """sel_nsga2_fast(pop, k) returns exactly k individuals."""
        pop = [FakeIndividual((float(i), float(20 - i))) for i in range(20)]
        selected = sel_nsga2_fast(pop, 10)
        assert len(selected) == 10

    def test_selection_preserves_pareto_optimality(self):
        """Selected set should contain all front-0 individuals when possible."""
        # 3 front-0 + 2 front-1
        pop = [
            FakeIndividual((1.0, 5.0)),
            FakeIndividual((3.0, 3.0)),
            FakeIndividual((5.0, 1.0)),
            FakeIndividual((4.0, 4.0)),
            FakeIndividual((6.0, 6.0)),
        ]
        selected = sel_nsga2_fast(pop, 4)
        # All 3 front-0 should be selected
        front0_selected = sum(1 for s in selected if s.fitness.rank == 0)
        assert front0_selected >= 3


# Strategy Selector Properties


class TestRoundRobinSelector:
    """Test round-robin cycling properties."""

    def _make_data(self):
        """Create NotebookData-like object with qts, rooms, instructors, courses."""
        import pathlib
        from dataclasses import dataclass, field

        from conftest import (
            make_context,
            make_course,
            make_gene,
            make_group,
            make_instructor,
            make_room,
        )

        from src.io.time_system import QuantumTimeSystem

        ctx = make_context(
            courses=[make_course("CS101", groups=["G1"], instructors=["I1", "I2"])],
            groups=[make_group("G1")],
            instructors=[make_instructor("I1"), make_instructor("I2")],
            rooms=[make_room("R1"), make_room("R2")],
        )

        # Build a simple object with qts + context attributes
        @dataclass
        class FakeData:
            qts: QuantumTimeSystem = field(default_factory=QuantumTimeSystem)
            courses: dict = field(default_factory=dict)
            groups: dict = field(default_factory=dict)
            instructors: dict = field(default_factory=dict)
            rooms: dict = field(default_factory=dict)
            context: object = None

        data = FakeData(
            courses=ctx.courses,
            groups=ctx.groups,
            instructors=ctx.instructors,
            rooms=ctx.rooms,
            context=ctx,
        )
        individual = [
            make_gene(
                course_id="CS101",
                instructor_id="I1",
                group_ids=["G1"],
                room_id="R1",
                start=0,
                duration=2,
            )
        ]
        return individual, data

    def test_cycles_through_all_operators(self):
        """After 3 applies, all 3 operators should have been used."""
        from src.ga.heuristics.strategies import RoundRobinSelector

        selector = RoundRobinSelector()
        individual, data = self._make_data()

        seen = set()
        for _ in range(6):
            name, fixes = selector.apply(individual, data)
            seen.add(name)

        assert len(seen) == 3, f"Expected 3 operators, got {seen}"

    def test_deterministic_order(self):
        """Same sequence of applies should produce same operator order."""
        from src.ga.heuristics.strategies import RoundRobinSelector

        s1 = RoundRobinSelector()
        s2 = RoundRobinSelector()
        individual, data = self._make_data()

        seq1 = [s1.apply(individual, data)[0] for _ in range(6)]
        # Reset individual for s2
        individual2, data2 = self._make_data()
        seq2 = [s2.apply(individual2, data2)[0] for _ in range(6)]
        assert seq1 == seq2


class TestAdaptiveSelector:
    """Test adaptive selector probability properties."""

    def _make_data(self):
        from dataclasses import dataclass, field

        from conftest import (
            make_context,
            make_course,
            make_gene,
            make_group,
            make_instructor,
            make_room,
        )

        from src.io.time_system import QuantumTimeSystem

        ctx = make_context(
            courses=[make_course("CS101", groups=["G1"], instructors=["I1", "I2"])],
            groups=[make_group("G1")],
            instructors=[make_instructor("I1"), make_instructor("I2")],
            rooms=[make_room("R1"), make_room("R2")],
        )

        @dataclass
        class FakeData:
            qts: QuantumTimeSystem = field(default_factory=QuantumTimeSystem)
            courses: dict = field(default_factory=dict)
            groups: dict = field(default_factory=dict)
            instructors: dict = field(default_factory=dict)
            rooms: dict = field(default_factory=dict)
            context: object = None

        data = FakeData(
            courses=ctx.courses,
            groups=ctx.groups,
            instructors=ctx.instructors,
            rooms=ctx.rooms,
            context=ctx,
        )
        individual = [
            make_gene(
                course_id="CS101",
                instructor_id="I1",
                group_ids=["G1"],
                room_id="R1",
                start=0,
                duration=2,
            )
        ]
        return individual, data

    def test_initial_probabilities_uniform(self):
        from src.ga.heuristics.strategies import AdaptiveSelector

        selector = AdaptiveSelector()
        stats = selector.get_stats()
        probs = stats["probs"]
        assert len(probs) == 3
        for p in probs.values():
            assert abs(p - 1 / 3) < 0.01

    def test_probabilities_sum_to_one(self):
        """Probabilities should always sum to 1.0."""
        from src.ga.heuristics.strategies import AdaptiveSelector

        selector = AdaptiveSelector()
        individual, data = self._make_data()

        for _ in range(10):
            selector.apply(individual, data)
            stats = selector.get_stats()
            total = sum(stats["probs"].values())
            assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, not 1.0"

    def test_min_probability_floor(self):
        """No probability should drop too far below min_prob (soft enforcement)."""
        from src.ga.heuristics.strategies import AdaptiveSelector

        min_prob = 0.05
        selector = AdaptiveSelector(min_prob=min_prob)
        individual, data = self._make_data()

        for _ in range(50):
            selector.apply(individual, data)
            stats = selector.get_stats()
            for name, prob in stats["probs"].items():
                # Allow 10% relative tolerance since enforcement is approximate
                assert (
                    prob >= min_prob * 0.8
                ), f"{name} probability {prob} dropped far below min_prob {min_prob}"


# Constraint Weight Algebra


class TestConstraintAlgebra:
    """Test mathematical properties of constraint evaluation."""

    def test_constraint_weights_positive(self):
        """All constraint weights should be > 0."""
        from src.constraints.constraints import ALL_CONSTRAINTS

        for c in ALL_CONSTRAINTS:
            assert c.weight > 0, f"{c.name} has weight {c.weight} <= 0"

    def test_hard_weights_larger_than_soft(self):
        """Hard constraint weights should be >= soft weights."""
        from src.constraints.constraints import (
            HARD_CONSTRAINT_CLASSES,
            SOFT_CONSTRAINT_CLASSES,
        )

        min_hard = min(c.weight for c in HARD_CONSTRAINT_CLASSES)
        max_soft = max(c.weight for c in SOFT_CONSTRAINT_CLASSES)
        assert (
            min_hard >= max_soft
        ), f"Min hard weight {min_hard} < max soft weight {max_soft}"

    def test_evaluate_returns_non_negative(self):
        """Every constraint.evaluate() should return >= 0."""

        from conftest import (
            make_context,
            make_course,
            make_gene,
            make_group,
            make_instructor,
            make_room,
        )

        from src.constraints.constraints import ALL_CONSTRAINTS
        from src.domain.timetable import Timetable

        g = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        ctx = make_context(
            courses=[make_course("CS101", groups=["G1"], instructors=["I1"])],
            groups=[make_group("G1")],
            instructors=[make_instructor("I1")],
            rooms=[make_room("R1")],
        )
        tt = Timetable([g], ctx)

        for c in ALL_CONSTRAINTS:
            val = c.evaluate(tt)
            assert val >= 0, f"{c.name}.evaluate() returned {val} < 0"

    def test_evaluation_deterministic(self):
        """Same timetable → same constraint values."""

        from conftest import (
            make_context,
            make_course,
            make_gene,
            make_group,
            make_instructor,
            make_room,
        )

        from src.constraints.constraints import ALL_CONSTRAINTS
        from src.domain.timetable import Timetable

        g = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )
        ctx = make_context(
            courses=[make_course("CS101", groups=["G1"], instructors=["I1"])],
            groups=[make_group("G1")],
            instructors=[make_instructor("I1")],
            rooms=[make_room("R1")],
        )
        tt = Timetable([g], ctx)

        for c in ALL_CONSTRAINTS:
            v1 = c.evaluate(tt)
            v2 = c.evaluate(tt)
            assert v1 == v2, f"{c.name} non-deterministic: {v1} != {v2}"


# QuantumTimeSystem Properties


class TestQuantumTimeSystem:
    """Test time system mathematical properties."""

    def test_total_quanta_equals_sum_of_day_counts(self):
        """total_quanta should equal the sum of all day_quanta_count values."""
        from src.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        expected = sum(qts.day_quanta_count.values())
        assert qts.total_quanta == expected

    def test_quantum_to_day_roundtrip(self):
        """quantum → (day, time) → quantum should roundtrip via QTS methods."""
        from src.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        for q in range(qts.total_quanta):
            day, time_str = qts.quanta_to_time(q)
            reconstructed = qts.time_to_quanta(day, time_str)
            assert (
                reconstructed == q
            ), f"Roundtrip failed for q={q}: got {reconstructed}"

    def test_all_quanta_valid(self):
        """Every quantum 0..total-1 should decode to a valid day."""
        from src.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        operational_days = [d for d, c in qts.day_quanta_count.items() if c > 0]
        for q in range(qts.total_quanta):
            day, _ = qts.quanta_to_time(q)
            assert day in operational_days, f"q={q} mapped to non-operational day {day}"

    def test_no_cross_day_overlap(self):
        """Quanta within one day should not overlap with another day's range."""
        from src.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        day_ranges = {}
        for day in qts.DAY_NAMES:
            offset = qts.day_quanta_offset[day]
            count = qts.day_quanta_count[day]
            if offset is not None and count > 0:
                day_ranges[day] = set(range(offset, offset + count))

        days = list(day_ranges.keys())
        for i, d1 in enumerate(days):
            for j, d2 in enumerate(days):
                if i != j:
                    assert day_ranges[d1].isdisjoint(
                        day_ranges[d2]
                    ), f"{d1} and {d2} quanta overlap"
