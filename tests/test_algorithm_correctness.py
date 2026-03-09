"""Phase 8: Algorithm Correctness Tests.

Tests mathematical/algorithmic properties that must hold:
    1. CONSTRAINT ALGEBRA:  Weights, commutativity, additivity
    2. QUANTUM TIME SYSTEM:  Quanta boundaries, round-trip consistency
"""

from __future__ import annotations

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
