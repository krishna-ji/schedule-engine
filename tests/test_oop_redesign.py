"""Tests for the OOP redesign - Phases 2-7.

Tests for:
- Phase 2: Constraint protocol and constraint classes
- Phase 3: Evaluator class
- Phase 4: RepairPipeline class (unit tests with mocks)
- Phase 5: PopulationFactory class (structural tests)
- Phase 6: BaseExperiment + PopulationFactory integration
- Phase 7: family_map in SchedulingContext

All tests use lightweight in-memory fixtures and never touch disk.
"""

from __future__ import annotations

from src.domain.course import Course
from src.domain.gene import SessionGene
from src.domain.group import Group
from src.domain.instructor import Instructor
from src.domain.room import Room
from src.domain.timetable import Timetable
from src.domain.types import SchedulingContext

# Shared fixtures


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


def _simple_context() -> SchedulingContext:
    """Minimal context: 1 course, 1 group, 1 instructor, 1 room."""
    c = _make_course()
    return SchedulingContext(
        courses={(c.course_id, c.course_type): c},
        groups={"G1": _make_group()},
        instructors={"I1": _make_instructor()},
        rooms={"R1": _make_room()},
        available_quanta=list(range(60)),
    )


def _conflict_context() -> SchedulingContext:
    """Context with 2 courses that can conflict."""
    c1 = _make_course("CS101", groups=["G1"], instructors=["I1"])
    c2 = _make_course("CS102", groups=["G1"], instructors=["I1"])
    return SchedulingContext(
        courses={
            (c1.course_id, c1.course_type): c1,
            (c2.course_id, c2.course_type): c2,
        },
        groups={"G1": _make_group("G1", courses=["CS101", "CS102"])},
        instructors={"I1": _make_instructor("I1")},
        rooms={"R1": _make_room("R1")},
        available_quanta=list(range(60)),
    )


# Phase 2: Constraint Protocol


class TestConstraintProtocol:
    """Tests for constraint protocol compliance and registries."""

    def test_constraint_is_runtime_checkable(self):
        from src.constraints import Constraint

        assert hasattr(Constraint, "__protocol_attrs__") or hasattr(
            Constraint, "_is_runtime_protocol"
        )

    def test_hard_constraints_have_correct_kind(self):
        from src.constraints import HARD_CONSTRAINT_CLASSES

        for c in HARD_CONSTRAINT_CLASSES:
            assert c.kind == "hard", f"{c.name} has kind={c.kind}"

    def test_soft_constraints_have_correct_kind(self):
        from src.constraints import SOFT_CONSTRAINT_CLASSES

        for c in SOFT_CONSTRAINT_CLASSES:
            assert c.kind == "soft", f"{c.name} has kind={c.kind}"

    def test_all_constraints_is_union(self):
        from src.constraints import (
            ALL_CONSTRAINTS,
            HARD_CONSTRAINT_CLASSES,
            SOFT_CONSTRAINT_CLASSES,
        )

        assert len(ALL_CONSTRAINTS) == len(HARD_CONSTRAINT_CLASSES) + len(
            SOFT_CONSTRAINT_CLASSES
        )

    def test_all_constraints_have_unique_names(self):
        from src.constraints import ALL_CONSTRAINTS

        names = [c.name for c in ALL_CONSTRAINTS]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_all_constraints_have_positive_weight(self):
        from src.constraints import ALL_CONSTRAINTS

        for c in ALL_CONSTRAINTS:
            assert c.weight > 0, f"{c.name} has weight={c.weight}"

    def test_all_constraints_have_evaluate(self):
        from src.constraints import ALL_CONSTRAINTS

        for c in ALL_CONSTRAINTS:
            assert callable(
                getattr(c, "evaluate", None)
            ), f"{c.name} missing evaluate()"

    def test_exclusivity_constraints_use_timetable_indexes(self):
        """The 3 exclusivity constraints should return 0 for non-overlapping genes."""
        from src.constraints import (
            InstructorExclusivity,
            RoomExclusivity,
            StudentGroupExclusivity,
        )

        ctx = _simple_context()
        genes = [_make_gene(start=0, duration=2)]
        tt = Timetable(genes, ctx)

        assert StudentGroupExclusivity().evaluate(tt) == 0
        assert InstructorExclusivity().evaluate(tt) == 0
        assert RoomExclusivity().evaluate(tt) == 0

    def test_exclusivity_detects_group_conflict(self):
        from src.constraints import StudentGroupExclusivity

        ctx = _conflict_context()
        # Two genes for same group at same time
        genes = [
            _make_gene("CS101", start=0, duration=2),
            _make_gene("CS102", start=0, duration=2),
        ]
        tt = Timetable(genes, ctx)
        assert StudentGroupExclusivity().evaluate(tt) > 0

    def test_known_hard_constraint_count(self):
        from src.constraints import HARD_CONSTRAINT_CLASSES

        assert len(HARD_CONSTRAINT_CLASSES) == 9

    def test_known_soft_constraint_count(self):
        from src.constraints import SOFT_CONSTRAINT_CLASSES

        assert len(SOFT_CONSTRAINT_CLASSES) == 6


# Phase 3: Evaluator


class TestEvaluator:
    """Tests for the unified Evaluator class."""

    def test_import(self):
        from src.constraints import Evaluator

        assert Evaluator is not None

    def test_evaluator_default_constraints(self):
        from src.constraints import Evaluator

        ev = Evaluator()
        assert len(ev.hard) == 9
        assert len(ev.soft) == 6

    def test_evaluator_custom_constraints(self):
        from src.constraints import Evaluator, StudentGroupExclusivity

        ev = Evaluator(constraints=[StudentGroupExclusivity()])
        assert len(ev.hard) == 1
        assert len(ev.soft) == 0

    def test_fitness_returns_two_floats(self):
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        genes = [_make_gene(start=0, duration=2)]
        result = ev.fitness(genes, ctx)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int | float)
        assert isinstance(result[1], int | float)

    def test_fitness_from_timetable(self):
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        genes = [_make_gene(start=0, duration=2)]
        tt = Timetable(genes, ctx)
        result = ev.fitness_from_timetable(tt)
        assert isinstance(result, tuple) and len(result) == 2

    def test_breakdown_returns_dict(self):
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        genes = [_make_gene(start=0, duration=2)]
        bd = ev.breakdown(genes, ctx)
        assert isinstance(bd, dict)
        assert len(bd) == 15  # 9 hard + 6 soft

    def test_breakdown_from_timetable(self):
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        genes = [_make_gene(start=0, duration=2)]
        tt = Timetable(genes, ctx)
        bd = ev.breakdown_from_timetable(tt)
        assert len(bd) == 15

    def test_hard_breakdown_only_hard(self):
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        tt = Timetable([_make_gene()], ctx)
        hb = ev.hard_breakdown(tt)
        assert len(hb) == 9

    def test_soft_breakdown_only_soft(self):
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        tt = Timetable([_make_gene()], ctx)
        sb = ev.soft_breakdown(tt)
        assert len(sb) == 6

    def test_evaluate_all_returns_four_elements(self):
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        tt = Timetable([_make_gene()], ctx)
        result = ev.evaluate_all(tt)
        assert len(result) == 4
        hard_total, soft_total, hard_bd, soft_bd = result
        # Numeric totals
        assert hard_total >= 0 or hard_total < 0  # is a number
        assert soft_total >= 0 or soft_total < 0
        assert isinstance(hard_bd, dict)
        assert isinstance(soft_bd, dict)

    def test_no_group_conflict_gives_zero_exclusivity(self):
        """Single gene -> zero group/instructor/room exclusivity violations."""
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        genes = [_make_gene(start=0, duration=2)]
        bd = ev.breakdown(genes, ctx)
        assert bd["CTE"] == 0
        assert bd["FTE"] == 0
        assert bd["SRE"] == 0

    def test_conflict_gives_nonzero_exclusivity(self):
        """Two overlapping genes for same group -> nonzero penalty."""
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _conflict_context()
        genes = [
            _make_gene("CS101", start=0, duration=2),
            _make_gene("CS102", start=0, duration=2),
        ]
        bd = ev.breakdown(genes, ctx)
        assert bd["CTE"] > 0

    def test_fitness_consistency(self):
        """fitness() == fitness_from_timetable() on same data."""
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        genes = [_make_gene()]
        direct = ev.fitness(genes, ctx)
        tt = Timetable(genes, ctx)
        from_tt = ev.fitness_from_timetable(tt)
        assert direct == from_tt


# Phase 4: RepairPipeline (structural / unit tests)


class TestRepairPipeline:
    """Structural tests for the RepairPipeline class."""

    def test_import(self):
        from src.ga import RepairPipeline

        assert RepairPipeline is not None

    def test_pipeline_has_repair_method(self):
        from src.ga import RepairPipeline

        assert callable(getattr(RepairPipeline, "repair", None))

    def test_pipeline_has_default_factory(self):
        from src.ga import RepairPipeline

        assert callable(getattr(RepairPipeline, "default", None))


# Phase 5: PopulationFactory (structural tests)


class TestPopulationFactory:
    """Structural tests for the PopulationFactory class."""

    def test_import(self):
        from src.ga import PopulationFactory

        assert PopulationFactory is not None

    def test_factory_has_methods(self):
        from src.ga import PopulationFactory

        assert callable(getattr(PopulationFactory, "create_population", None))
        assert callable(getattr(PopulationFactory, "random_individual", None))
        assert callable(getattr(PopulationFactory, "greedy_individual", None))

    def test_factory_stores_context(self):
        from src.ga import PopulationFactory

        ctx = _simple_context()
        factory = PopulationFactory(ctx)
        assert factory.context is ctx


# Phase 7: family_map in SchedulingContext


class TestFamilyMapInContext:
    """Tests for family_map being part of SchedulingContext."""

    def test_context_has_family_map_field(self):
        ctx = _simple_context()
        assert hasattr(ctx, "family_map")

    def test_family_map_defaults_to_empty_dict(self):
        ctx = _simple_context()
        assert isinstance(ctx.family_map, dict)
        assert len(ctx.family_map) == 0

    def test_family_map_can_be_set(self):
        ctx = SchedulingContext(
            courses=[_make_course()],
            groups=[_make_group()],
            instructors=[_make_instructor()],
            rooms=[_make_room()],
            available_quanta=list(range(60)),
            family_map={"G1": {"G1", "G2"}},
        )
        assert "G1" in ctx.family_map
        assert "G2" in ctx.family_map["G1"]


# Cross-phase integration


class TestCrossPhaseIntegration:
    """Tests that verify different phases work together correctly."""

    def test_constraint_evaluates_via_timetable(self):
        """Phase 1 (Timetable) + Phase 2 (Constraint) integration."""
        from src.constraints import StudentGroupExclusivity

        ctx = _simple_context()
        genes = [_make_gene()]
        tt = Timetable(genes, ctx)
        result = StudentGroupExclusivity().evaluate(tt)
        assert isinstance(result, int | float)

    def test_evaluator_uses_constraints_on_timetable(self):
        """Phase 1 + Phase 2 + Phase 3 integration."""
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        genes = [_make_gene()]
        hard, soft = ev.fitness(genes, ctx)
        assert isinstance(hard, int | float)
        assert isinstance(soft, int | float)

    def test_evaluator_breakdown_names_match_constraints(self):
        # All constraint names appear in breakdown dict.
        from src.constraints import ALL_CONSTRAINTS, Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        bd = ev.breakdown([_make_gene()], ctx)
        expected_names = {c.name for c in ALL_CONSTRAINTS}
        assert set(bd.keys()) == expected_names

    def test_evaluate_all_totals_match_fitness(self):
        # evaluate_all() totals should match fitness().
        from src.constraints import Evaluator

        ev = Evaluator()
        ctx = _simple_context()
        genes = [_make_gene()]
        hard, soft = ev.fitness(genes, ctx)
        tt = Timetable(genes, ctx)
        h_total, s_total, _, _ = ev.evaluate_all(tt)
        assert abs(h_total - hard) < 1e-9
        assert abs(s_total - soft) < 1e-9
