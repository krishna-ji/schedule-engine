"""Phase 7: System Integration Tests.

End-to-end pipeline tests verifying component interplay:
    1. CONSTRUCT→EVALUATE:  Constructed schedules can be evaluated
    2. CONSTRUCT→REPAIR→EVALUATE:  Repair reduces violations
    3. CROSSOVER→REPAIR:  Crossover + repair chain works
    4. MUTATION→EVALUATE:  Mutation preserves evaluability
    5. FULL PIPELINE:  Full GA pipeline step produces valid output
"""

from __future__ import annotations

from conftest import (
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
)

from src.config import Config, init_config
from src.constraints.constraints import (
    InstructorExclusivity,
    RoomExclusivity,
    StudentGroupExclusivity,
)
from src.constraints.evaluator import Evaluator
from src.domain.timetable import Timetable
from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import mutate_gene, mutate_individual
from src.ga.repair.basic import repair_individual_unified


def _init():
    """Initialize config for tests that need repair pipeline."""
    init_config(Config(repair={"enabled": True, "heuristics": {}}))


def _make_medium_scenario():
    """A realistic-ish scenario with multiple courses, groups, instructors, rooms.

    Creates 4 courses x 2 groups x 3 instructors x 4 rooms.
    """
    courses = [
        make_course(
            "CS101", course_type="theory", groups=["G1"], instructors=["I1", "I2"]
        ),
        make_course(
            "CS102", course_type="theory", groups=["G2"], instructors=["I2", "I3"]
        ),
        make_course(
            "CS103",
            course_type="practical",
            groups=["G1"],
            instructors=["I1"],
            room_feat="practical",
        ),
        make_course(
            "CS104",
            course_type="theory",
            groups=["G1", "G2"],
            instructors=["I1", "I2", "I3"],
        ),
    ]
    groups = [make_group("G1"), make_group("G2")]
    instructors = [make_instructor("I1"), make_instructor("I2"), make_instructor("I3")]
    rooms = [
        make_room("R1", capacity=50),
        make_room("R2", capacity=40),
        make_room("R3", capacity=60, features="lab"),
        make_room("R4", capacity=80),
    ]
    return make_context(
        courses=courses, groups=groups, instructors=instructors, rooms=rooms
    )


def _make_individual_for_ctx(ctx):
    """Create a simple individual with one gene per course."""
    genes = []
    start = 0
    for key, course in ctx.courses.items():
        cid, ctype = key
        instructor_id = (
            course.qualified_instructor_ids[0]
            if course.qualified_instructor_ids
            else "I1"
        )
        group_id = course.enrolled_group_ids[0] if course.enrolled_group_ids else "G1"
        genes.append(
            make_gene(
                course_id=cid,
                course_type=ctype,
                instructor_id=instructor_id,
                group_ids=[group_id],
                room_id="R1",
                start=start,
                duration=2,
            )
        )
        start += 3  # Space them out
    return genes


# Construct → Evaluate


class TestConstructEvaluate:
    """Test that constructed schedules can be evaluated."""

    def test_evaluator_returns_two_floats(self):
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)
        evaluator = Evaluator()
        result = evaluator.fitness(individual, ctx)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int | float)
        assert isinstance(result[1], int | float)

    def test_evaluator_hard_soft_split(self):
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)
        evaluator = Evaluator()
        hard, soft = evaluator.fitness(individual, ctx)
        assert hard >= 0
        assert soft >= 0

    def test_breakdown_has_all_constraints(self):
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)
        evaluator = Evaluator()
        breakdown = evaluator.breakdown(individual, ctx)
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

    def test_evaluate_all_returns_four_elements(self):
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)
        evaluator = Evaluator()
        tt = Timetable(individual, ctx)
        result = evaluator.evaluate_all(tt)
        assert len(result) == 4
        hard_total, soft_total, hard_bd, soft_bd = result
        assert isinstance(hard_total, int | float)
        assert isinstance(soft_total, int | float)
        assert isinstance(hard_bd, dict)
        assert isinstance(soft_bd, dict)


# Construct → Repair → Evaluate


class TestConstructRepairEvaluate:
    """Full chain: create violating schedule → repair → measure improvement."""

    def setup_method(self):
        _init()

    def test_repair_reduces_hard_violations(self):
        """INTENT: repair pipeline should reduce hard violations."""
        ctx = _make_medium_scenario()
        # Create schedule with deliberate conflicts
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
                course_id="CS104",
                course_type="theory",
                instructor_id="I1",
                group_ids=["G1"],
                room_id="R1",
                start=0,
                duration=2,
            ),  # Triple conflict!
        ]
        evaluator = Evaluator()
        pre_hard, _ = evaluator.fitness(individual, ctx)

        repair_individual_unified(individual, ctx, selective=False)

        post_hard, _ = evaluator.fitness(individual, ctx)
        assert post_hard <= pre_hard, f"Repair worsened: {pre_hard} → {post_hard}"

    def test_repair_preserves_gene_count(self):
        """Repair should never add or remove genes."""
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)
        before_len = len(individual)

        repair_individual_unified(individual, ctx, selective=False)

        assert len(individual) == before_len

    def test_repair_produces_evaluable_schedule(self):
        """After repair, schedule should still be evaluable."""
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)

        repair_individual_unified(individual, ctx, selective=False)

        evaluator = Evaluator()
        hard, soft = evaluator.fitness(individual, ctx)
        assert isinstance(hard, int | float)
        assert isinstance(soft, int | float)


# Crossover → Repair → Evaluate


class TestCrossoverRepairEvaluate:
    """Crossover offspring can be repaired and evaluated."""

    def setup_method(self):
        _init()

    def test_crossover_then_repair(self):
        ctx = _make_medium_scenario()
        ind1 = _make_individual_for_ctx(ctx)
        ind2 = _make_individual_for_ctx(ctx)
        # Mutate ind2 to differ
        for g in ind2:
            g.start_quanta = (g.start_quanta + 7) % 35

        child1, child2 = crossover_course_group_aware(
            ind1, ind2, cx_prob=0.8, validate=False
        )

        evaluator = Evaluator()
        pre_hard, _ = evaluator.fitness(child1, ctx)

        repair_individual_unified(child1, ctx, selective=False)

        post_hard, _ = evaluator.fitness(child1, ctx)
        assert post_hard <= pre_hard


# Mutation → Evaluate


class TestMutationEvaluate:
    """Mutation produces evaluable schedules."""

    def test_mutated_individual_evaluable(self):
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)

        mutated = mutate_individual(individual, ctx, guided=False)

        evaluator = Evaluator()
        hard, soft = evaluator.fitness(mutated[0], ctx)
        assert isinstance(hard, int | float)

    def test_mutated_gene_valid(self):
        ctx = _make_medium_scenario()
        g = make_gene(
            course_id="CS101",
            instructor_id="I1",
            group_ids=["G1"],
            room_id="R1",
            start=0,
            duration=2,
        )

        mutated = mutate_gene(g, ctx)

        assert isinstance(mutated, type(g))
        assert mutated.course_id == "CS101"
        assert mutated.num_quanta == 2


# Evaluator Properties


class TestEvaluatorProperties:
    """Test evaluator algebraic properties."""

    def test_deterministic(self):
        """Same schedule → same fitness."""
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)
        evaluator = Evaluator()

        r1 = evaluator.fitness(individual, ctx)
        r2 = evaluator.fitness(individual, ctx)
        assert r1 == r2

    def test_non_negative(self):
        """Fitness values should always be non-negative (penalties)."""
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)
        evaluator = Evaluator()
        hard, soft = evaluator.fitness(individual, ctx)
        assert hard >= 0
        assert soft >= 0

    def test_perfect_schedule_zero_hard(self):
        """A schedule with NO exclusivity/qualification violations should have those at 0."""
        # Single gene, no conflicts possible
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
        # No exclusivity violations (only 1 gene → no overlaps)
        assert StudentGroupExclusivity().evaluate(tt) == 0
        assert InstructorExclusivity().evaluate(tt) == 0
        assert RoomExclusivity().evaluate(tt) == 0

    def test_fitness_and_fitness_from_timetable_agree(self):
        """Both fitness paths should produce the same result."""
        ctx = _make_medium_scenario()
        individual = _make_individual_for_ctx(ctx)
        evaluator = Evaluator()

        f1 = evaluator.fitness(individual, ctx)
        tt = Timetable(individual, ctx)
        f2 = evaluator.fitness_from_timetable(tt)
        assert f1 == f2


# RepairEngine (OOP)


class TestRepairEngine:
    """Test OOP RepairEngine with policies."""

    def _make_engine(self, policy="round_robin"):
        from src.ga.repair.engine import RepairEngine

        ctx = _make_medium_scenario()
        evaluator = Evaluator()

        def fitness_fn(individual):
            return evaluator.fitness(individual, ctx)

        return RepairEngine(ctx, fitness_fn, policy=policy, max_steps=3), ctx

    def test_round_robin_creates(self):
        engine, _ = self._make_engine("round_robin")
        assert engine is not None

    def test_epsilon_greedy_creates(self):
        engine, _ = self._make_engine("epsilon_greedy")
        assert engine is not None

    def test_action_space(self):
        engine, _ = self._make_engine()
        actions = engine.get_action_space()
        assert "move_time" in actions
        assert "swap_room" in actions
        assert "reassign_instructor" in actions

    def test_repair_returns_stats(self):
        engine, ctx = self._make_engine()
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
                course_id="CS104",
                course_type="theory",
                instructor_id="I1",
                group_ids=["G1"],
                room_id="R1",
                start=0,
                duration=2,
            ),
        ]
        stats = engine.repair_individual(individual)
        assert hasattr(stats, "steps")
        assert hasattr(stats, "applied_steps")
        assert hasattr(stats, "total_delta_hard")

    def test_repair_does_not_worsen(self):
        """ALGORITHM: RepairEngine should never increase hard violations."""
        engine, ctx = self._make_engine()
        individual = _make_individual_for_ctx(ctx)
        evaluator = Evaluator()
        pre_hard, _ = evaluator.fitness(individual, ctx)

        engine.repair_individual(individual)

        post_hard, _ = evaluator.fitness(individual, ctx)
        assert post_hard <= pre_hard
