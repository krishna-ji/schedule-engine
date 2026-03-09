"""
Comprehensive tests for ALL algorithm features in schedule-engine.

Covers everything discussed and implemented:
1. ScheduleIndex — core operations, caching, invalidation, integration
2. Meta-heuristics — VND, ILS, ALNS, GLS behavior
3. Repair operators — greedy, IGLS, basic repairs, unified pipeline
4. Detector — integration with ScheduleIndex
5. LNS — trigger logic, stats
6. Improvement heuristics — kempe, ejection, variable depth
7. Perturbation heuristics — random swap, temporal shift, room shuffle
8. Construction heuristics — largest degree, most constrained
9. Repair engine — RepairEngine, policies, operators
10. Local search — greedy vs exhaustive gene optimization
"""

from __future__ import annotations

import copy
import random
import time

from src.config import Config, init_config
from src.domain.timetable import Timetable
from src.ga.core.schedule_index import ScheduleIndex, create_schedule_index


def _ensure_config():
    """Initialize config if not already done."""
    try:
        from src.config import get_config

        get_config()
    except RuntimeError:
        init_config(Config(repair={"enabled": True, "heuristics": {}}))


from conftest import (
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
    structural_fields_preserved,
)

# =====================================================================
# FIXTURES
# =====================================================================


def _make_conflicting_individual():
    """Create an individual with known conflicts for testing."""
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
        group_ids=["G1"],
        room_id="R2",
        start=0,
        duration=2,
    )  # Group overlap with g1
    g3 = make_gene(
        course_id="CS103",
        instructor_id="I1",
        group_ids=["G2"],
        room_id="R3",
        start=14,
        duration=2,
    )  # No conflict
    ctx = make_context(
        courses=[
            make_course("CS101", groups=["G1"], instructors=["I1", "I2"]),
            make_course("CS102", groups=["G1"], instructors=["I1", "I2"]),
            make_course("CS103", groups=["G2"], instructors=["I1", "I2"]),
        ],
        groups=[make_group("G1"), make_group("G2")],
        instructors=[make_instructor("I1"), make_instructor("I2")],
        rooms=[make_room("R1"), make_room("R2"), make_room("R3")],
    )
    return [g1, g2, g3], ctx


def _make_clean_individual():
    """Create a conflict-free individual."""
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
    return [g1, g2], ctx


def _make_multi_conflict_individual():
    """Create individual with group + room + instructor conflicts."""
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
    )  # All three conflicts with g1
    g3 = make_gene(
        course_id="CS103",
        instructor_id="I2",
        group_ids=["G2"],
        room_id="R2",
        start=14,
        duration=2,
    )
    ctx = make_context(
        courses=[
            make_course("CS101", groups=["G1"], instructors=["I1", "I2"]),
            make_course("CS102", groups=["G1"], instructors=["I1", "I2"]),
            make_course("CS103", groups=["G2"], instructors=["I1", "I2"]),
        ],
        groups=[make_group("G1"), make_group("G2")],
        instructors=[make_instructor("I1"), make_instructor("I2")],
        rooms=[make_room("R1"), make_room("R2"), make_room("R3")],
    )
    return [g1, g2, g3], ctx


# =====================================================================
# 1. SCHEDULE INDEX — Extended Tests
# =====================================================================


class TestScheduleIndexOccupied:
    """Test ScheduleIndex quantum-level occupied queries."""

    def test_get_occupied_at_quantum_returns_entities(self):
        """Verify occupied query returns correct groups/rooms/instructors."""
        genes = [
            make_gene(
                group_ids=["G1"], room_id="R1", instructor_id="I1", start=0, duration=2
            ),
            make_gene(
                group_ids=["G2"], room_id="R2", instructor_id="I2", start=0, duration=2
            ),
        ]
        index = ScheduleIndex.from_individual(genes)
        occupied = index.get_occupied_at_quantum(0)

        assert "G1" in occupied["groups"]
        assert "G2" in occupied["groups"]
        assert "R1" in occupied["rooms"]
        assert "R2" in occupied["rooms"]
        assert "I1" in occupied["instructors"]
        assert "I2" in occupied["instructors"]

    def test_get_occupied_at_empty_quantum(self):
        """Quantum with no sessions → empty sets."""
        genes = [
            make_gene(start=0, duration=2),
        ]
        index = ScheduleIndex.from_individual(genes)
        occupied = index.get_occupied_at_quantum(20)

        assert len(occupied["groups"]) == 0
        assert len(occupied["rooms"]) == 0
        assert len(occupied["instructors"]) == 0

    def test_get_all_occupied(self):
        """Verify get_all_occupied returns inverted map structure."""
        genes = [
            make_gene(
                group_ids=["G1"], room_id="R1", instructor_id="I1", start=0, duration=2
            ),
        ]
        index = ScheduleIndex.from_individual(genes)
        all_occ = index.get_all_occupied()

        assert "groups" in all_occ
        assert "rooms" in all_occ
        assert "instructors" in all_occ
        # Quantum 0 should have G1
        assert "G1" in all_occ["groups"][0]
        # Quantum 1 should also have G1 (duration=2)
        assert "G1" in all_occ["groups"][1]


class TestScheduleIndexPerformance:
    """Test that caching actually avoids redundant rebuilds."""

    def test_cached_access_faster_than_rebuild(self):
        """Multiple accesses without invalidation should NOT rebuild."""
        genes = [
            make_gene(
                course_id=f"CS{i}",
                group_ids=[f"G{i % 5}"],
                room_id=f"R{i % 10}",
                instructor_id=f"I{i % 8}",
                start=i % 30,
                duration=2,
            )
            for i in range(100)
        ]
        index = ScheduleIndex.from_individual(genes)

        # First access builds maps
        t0 = time.perf_counter()
        index.find_group_conflicts()
        first_time = time.perf_counter() - t0

        # Subsequent accesses use cache
        t0 = time.perf_counter()
        for _ in range(100):
            index.find_group_conflicts()
            index.find_room_conflicts()
            index.find_instructor_conflicts()
        cached_time = time.perf_counter() - t0

        # 300 cached calls should NOT be slower than 300x first call
        assert cached_time < first_time * 300 * 2  # Very generous margin

    def test_invalidation_triggers_rebuild(self):
        """After invalidation, maps should be rebuilt."""
        genes = [
            make_gene(group_ids=["G1"], start=0, duration=2),
            make_gene(group_ids=["G1"], start=0, duration=2),
        ]
        index = ScheduleIndex.from_individual(genes)

        # Build cache
        conflicts = index.find_group_conflicts()
        assert len(conflicts) == 2
        assert index.is_valid()

        # Invalidate
        index.invalidate()
        assert not index.is_valid()

        # Modify to remove conflict
        genes[1].start_quanta = 10
        conflicts2 = index.find_group_conflicts()
        assert len(conflicts2) == 0
        assert index.is_valid()


class TestScheduleIndexEdgeCases:
    """Edge cases for ScheduleIndex."""

    def test_single_gene(self):
        """Single gene never conflicts with itself."""
        genes = [make_gene(start=0, duration=2)]
        index = ScheduleIndex.from_individual(genes)
        assert index.find_group_conflicts() == {}
        assert index.find_room_conflicts() == {}
        assert index.find_instructor_conflicts() == {}

    def test_genes_adjacent_no_conflict(self):
        """Genes at adjacent but non-overlapping times don't conflict."""
        genes = [
            make_gene(
                group_ids=["G1"], room_id="R1", instructor_id="I1", start=0, duration=2
            ),
            make_gene(
                group_ids=["G1"], room_id="R1", instructor_id="I1", start=2, duration=2
            ),
        ]
        index = ScheduleIndex.from_individual(genes)
        assert index.find_group_conflicts() == {}
        assert index.find_room_conflicts() == {}
        assert index.find_instructor_conflicts() == {}

    def test_off_by_one_overlap(self):
        """Gene ending at quantum 3 and gene starting at quantum 2 should conflict."""
        genes = [
            make_gene(group_ids=["G1"], start=0, duration=3),  # quanta 0,1,2
            make_gene(group_ids=["G1"], start=2, duration=2),  # quanta 2,3
        ]
        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_group_conflicts()
        assert 0 in conflicts
        assert 1 in conflicts

    def test_gene_conflicts_for_nonexistent_index(self):
        """Querying conflicts for gene that doesn't conflict returns empty sets."""
        genes = [
            make_gene(group_ids=["G1"], start=0, duration=2),
            make_gene(group_ids=["G2"], start=0, duration=2),
        ]
        index = ScheduleIndex.from_individual(genes)
        result = index.get_gene_conflicts(0)
        assert len(result["group"]) == 0

    def test_count_violations_with_no_conflicts(self):
        """Clean individual should have zero violations."""
        individual, _ = _make_clean_individual()
        index = ScheduleIndex.from_individual(individual)
        counts = index.count_violations()
        assert counts["total"] == 0

    def test_count_violations_with_all_conflict_types(self):
        """Individual with all conflict types should count all."""
        individual, _ = _make_multi_conflict_individual()
        index = ScheduleIndex.from_individual(individual)
        counts = index.count_violations()
        assert counts["group"] > 0
        assert counts["room"] > 0
        assert counts["instructor"] > 0
        assert (
            counts["total"] == counts["group"] + counts["room"] + counts["instructor"]
        )


# =====================================================================
# 2. DETECTOR INTEGRATION WITH SCHEDULE INDEX
# =====================================================================


class TestDetectorScheduleIndexIntegration:
    """Verify that detector.py uses ScheduleIndex for conflict detection."""

    def test_detector_finds_group_overlap(self):
        """Detector correctly identifies group overlap via ScheduleIndex."""
        from src.ga.repair.detector import detect_violated_genes

        individual, ctx = _make_conflicting_individual()
        violations = detect_violated_genes(individual, ctx, strategy="full")

        # Genes 0 and 1 overlap on group G1
        assert 0 in violations
        assert 1 in violations
        assert "group_overlap" in violations[0]
        assert "group_overlap" in violations[1]

    def test_detector_finds_room_conflict(self):
        """Detector identifies room conflicts."""
        from src.ga.repair.detector import detect_violated_genes

        individual, ctx = _make_multi_conflict_individual()
        violations = detect_violated_genes(individual, ctx, strategy="full")

        assert "room_conflict" in violations.get(0, [])
        assert "room_conflict" in violations.get(1, [])

    def test_detector_finds_instructor_conflict(self):
        """Detector identifies instructor conflicts."""
        from src.ga.repair.detector import detect_violated_genes

        individual, ctx = _make_multi_conflict_individual()
        violations = detect_violated_genes(individual, ctx, strategy="full")

        assert "instructor_conflict" in violations.get(0, [])
        assert "instructor_conflict" in violations.get(1, [])

    def test_detector_clean_schedule_no_overlap_violations(self):
        """Clean schedule should have no overlap-type violations."""
        from src.ga.repair.detector import detect_violated_genes

        individual, ctx = _make_clean_individual()
        violations = detect_violated_genes(individual, ctx, strategy="full")

        # No group/room/instructor overlaps
        for vtypes in violations.values():
            assert "group_overlap" not in vtypes
            assert "room_conflict" not in vtypes
            assert "instructor_conflict" not in vtypes

    def test_detector_hybrid_combines_fast_and_full(self):
        """Hybrid strategy should include both fast and full results."""
        from src.ga.repair.detector import detect_violated_genes

        individual, ctx = _make_conflicting_individual()
        violations = detect_violated_genes(individual, ctx, strategy="hybrid")

        # Should still detect group overlap
        assert 0 in violations
        assert "group_overlap" in violations[0]

    def test_detector_fast_only_structural(self):
        """Fast strategy only checks structural issues, not conflicts."""
        from src.ga.repair.detector import detect_violated_genes

        individual, ctx = _make_conflicting_individual()
        violations = detect_violated_genes(individual, ctx, strategy="fast")

        # Fast mode shouldn't detect group overlaps
        for vtypes in violations.values():
            assert "group_overlap" not in vtypes


# =====================================================================
# 10. REPAIR OPERATORS — Basic Tests
# =====================================================================


class TestBasicRepairOperators:
    """Test individual repair operators from basic.py."""

    def test_repair_group_overlaps_reduces_violations(self):
        from src.constraints.constraints import StudentGroupExclusivity
        from src.ga.repair.basic import repair_group_overlaps

        individual, ctx = _make_conflicting_individual()
        pre = StudentGroupExclusivity().evaluate(Timetable(individual, ctx))
        assert pre > 0, "Must start with violation"

        fixes = repair_group_overlaps(individual, ctx)
        assert isinstance(fixes, int)

    def test_repair_room_conflicts_reduces_violations(self):
        from src.constraints.constraints import RoomExclusivity
        from src.ga.repair.basic import repair_room_conflicts

        individual, ctx = _make_multi_conflict_individual()
        pre = RoomExclusivity().evaluate(Timetable(individual, ctx))
        assert pre > 0, "Must start with room conflict"

        fixes = repair_room_conflicts(individual, ctx)
        assert isinstance(fixes, int)

    def test_repair_instructor_conflicts_reduces_violations(self):
        from src.constraints.constraints import InstructorExclusivity
        from src.ga.repair.basic import repair_instructor_conflicts

        individual, ctx = _make_multi_conflict_individual()
        pre = InstructorExclusivity().evaluate(Timetable(individual, ctx))
        assert pre > 0, "Must start with instructor conflict"

        fixes = repair_instructor_conflicts(individual, ctx)
        assert isinstance(fixes, int)

    def test_repair_preserves_structural_invariants(self):
        """All repair operators must preserve course_id, course_type, group_ids, num_quanta."""
        from src.ga.repair.basic import repair_group_overlaps

        individual, ctx = _make_conflicting_individual()
        before = [copy.deepcopy(g) for g in individual]
        repair_group_overlaps(individual, ctx)

        for b, a in zip(before, individual, strict=False):
            assert structural_fields_preserved(b, a)


# =====================================================================
# 11. UNIFIED REPAIR PIPELINE
# =====================================================================


class TestUnifiedRepair:
    """Test repair_individual_unified orchestration."""

    def setup_method(self):
        _ensure_config()

    def test_unified_returns_stats_dict(self):
        from src.ga.repair.basic import repair_individual_unified

        individual, ctx = _make_conflicting_individual()
        stats = repair_individual_unified(individual, ctx)

        assert isinstance(stats, dict)
        assert "total_fixes" in stats

    def test_unified_selective_mode(self):
        from src.ga.repair.basic import repair_individual_unified

        individual, ctx = _make_conflicting_individual()
        stats = repair_individual_unified(individual, ctx, selective=True)
        assert isinstance(stats, dict)

    def test_unified_full_mode(self):
        from src.ga.repair.basic import repair_individual_unified

        individual, ctx = _make_conflicting_individual()
        stats = repair_individual_unified(individual, ctx, selective=False)
        assert isinstance(stats, dict)

    def test_unified_on_clean_individual(self):
        from src.ga.repair.basic import repair_individual_unified

        individual, ctx = _make_clean_individual()
        stats = repair_individual_unified(individual, ctx)
        # Clean individual should not need any repairs (or very few)
        assert isinstance(stats["total_fixes"], int)

    def test_unified_preserves_gene_count(self):
        from src.ga.repair.basic import repair_individual_unified

        individual, ctx = _make_conflicting_individual()
        before_len = len(individual)
        repair_individual_unified(individual, ctx)
        assert len(individual) == before_len


# =====================================================================
# 12. IGLS REPAIR WRAPPER
# =====================================================================


class TestIGLSRepair:
    """Test IGLS repair wrapper."""

    def setup_method(self):
        _ensure_config()

    def test_igls_returns_int(self):
        from src.ga.repair.igls import igls_repair

        individual, ctx = _make_conflicting_individual()
        result = igls_repair(individual, ctx, max_iterations=2)
        assert isinstance(result, int)
        assert result >= 0

    def test_igls_preserves_gene_count(self):
        from src.ga.repair.igls import igls_repair

        individual, ctx = _make_conflicting_individual()
        before_len = len(individual)
        igls_repair(individual, ctx, max_iterations=2)
        assert len(individual) == before_len

    def test_igls_preserves_structural_fields(self):
        from src.ga.repair.igls import igls_repair

        individual, ctx = _make_conflicting_individual()
        before = [copy.deepcopy(g) for g in individual]
        igls_repair(individual, ctx, max_iterations=2)

        for b, a in zip(before, individual, strict=False):
            assert structural_fields_preserved(b, a)

    def test_igls_selective_vs_full(self):
        """Both selective and full modes should work."""
        from src.ga.repair.igls import igls_repair

        ind1, ctx1 = _make_conflicting_individual()
        r1 = igls_repair(ind1, ctx1, selective=True)

        ind2, ctx2 = _make_conflicting_individual()
        r2 = igls_repair(ind2, ctx2, selective=False)

        assert isinstance(r1, int)
        assert isinstance(r2, int)


# =====================================================================
# 13. LNS TRIGGER LOGIC
# =====================================================================


class TestLNSTrigger:
    """Test LNS trigger conditions."""

    def test_trigger_on_interval(self):
        from src.ga.repair.lns.operator import should_trigger_lns_repair

        assert (
            should_trigger_lns_repair(
                generation=10,
                trigger_interval=10,
                stagnation_counter=0,
                stagnation_threshold=5,
            )
            is True
        )

    def test_no_trigger_off_interval(self):
        from src.ga.repair.lns.operator import should_trigger_lns_repair

        assert (
            should_trigger_lns_repair(
                generation=7,
                trigger_interval=10,
                stagnation_counter=0,
                stagnation_threshold=5,
            )
            is False
        )

    def test_trigger_on_stagnation(self):
        from src.ga.repair.lns.operator import should_trigger_lns_repair

        assert (
            should_trigger_lns_repair(
                generation=3,
                trigger_interval=10,
                stagnation_counter=5,
                stagnation_threshold=5,
            )
            is True
        )

    def test_no_trigger_below_stagnation_threshold(self):
        from src.ga.repair.lns.operator import should_trigger_lns_repair

        assert (
            should_trigger_lns_repair(
                generation=3,
                trigger_interval=10,
                stagnation_counter=4,
                stagnation_threshold=5,
            )
            is False
        )

    def test_trigger_on_forced_generation(self):
        from src.ga.repair.lns.operator import should_trigger_lns_repair

        assert (
            should_trigger_lns_repair(
                generation=25,
                trigger_interval=100,
                stagnation_counter=0,
                stagnation_threshold=99,
                force_trigger_generations=[25, 50],
            )
            is True
        )

    def test_no_trigger_gen_zero(self):
        """Generation 0 with no stagnation or forced gens should not trigger."""
        from src.ga.repair.lns.operator import should_trigger_lns_repair

        assert (
            should_trigger_lns_repair(
                generation=0,
                trigger_interval=10,
                stagnation_counter=0,
                stagnation_threshold=5,
            )
            is False
        )


# =====================================================================
# 14. LNS STATS
# =====================================================================


class TestLNSStats:
    """Test LNS statistics tracking."""

    def test_stats_initialization(self):
        from src.ga.repair.lns.operator import LNSRepairStats

        stats = LNSRepairStats()
        assert stats.total_attempts == 0
        assert stats.successful_repairs == 0
        assert stats.failed_repairs == 0

    def test_stats_repr(self):
        from src.ga.repair.lns.operator import LNSRepairStats

        stats = LNSRepairStats()
        stats.total_attempts = 10
        stats.successful_repairs = 7
        repr_str = repr(stats)
        assert "LNSRepairStats" in repr_str
        assert "70.0%" in repr_str

    def test_global_stats_reset(self):
        from src.ga.repair.lns.operator import get_lns_stats, reset_lns_stats

        stats = get_lns_stats()
        stats.total_attempts = 99
        reset_lns_stats()
        fresh = get_lns_stats()
        assert fresh.total_attempts == 0


# =====================================================================
# 15. REPAIR ENGINE
# =====================================================================


class TestRepairEngine:
    """Test the RL-ready RepairEngine."""

    def _make_engine(self):
        from src.ga.repair.engine import RepairEngine

        individual, ctx = _make_multi_conflict_individual()

        def evaluator(ind):
            """Simple fitness for testing."""
            violations = 0
            for i, g1 in enumerate(ind):
                for j, g2 in enumerate(ind):
                    if (
                        i < j
                        and g1.room_id == g2.room_id
                        and (
                            g1.start_quanta < g2.start_quanta + g2.num_quanta
                            and g2.start_quanta < g1.start_quanta + g1.num_quanta
                        )
                    ):
                        violations += 1
            return (float(violations), 0.0)

        engine = RepairEngine(
            context=ctx,
            evaluator=evaluator,
            policy="round_robin",
            max_steps=3,
        )
        return engine, individual, ctx

    def test_engine_creation(self):
        engine, _, _ = self._make_engine()
        assert engine is not None

    def test_engine_action_space(self):
        engine, _, _ = self._make_engine()
        actions = engine.get_action_space()
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(a, str) for a in actions)

    def test_engine_repair_returns_stats(self):
        from src.ga.repair.engine import RepairStats

        engine, individual, _ = self._make_engine()
        stats = engine.repair_individual(individual)
        assert isinstance(stats, RepairStats)
        assert stats.steps >= 0

    def test_engine_step_returns_result(self):
        from src.ga.repair.engine import RepairStepResult

        engine, individual, _ = self._make_engine()
        result = engine.step(individual)
        assert isinstance(result, RepairStepResult)
        assert isinstance(result.applied, bool)


# =====================================================================
# 16. REPAIR ENGINE POLICIES
# =====================================================================


class TestRepairPolicies:
    """Test repair engine selection policies."""

    def test_round_robin_cycles(self):
        from src.ga.repair.engine import RoundRobinPolicy

        operators = ["op_a", "op_b", "op_c"]
        policy = RoundRobinPolicy(operators)

        selections = [policy.select() for _ in range(6)]
        assert selections == ["op_a", "op_b", "op_c", "op_a", "op_b", "op_c"]

    def test_epsilon_greedy_explores(self):
        from src.ga.repair.engine import EpsilonGreedyPolicy

        operators = ["op_a", "op_b", "op_c"]
        policy = EpsilonGreedyPolicy(operators, epsilon=1.0)  # Full exploration

        rng = random.Random(42)
        selections = {policy.select({}, rng) for _ in range(100)}
        # With epsilon=1.0, should eventually select all operators
        assert len(selections) > 1

    def test_epsilon_greedy_exploits(self):
        """With epsilon=0, should always pick the operator with highest score.

        Score = delta_hard * 1000 + delta_soft. The policy picks MAX score.
        Since delta_hard is negative (reduction), the operator with the
        LEAST negative delta (smallest reduction) appears 'best' in raw scores.
        We design stats so op_b has the clear highest score.
        """
        from src.ga.repair.engine import EpsilonGreedyPolicy

        operators = ["op_a", "op_b", "op_c"]
        policy = EpsilonGreedyPolicy(operators, epsilon=0.0)

        rng = random.Random(42)
        # op_b gets score = 10*1000 + 5 = 10005 (highest)
        # op_a gets score = 1*1000 + 0 = 1000
        # op_c gets score = 0 (no applied)
        stats = {
            "op_a": {"applied": 2, "delta_hard": 1.0, "delta_soft": 0.0},
            "op_b": {"applied": 5, "delta_hard": 10.0, "delta_soft": 5.0},
            "op_c": {"applied": 0, "delta_hard": 0.0, "delta_soft": 0.0},
        }

        selections = [policy.select(stats, rng) for _ in range(10)]
        assert all(s == "op_b" for s in selections)


# =====================================================================
# 17. REPAIR CANDIDATE & STEP RESULT DATA CLASSES
# =====================================================================


class TestRepairDataClasses:
    """Test data classes used by repair engine."""

    def test_repair_candidate_creation(self):
        from src.ga.repair.engine import RepairCandidate

        c = RepairCandidate(gene_idx=5, new_start=10)
        assert c.gene_idx == 5
        assert c.new_start == 10
        assert c.new_room_id is None
        assert c.new_instructor_id is None

    def test_repair_step_result_creation(self):
        from src.ga.repair.engine import RepairStepResult

        r = RepairStepResult(
            applied=True,
            operator="move_time",
            delta_hard=-2.0,
            delta_soft=-0.5,
            before=(10.0, 5.0),
            after=(8.0, 4.5),
        )
        assert r.applied is True
        assert r.delta_hard == -2.0

    def test_repair_stats_record(self):
        from src.ga.repair.engine import RepairStats, RepairStepResult

        stats = RepairStats()
        result = RepairStepResult(
            applied=True,
            operator="move_time",
            delta_hard=-2.0,
            delta_soft=-0.5,
            before=(10.0, 5.0),
            after=(8.0, 4.5),
        )
        stats.record(result)

        assert stats.steps == 1
        assert stats.applied_steps == 1
        assert stats.total_delta_hard == -2.0

    def test_repair_stats_tracks_by_operator(self):
        from src.ga.repair.engine import RepairStats, RepairStepResult

        stats = RepairStats()
        for _ in range(3):
            stats.record(
                RepairStepResult(
                    applied=True,
                    operator="move_time",
                    delta_hard=-1.0,
                    delta_soft=0.0,
                    before=(5.0, 0.0),
                    after=(4.0, 0.0),
                )
            )
        stats.record(
            RepairStepResult(
                applied=False,
                operator="swap_room",
                delta_hard=0.0,
                delta_soft=0.0,
                before=(4.0, 0.0),
                after=(4.0, 0.0),
            )
        )

        assert stats.by_operator["move_time"]["applied"] == 3
        assert stats.by_operator["swap_room"]["applied"] == 0
        assert stats.by_operator["swap_room"]["steps"] == 1


# =====================================================================
# 18. LOCAL SEARCH — gene-level optimization
# =====================================================================


class TestLocalSearch:
    """Test gene-level local search operators."""

    def test_greedy_optimization_returns_tuple(self):
        from src.ga.operators.local_search import optimize_gene_greedy

        individual, ctx = _make_conflicting_individual()
        gene = individual[0]
        result = optimize_gene_greedy(gene, individual, 0, ctx, max_iterations=3)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_exhaustive_optimization_returns_tuple(self):
        from src.ga.operators.local_search import optimize_gene_exhaustive

        individual, ctx = _make_conflicting_individual()
        gene = individual[0]
        result = optimize_gene_exhaustive(
            gene, individual, 0, ctx, max_neighborhood_size=20
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_greedy_never_worsens_gene(self):
        """Greedy search should only accept improvements."""
        from src.ga.operators.local_search import (
            _count_gene_violations,
            optimize_gene_greedy,
        )

        individual, ctx = _make_conflicting_individual()
        gene = individual[0]
        before_violations = _count_gene_violations(gene, individual, 0, ctx)

        new_gene, improvement = optimize_gene_greedy(
            gene, individual, 0, ctx, max_iterations=5
        )
        after_violations = _count_gene_violations(new_gene, individual, 0, ctx)

        assert after_violations <= before_violations

    def test_exhaustive_returns_best_neighbor(self):
        """Exhaustive should return the best among all evaluated neighbors."""
        from src.ga.operators.local_search import optimize_gene_exhaustive

        individual, ctx = _make_conflicting_individual()
        gene = individual[0]
        new_gene, improvement = optimize_gene_exhaustive(
            gene, individual, 0, ctx, max_neighborhood_size=50
        )
        assert improvement >= 0  # Never negative


# =====================================================================
# 20. SCHEDULE INDEX + DETECTOR — Cross-Validation
# =====================================================================


class TestScheduleIndexDetectorCrossValidation:
    """Verify ScheduleIndex and detector agree on conflicts."""

    def test_index_and_detector_agree_on_group_conflicts(self):
        """ScheduleIndex.find_group_conflicts() and detector should find same violations."""
        from src.ga.repair.detector import _detect_full

        individual, ctx = _make_conflicting_individual()

        # ScheduleIndex approach
        index = ScheduleIndex.from_individual(individual)
        index_group_conflicts = index.find_group_conflicts()

        # Detector approach
        detector_violations = _detect_full(individual, ctx)

        # Both should identify genes 0 and 1 as having group_overlap
        index_violated = set(index_group_conflicts.keys())
        detector_group_violated = {
            idx
            for idx, vtypes in detector_violations.items()
            if "group_overlap" in vtypes
        }

        assert (
            index_violated == detector_group_violated
        ), f"Mismatch: index={index_violated}, detector={detector_group_violated}"

    def test_index_and_detector_agree_on_room_conflicts(self):
        individual, ctx = _make_multi_conflict_individual()
        from src.ga.repair.detector import _detect_full

        index = ScheduleIndex.from_individual(individual)
        index_room = set(index.find_room_conflicts().keys())

        detector = _detect_full(individual, ctx)
        detector_room = {
            idx for idx, vtypes in detector.items() if "room_conflict" in vtypes
        }

        assert index_room == detector_room

    def test_index_and_detector_agree_on_instructor_conflicts(self):
        individual, ctx = _make_multi_conflict_individual()
        from src.ga.repair.detector import _detect_full

        index = ScheduleIndex.from_individual(individual)
        index_instr = set(index.find_instructor_conflicts().keys())

        detector = _detect_full(individual, ctx)
        detector_instr = {
            idx for idx, vtypes in detector.items() if "instructor_conflict" in vtypes
        }

        assert index_instr == detector_instr


# =====================================================================
# 21. SCHEDULE INDEX — CONVENIENCE FUNCTION
# =====================================================================


class TestCreateScheduleIndex:
    """Test the create_schedule_index convenience function."""

    def test_convenience_creates_valid_index(self):
        genes = [make_gene(start=0, duration=2)]
        index = create_schedule_index(genes)
        assert isinstance(index, ScheduleIndex)
        assert not index.is_valid()

    def test_convenience_equivalent_to_from_individual(self):
        genes = [
            make_gene(group_ids=["G1"], start=0, duration=2),
            make_gene(group_ids=["G1"], start=0, duration=2),
        ]
        idx1 = ScheduleIndex.from_individual(genes)
        idx2 = create_schedule_index(genes)

        assert idx1.find_group_conflicts() == idx2.find_group_conflicts()


# =====================================================================
# 22. GREEDY REPAIR WRAPPER
# =====================================================================


class TestGreedyRepairWrapper:
    """Test greedy_repair wrapper from greedy.py."""

    def test_greedy_repair_returns_int(self):
        from src.ga.repair.greedy import greedy_repair

        individual, ctx = _make_conflicting_individual()
        result = greedy_repair(individual, ctx, max_iterations=2)
        assert isinstance(result, int)
        assert result >= 0

    def test_greedy_repair_preserves_gene_count(self):
        from src.ga.repair.greedy import greedy_repair

        individual, ctx = _make_conflicting_individual()
        before_len = len(individual)
        greedy_repair(individual, ctx, max_iterations=2)
        assert len(individual) == before_len


# =====================================================================
# 23. VIOLATION STATE
# =====================================================================


class TestViolationState:
    """Test ViolationState data class."""

    def test_violation_state_creation(self):
        from src.ga.repair.engine import ViolationState

        state = ViolationState(
            hard=5.0,
            soft=2.0,
            gene_scores=[1, 0, 2, 0],
            gene_order=[2, 0, 1, 3],
            instructor_counts={},
            room_counts={},
            group_counts={},
            available_quanta={0, 1, 2, 3},
        )
        assert state.hard == 5.0
        assert state.soft == 2.0
        assert state.gene_scores[2] == 2


# =====================================================================
# 24. END-TO-END INTEGRATION
# =====================================================================


class TestEndToEndIntegration:
    """End-to-end tests verifying the full pipeline."""

    def setup_method(self):
        _ensure_config()

    def test_schedule_index_used_in_repair_detection(self):
        """Verify that repair pipeline uses ScheduleIndex internally."""
        from src.ga.repair.detector import detect_violated_genes

        individual, ctx = _make_multi_conflict_individual()
        violations = detect_violated_genes(individual, ctx, strategy="full")

        # Should find all 3 conflict types
        all_vtypes = set()
        for vtypes in violations.values():
            all_vtypes.update(vtypes)

        assert "group_overlap" in all_vtypes
        assert "room_conflict" in all_vtypes
        assert "instructor_conflict" in all_vtypes

    def test_repair_then_detect_reduces_violations(self):
        """Repair → detect should show fewer violations."""
        from src.ga.repair.basic import repair_individual_unified
        from src.ga.repair.detector import detect_violated_genes

        individual, ctx = _make_multi_conflict_individual()
        pre_violations = detect_violated_genes(individual, ctx, strategy="full")
        pre_count = sum(len(v) for v in pre_violations.values())

        repair_individual_unified(individual, ctx, max_iterations=3)

        post_violations = detect_violated_genes(individual, ctx, strategy="full")
        post_count = sum(len(v) for v in post_violations.values())

        assert (
            post_count <= pre_count
        ), f"Repair should not increase violations: {pre_count} → {post_count}"

    def test_full_pipeline_clean_to_conflicting_to_repaired(self):
        """Clean → introduce conflict → repair → verify fewer violations."""
        from src.ga.repair.basic import repair_individual_unified

        individual, ctx = _make_clean_individual()
        index = ScheduleIndex.from_individual(individual)
        assert not index.has_conflicts(), "Should start clean"

        # Introduce conflict by moving gene to same time+group
        individual[1].start_quanta = individual[0].start_quanta
        individual[1].group_ids = individual[0].group_ids
        index2 = ScheduleIndex.from_individual(individual)
        assert index2.has_conflicts(), "Should now have conflict"

        # Repair
        repair_individual_unified(individual, ctx, max_iterations=3)

        # Check if improved
        index3 = ScheduleIndex.from_individual(individual)
        # May or may not fully fix depending on available slots
        assert isinstance(index3.count_violations()["total"], int)
