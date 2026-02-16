"""Tests for the CP-SAT repair pipeline.

Covers:
  - Supergroup + Cluster building
  - Gene partitioning
  - CP-SAT solver (small synthetic examples)
  - Solution merger
  - Full pipeline integration

All tests use synthetic data from conftest.py factory functions to avoid
external dependencies.
"""

from __future__ import annotations

import copy

import pytest

from tests.conftest import (
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _setup_time_system():
    """Ensure the global time system is configured for tests."""
    from src.domain.gene import get_time_system, set_time_system
    from src.io.time_system import QuantumTimeSystem

    if get_time_system() is None:
        qts = QuantumTimeSystem()
        set_time_system(qts)
    return get_time_system()


def _make_two_programme_context():
    """Create a minimal context with 2 programmes sharing one course.

    Programmes:
      - AAA: groups AAA1A, AAA2A
      - BBB: groups BBB1A, BBB2A

    Courses:
      - SHARED101(theory) enrolled by AAA1A and BBB1A
      - SHARED102(theory) enrolled by AAA2A and BBB2A  (2nd shared course)
      - AAA201(theory) enrolled by AAA2A only
      - BBB201(theory) enrolled by BBB2A only
    """
    _setup_time_system()

    courses = [
        make_course(
            "SHARED101",
            "theory",
            quanta=2,
            groups=["AAA1A", "BBB1A"],
            instructors=["I_SHARED"],
        ),
        make_course(
            "SHARED102",
            "theory",
            quanta=2,
            groups=["AAA2A", "BBB2A"],
            instructors=["I_SHARED"],
        ),
        make_course(
            "AAA201",
            "theory",
            quanta=2,
            groups=["AAA2A"],
            instructors=["I_A"],
        ),
        make_course(
            "BBB201",
            "theory",
            quanta=2,
            groups=["BBB2A"],
            instructors=["I_B"],
        ),
    ]
    groups = [
        make_group("AAA1A", courses=["SHARED101"]),
        make_group("AAA2A", courses=["SHARED102", "AAA201"]),
        make_group("BBB1A", courses=["SHARED101"]),
        make_group("BBB2A", courses=["SHARED102", "BBB201"]),
    ]
    instructors = [
        make_instructor("I_SHARED"),
        make_instructor("I_A"),
        make_instructor("I_B"),
    ]
    rooms = [make_room("R1"), make_room("R2")]

    family_map = {
        "AAA1A": {"AAA1A"},
        "AAA2A": {"AAA2A"},
        "BBB1A": {"BBB1A"},
        "BBB2A": {"BBB2A"},
    }

    ctx = make_context(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        family_map=family_map,
    )
    return ctx, family_map


def _make_independent_programmes_context():
    """Two programmes with ZERO shared courses (→ separate clusters)."""
    _setup_time_system()

    courses = [
        make_course(
            "AAA101",
            "theory",
            quanta=2,
            groups=["AAA1A"],
            instructors=["I_A"],
        ),
        make_course(
            "BBB101",
            "theory",
            quanta=2,
            groups=["BBB1A"],
            instructors=["I_B"],
        ),
    ]
    groups = [
        make_group("AAA1A", courses=["AAA101"]),
        make_group("BBB1A", courses=["BBB101"]),
    ]
    instructors = [make_instructor("I_A"), make_instructor("I_B")]
    rooms = [make_room("R1"), make_room("R2")]

    family_map = {"AAA1A": {"AAA1A"}, "BBB1A": {"BBB1A"}}
    ctx = make_context(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        family_map=family_map,
    )
    return ctx, family_map


# ══════════════════════════════════════════════════════════════════════════
# 1. Supergroup / Cluster tests
# ══════════════════════════════════════════════════════════════════════════


class TestSupergroups:
    """Tests for build_supergroups and build_clusters."""

    def test_build_supergroups_basic(self):
        ctx, _ = _make_two_programme_context()
        from src.domain.supergroup import build_supergroups

        sgs = build_supergroups(ctx)
        assert "AAA" in sgs
        assert "BBB" in sgs
        assert "AAA1A" in sgs["AAA"].group_ids
        assert "AAA2A" in sgs["AAA"].group_ids
        assert "BBB1A" in sgs["BBB"].group_ids

    def test_build_clusters_merges_coupled_programmes(self):
        """AAA and BBB share 2 courses → should be in same cluster."""
        ctx, _ = _make_two_programme_context()
        from src.domain.supergroup import build_clusters

        clusters = build_clusters(ctx, min_shared_courses=2)
        # With 2 shared courses (SHARED101 + SHARED102), AAA and BBB
        # should be merged into a single cluster.
        assert len(clusters) == 1
        cl = clusters[0]
        assert "AAA" in cl.programmes
        assert "BBB" in cl.programmes

    def test_build_clusters_independent_programmes(self):
        """No shared courses → separate clusters."""
        ctx, _ = _make_independent_programmes_context()
        from src.domain.supergroup import build_clusters

        clusters = build_clusters(ctx, min_shared_courses=2)
        assert len(clusters) == 2
        progs = {frozenset(c.programmes) for c in clusters}
        assert frozenset({"AAA"}) in progs
        assert frozenset({"BBB"}) in progs

    def test_extract_programme_prefix(self):
        from src.domain.supergroup import extract_programme_prefix

        assert extract_programme_prefix("BCT1AB") == "BCT"
        assert extract_programme_prefix("BEI3A") == "BEI"
        assert extract_programme_prefix("MEE1A") == "MEE"
        assert extract_programme_prefix("BAR2AB") == "BAR"


# ══════════════════════════════════════════════════════════════════════════
# 2. Partitioner tests
# ══════════════════════════════════════════════════════════════════════════


class TestPartitioner:
    """Tests for partition_genes."""

    def test_partition_with_bridge_genes(self):
        """When min_shared_courses is high enough to keep programmes apart,
        a gene with groups in both should be a bridge."""
        ctx, _ = _make_two_programme_context()
        from src.ga.repair.cp.partitioner import partition_genes

        genes = [
            # Bridge gene: groups span both programmes
            make_gene(
                "SHARED101",
                "theory",
                "I_SHARED",
                ["AAA1A", "BBB1A"],
                "R1",
                start=0,
                duration=2,
            ),
            # AAA-only gene
            make_gene("AAA201", "theory", "I_A", ["AAA2A"], "R1", start=7, duration=2),
            # BBB-only gene
            make_gene("BBB201", "theory", "I_B", ["BBB2A"], "R2", start=7, duration=2),
        ]

        # Use min_shared_courses=99 so AAA and BBB stay as separate clusters
        part = partition_genes(genes, ctx, min_shared_courses=99)

        # Gene 0 (SHARED101) has groups from both AAA and BBB → bridge
        assert 0 in part.bridge_gene_indices
        assert part.gene_cluster_map[0] == "__bridge__"

    def test_partition_merged_cluster_no_bridges(self):
        """When programmes merge into one cluster, no bridge genes."""
        ctx, _ = _make_two_programme_context()
        from src.ga.repair.cp.partitioner import partition_genes

        genes = [
            make_gene(
                "SHARED101",
                "theory",
                "I_SHARED",
                ["AAA1A", "BBB1A"],
                "R1",
                start=0,
                duration=2,
            ),
            make_gene("AAA201", "theory", "I_A", ["AAA2A"], "R1", start=7, duration=2),
            make_gene("BBB201", "theory", "I_B", ["BBB2A"], "R2", start=7, duration=2),
        ]

        # min_shared_courses=2 → AAA+BBB share SHARED101+SHARED102 → merged
        part = partition_genes(genes, ctx, min_shared_courses=2)

        # All genes in one cluster, no bridges
        assert len(part.bridge_gene_indices) == 0
        assert len(part.clusters) == 1

    def test_partition_independent_no_bridges(self):
        """With independent programmes, no bridge genes expected."""
        ctx, _ = _make_independent_programmes_context()
        from src.ga.repair.cp.partitioner import partition_genes

        genes = [
            make_gene("AAA101", "theory", "I_A", ["AAA1A"], "R1", start=0, duration=2),
            make_gene("BBB101", "theory", "I_B", ["BBB1A"], "R2", start=7, duration=2),
        ]

        part = partition_genes(genes, ctx, min_shared_courses=2)

        # No shared instructors & no shared groups → no bridges
        assert len(part.bridge_gene_indices) == 0
        assert len(part.clusters) == 2
        # Each cluster should have exactly one gene
        total_cluster_genes = sum(len(v) for v in part.cluster_gene_indices.values())
        assert total_cluster_genes == 2

    def test_partition_shared_instructor_bridge(self):
        """An instructor qualified in both clusters' courses → bridge genes."""
        _setup_time_system()
        from src.ga.repair.cp.partitioner import partition_genes

        # Create two independent programmes but with a SHARED instructor
        # qualified for courses in both programmes.
        c_a = make_course(
            "AAA101",
            "theory",
            quanta=2,
            groups=["AAA1A"],
            instructors=["I_CROSS", "I_A"],
        )
        c_b = make_course(
            "BBB101",
            "theory",
            quanta=2,
            groups=["BBB1A"],
            instructors=["I_CROSS", "I_B"],
        )
        ctx = make_context(
            courses=[c_a, c_b],
            groups=[
                make_group("AAA1A", courses=["AAA101"]),
                make_group("BBB1A", courses=["BBB101"]),
            ],
            instructors=[
                make_instructor("I_CROSS"),
                make_instructor("I_A"),
                make_instructor("I_B"),
            ],
            rooms=[make_room("R1"), make_room("R2")],
            family_map={"AAA1A": {"AAA1A"}, "BBB1A": {"BBB1A"}},
        )

        genes = [
            make_gene(
                "AAA101", "theory", "I_CROSS", ["AAA1A"], "R1", start=0, duration=2
            ),
            make_gene(
                "BBB101", "theory", "I_CROSS", ["BBB1A"], "R2", start=7, duration=2
            ),
        ]

        part = partition_genes(genes, ctx, min_shared_courses=2)

        # I_CROSS is qualified in both programmes → shared instructor
        assert "I_CROSS" in part.shared_instructor_ids
        assert 0 in part.bridge_gene_indices
        assert 1 in part.bridge_gene_indices


# ══════════════════════════════════════════════════════════════════════════
# 3. Solver tests
# ══════════════════════════════════════════════════════════════════════════


class TestCPSATSolver:
    """Tests for the CP-SAT solver on small instances."""

    @pytest.fixture(autouse=True)
    def setup_time(self):
        _setup_time_system()

    def test_trivial_single_gene(self):
        """Single gene with one valid room/instructor → always feasible."""
        ctx, family_map = _make_independent_programmes_context()
        from src.ga.repair.cp.solver import CPSATSolver

        genes = [
            make_gene("AAA101", "theory", "I_A", ["AAA1A"], "R1", start=0, duration=2),
        ]

        solver = CPSATSolver(ctx, family_map, timeout_seconds=10, num_workers=1)
        result = solver.solve(genes, [0], warm_start=True)

        assert result.success is True
        assert 0 in result.assignments
        iid, rid, sq = result.assignments[0]
        assert iid in ("I_A", "I_B")  # Must be a valid instructor
        assert rid in ("R1", "R2")  # Must be a valid room

    def test_two_genes_no_conflict(self):
        """Two independent genes → both should be feasibly scheduled."""
        ctx, family_map = _make_independent_programmes_context()
        from src.ga.repair.cp.solver import CPSATSolver

        genes = [
            make_gene("AAA101", "theory", "I_A", ["AAA1A"], "R1", start=0, duration=2),
            make_gene("BBB101", "theory", "I_B", ["BBB1A"], "R2", start=0, duration=2),
        ]

        solver = CPSATSolver(ctx, family_map, timeout_seconds=10, num_workers=1)
        result = solver.solve(genes, [0, 1], warm_start=True)

        assert result.success is True
        assert len(result.assignments) == 2

    def test_group_conflict_resolved(self):
        """Two genes share a group at the same time → solver must separate them."""
        _setup_time_system()

        c1 = make_course("X101", "theory", quanta=2, groups=["G1"], instructors=["I1"])
        c2 = make_course("X102", "theory", quanta=2, groups=["G1"], instructors=["I2"])

        ctx = make_context(
            courses=[c1, c2],
            groups=[make_group("G1", courses=["X101", "X102"])],
            instructors=[make_instructor("I1"), make_instructor("I2")],
            rooms=[make_room("R1"), make_room("R2")],
            family_map={"G1": {"G1"}},
        )

        # Both genes start at the same time → group conflict
        genes = [
            make_gene("X101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("X102", "theory", "I2", ["G1"], "R1", start=0, duration=2),
        ]

        from src.ga.repair.cp.solver import CPSATSolver

        solver = CPSATSolver(ctx, {"G1": {"G1"}}, timeout_seconds=10, num_workers=1)
        result = solver.solve(genes, [0, 1], warm_start=True)

        assert result.success is True
        _, _, sq0 = result.assignments[0]
        _, _, sq1 = result.assignments[1]
        # Their intervals must not overlap
        g0_end = sq0 + 2
        g1_end = sq1 + 2
        assert not (sq0 < g1_end and sq1 < g0_end), (
            f"Group conflict still present: gene0=[{sq0},{g0_end}), "
            f"gene1=[{sq1},{g1_end})"
        )

    def test_instructor_conflict_resolved(self):
        """Two genes with the same instructor at the same time → separated."""
        _setup_time_system()

        c1 = make_course("Y101", "theory", quanta=2, groups=["G1"], instructors=["I1"])
        c2 = make_course("Y102", "theory", quanta=2, groups=["G2"], instructors=["I1"])

        ctx = make_context(
            courses=[c1, c2],
            groups=[
                make_group("G1", courses=["Y101"]),
                make_group("G2", courses=["Y102"]),
            ],
            instructors=[make_instructor("I1")],
            rooms=[make_room("R1"), make_room("R2")],
            family_map={"G1": {"G1"}, "G2": {"G2"}},
        )

        genes = [
            make_gene("Y101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("Y102", "theory", "I1", ["G2"], "R1", start=0, duration=2),
        ]

        from src.ga.repair.cp.solver import CPSATSolver

        solver = CPSATSolver(
            ctx, {"G1": {"G1"}, "G2": {"G2"}}, timeout_seconds=10, num_workers=1
        )
        result = solver.solve(genes, [0, 1], warm_start=True)

        assert result.success is True
        _, _, sq0 = result.assignments[0]
        _, _, sq1 = result.assignments[1]
        g0_end = sq0 + 2
        g1_end = sq1 + 2
        assert not (sq0 < g1_end and sq1 < g0_end), "Instructor conflict still present"

    def test_frozen_assignment_respected(self):
        """Frozen gene occupies a slot, new gene must avoid it."""
        _setup_time_system()
        from src.ga.repair.cp.solver import CPSATSolver, FrozenAssignment

        c1 = make_course("Z101", "theory", quanta=2, groups=["G1"], instructors=["I1"])
        c2 = make_course("Z102", "theory", quanta=2, groups=["G1"], instructors=["I1"])

        ctx = make_context(
            courses=[c1, c2],
            groups=[make_group("G1", courses=["Z101", "Z102"])],
            instructors=[make_instructor("I1")],
            rooms=[make_room("R1"), make_room("R2")],
            family_map={"G1": {"G1"}},
        )

        genes = [
            make_gene("Z101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("Z102", "theory", "I1", ["G1"], "R1", start=0, duration=2),
        ]

        # Freeze gene 0 at start=0
        frozen = [FrozenAssignment.from_gene(0, genes[0])]

        solver = CPSATSolver(ctx, {"G1": {"G1"}}, timeout_seconds=10, num_workers=1)
        # Only solve gene 1 with gene 0 frozen
        result = solver.solve(genes, [1], frozen=frozen, warm_start=True)

        assert result.success is True
        _, _, sq1 = result.assignments[1]
        # Gene 1 must not overlap with frozen gene 0 (start=0, dur=2)
        assert not (
            sq1 < 2 and sq1 + 2 > 0
        ), f"Gene 1 at {sq1} overlaps frozen gene at 0"

    def test_empty_gene_indices(self):
        """Solving with zero genes returns trivial success."""
        ctx, family_map = _make_independent_programmes_context()
        from src.ga.repair.cp.solver import CPSATSolver

        solver = CPSATSolver(ctx, family_map, timeout_seconds=5)
        result = solver.solve([], [], warm_start=False)
        assert result.success is True
        assert result.status == "TRIVIAL"


# ══════════════════════════════════════════════════════════════════════════
# 4. Merger tests
# ══════════════════════════════════════════════════════════════════════════


class TestMerger:
    """Tests for apply_cp_results and audit_hard_violations."""

    def test_apply_changes_values(self):
        """CP result should update instructor, room, start on affected gene."""
        _setup_time_system()
        from src.ga.repair.cp.merger import apply_cp_results
        from src.ga.repair.cp.solver import CPSolveResult

        genes = [
            make_gene("X101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("X102", "theory", "I2", ["G2"], "R2", start=7, duration=2),
        ]

        result = CPSolveResult(
            success=True,
            assignments={0: ("I_NEW", "R_NEW", 14)},
        )

        new_genes = apply_cp_results(genes, result)
        assert new_genes[0].instructor_id == "I_NEW"
        assert new_genes[0].room_id == "R_NEW"
        assert new_genes[0].start_quanta == 14
        # Gene 1 should be unchanged
        assert new_genes[1].instructor_id == "I2"
        assert new_genes[1].room_id == "R2"
        assert new_genes[1].start_quanta == 7

    def test_apply_does_not_modify_original(self):
        """The original gene list must not be mutated."""
        _setup_time_system()
        from src.ga.repair.cp.merger import apply_cp_results
        from src.ga.repair.cp.solver import CPSolveResult

        genes = [
            make_gene("X101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
        ]
        original_start = genes[0].start_quanta

        result = CPSolveResult(
            success=True,
            assignments={0: ("I_NEW", "R_NEW", 21)},
        )

        new_genes = apply_cp_results(genes, result)
        assert new_genes[0].start_quanta == 21
        assert genes[0].start_quanta == original_start

    def test_apply_failed_result_is_noop(self):
        """A failed result should not modify any genes."""
        _setup_time_system()
        from src.ga.repair.cp.merger import apply_cp_results
        from src.ga.repair.cp.solver import CPSolveResult

        genes = [
            make_gene("X101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
        ]

        result = CPSolveResult(
            success=False,
            assignments={0: ("I_NEW", "R_NEW", 14)},
        )

        new_genes = apply_cp_results(genes, result)
        assert new_genes[0].instructor_id == "I1"
        assert new_genes[0].start_quanta == 0

    def test_audit_violation_free(self):
        """A violation-free schedule should audit to all zeros."""
        _setup_time_system()
        from src.ga.repair.cp.merger import audit_hard_violations

        c1 = make_course("A101", "theory", quanta=2, groups=["G1"], instructors=["I1"])
        c2 = make_course("A102", "theory", quanta=2, groups=["G2"], instructors=["I2"])

        ctx = make_context(
            courses=[c1, c2],
            groups=[
                make_group("G1", courses=["A101"]),
                make_group("G2", courses=["A102"]),
            ],
            instructors=[make_instructor("I1"), make_instructor("I2")],
            rooms=[make_room("R1"), make_room("R2")],
        )

        # No overlaps: different groups, instructors, rooms, same time is OK
        genes = [
            make_gene("A101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("A102", "theory", "I2", ["G2"], "R2", start=0, duration=2),
        ]

        bd = audit_hard_violations(genes, ctx)
        assert sum(bd.values()) == 0


# ══════════════════════════════════════════════════════════════════════════
# 5. Pipeline integration tests
# ══════════════════════════════════════════════════════════════════════════


class TestPipeline:
    """Integration tests for the full CP repair pipeline."""

    @pytest.fixture(autouse=True)
    def setup_time(self):
        _setup_time_system()

    def test_pipeline_resolves_group_conflict(self):
        """Pipeline should resolve a simple group overlap."""
        c1 = make_course("P101", "theory", quanta=2, groups=["G1"], instructors=["I1"])
        c2 = make_course("P102", "theory", quanta=2, groups=["G1"], instructors=["I2"])

        ctx = make_context(
            courses=[c1, c2],
            groups=[make_group("G1", courses=["P101", "P102"])],
            instructors=[make_instructor("I1"), make_instructor("I2")],
            rooms=[make_room("R1"), make_room("R2")],
            family_map={"G1": {"G1"}},
        )

        # Both at same time → group conflict
        genes = [
            make_gene("P101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("P102", "theory", "I2", ["G1"], "R1", start=0, duration=2),
        ]

        from src.ga.repair.cp import CPRepairPipeline

        pipeline = CPRepairPipeline(
            timeout_global=10,
            timeout_cluster=10,
            num_workers=1,
            min_shared_courses=2,
        )

        repaired, stats = pipeline.repair(genes, ctx, {"G1": {"G1"}})

        # Check that group overlap is resolved
        sq0 = repaired[0].start_quanta
        sq1 = repaired[1].start_quanta
        g0_end = sq0 + 2
        g1_end = sq1 + 2
        assert not (
            sq0 < g1_end and sq1 < g0_end
        ), f"Group conflict NOT resolved: [{sq0},{g0_end}) vs [{sq1},{g1_end})"

    def test_pipeline_preserves_structural_fields(self):
        """Course_id, course_type, group_ids, num_quanta must never change."""
        c1 = make_course("S101", "theory", quanta=2, groups=["G1"], instructors=["I1"])

        ctx = make_context(
            courses=[c1],
            groups=[make_group("G1", courses=["S101"])],
            instructors=[make_instructor("I1")],
            rooms=[make_room("R1")],
            family_map={"G1": {"G1"}},
        )

        genes = [
            make_gene("S101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
        ]

        from src.ga.repair.cp import CPRepairPipeline

        pipeline = CPRepairPipeline(
            timeout_global=10,
            timeout_cluster=10,
            num_workers=1,
        )

        repaired, _ = pipeline.repair(genes, ctx, {"G1": {"G1"}})

        for orig, rep in zip(genes, repaired, strict=True):
            assert orig.course_id == rep.course_id
            assert orig.course_type == rep.course_type
            assert orig.group_ids == rep.group_ids
            assert orig.num_quanta == rep.num_quanta

    def test_pipeline_stats_populated(self):
        """Pipeline stats should have correct metadata."""
        ctx, family_map = _make_independent_programmes_context()

        genes = [
            make_gene("AAA101", "theory", "I_A", ["AAA1A"], "R1", start=0, duration=2),
            make_gene("BBB101", "theory", "I_B", ["BBB1A"], "R2", start=7, duration=2),
        ]

        from src.ga.repair.cp import CPRepairPipeline

        pipeline = CPRepairPipeline(
            timeout_global=10,
            timeout_cluster=10,
            num_workers=1,
        )

        _, stats = pipeline.repair(genes, ctx, family_map)

        assert stats.total_genes == 2
        assert stats.num_clusters >= 1
        assert stats.total_time >= 0  # May be 0 on fast machines with tiny instances
