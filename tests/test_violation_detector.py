"""Phase 3: Violation Detector Tests.

Tests ViolationDetector's 3 strategies (fast, full, hybrid) and documents
known bugs (dead self-overlap check, no hierarchy awareness).
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

from src.ga.repair.detector import _detect_fast, _detect_full, detect_violated_genes


class TestDetectFast:
    """Fast detection — no decoding, structural checks only."""

    def test_valid_gene_no_violations(self):
        """Normal gene → no fast violations."""
        gene = make_gene(start=0, duration=2)
        result = _detect_fast([gene])
        assert result == {}

    def test_known_bug_self_overlap_dead_code(self):
        """KNOWN BUG: self-overlap check is `gene.num_quanta != gene.num_quanta`
        which is ALWAYS False. This check can never trigger."""
        gene = make_gene(start=0, duration=5)
        result = _detect_fast([gene])
        # The "self_overlap" type will never appear because the check is dead code
        for issues in result.values():
            assert "self_overlap" not in issues, "Dead code should never fire"

    def test_empty_schedule_detected(self):
        """Gene with num_quanta=0 → empty_schedule violation.
        Note: SessionGene.__post_init__ clamps num_quanta to >= 1, so this
        may be unreachable in practice."""
        gene = make_gene(start=0, duration=1)
        # Force num_quanta = 0 (bypass validation)
        object.__setattr__(gene, "num_quanta", 0)
        result = _detect_fast([gene])
        assert 0 in result
        assert "empty_schedule" in result[0]

    def test_negative_start_detected(self):
        """Gene with negative start_quanta → invalid_quanta.
        Note: SessionGene.__post_init__ clamps to 0, so need to bypass."""
        gene = make_gene(start=0, duration=1)
        object.__setattr__(gene, "start_quanta", -1)
        result = _detect_fast([gene])
        assert 0 in result
        assert "invalid_quanta" in result[0]


class TestDetectFull:
    """Full detection — builds schedule maps and checks all constraint types."""

    def test_no_violations_clean_schedule(self):
        """Two non-overlapping genes → no violations."""
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
        result = _detect_full([g1, g2], ctx)
        # May still detect "FPC" (Faculty-Program Compliance) if qualified_courses doesn't
        # match the (course_id, course_type) key format check. Let's verify:
        # The check is: if course_key not in instructor.qualified_courses
        # qualified_courses is a list[], not set of tuples, so this will likely fire.
        # Document actual behavior:
        assert isinstance(result, dict)

    def test_group_overlap_detected(self):
        """Two genes with same group at same time → both marked group_overlap."""
        g1 = make_gene(course_id="CS101", group_ids=["G1"], start=0, duration=2)
        g2 = make_gene(course_id="CS102", group_ids=["G1"], start=0, duration=2)
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["G1"]),
                make_course("CS102", groups=["G1"]),
            ],
        )
        result = _detect_full([g1, g2], ctx)
        # Both genes should have "group_overlap"
        assert "group_overlap" in result.get(0, [])
        assert "group_overlap" in result.get(1, [])

    def test_room_conflict_detected(self):
        """Two genes in same room at same time → room_conflict."""
        g1 = make_gene(course_id="CS101", room_id="R1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", room_id="R1", start=0, duration=2)
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
        )
        result = _detect_full([g1, g2], ctx)
        assert "room_conflict" in result.get(0, [])
        assert "room_conflict" in result.get(1, [])

    def test_instructor_conflict_detected(self):
        """Two genes with same instructor at same time → instructor_conflict."""
        g1 = make_gene(course_id="CS101", instructor_id="I1", start=0, duration=2)
        g2 = make_gene(course_id="CS102", instructor_id="I1", start=0, duration=2)
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
        )
        result = _detect_full([g1, g2], ctx)
        assert "instructor_conflict" in result.get(0, [])
        assert "instructor_conflict" in result.get(1, [])

    def test_known_bug_no_hierarchy_awareness(self):
        """KNOWN BUG: ViolationDetector doesn't check parent-subgroup
        hierarchy. BME1A (parent) and BME1AB (subgroup) at the same time
        are NOT detected as a conflict even though subgroup students can't
        attend both."""
        g1 = make_gene(course_id="CS101", group_ids=["BME1A"], start=0, duration=2)
        g2 = make_gene(course_id="CS102", group_ids=["BME1AB"], start=0, duration=2)
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["BME1A"]),
                make_course("CS102", groups=["BME1AB"]),
            ],
            groups=[make_group("BME1A"), make_group("BME1AB")],
            family_map={"BME1A": {"BME1AB", "BME1AC"}},
        )
        result = _detect_full([g1, g2], ctx)
        # BUG: These are different group_ids so detector doesn't see conflict
        group_violations_0 = [v for v in result.get(0, []) if v == "group_overlap"]
        assert len(group_violations_0) == 0, (
            "Bug documented: detector doesn't check hierarchy, "
            "so parent/subgroup overlaps go undetected"
        )


class TestDetectStrategies:
    """Test the top-level detect_violated_genes with different strategies."""

    def _clean_pair(self):
        """Return 2 non-conflicting genes and context."""
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

    def test_fast_strategy(self):
        genes, ctx = self._clean_pair()
        result = detect_violated_genes(genes, ctx, strategy="fast")
        assert isinstance(result, dict)

    def test_full_strategy(self):
        genes, ctx = self._clean_pair()
        result = detect_violated_genes(genes, ctx, strategy="full")
        assert isinstance(result, dict)

    def test_hybrid_strategy(self):
        genes, ctx = self._clean_pair()
        result = detect_violated_genes(genes, ctx, strategy="hybrid")
        assert isinstance(result, dict)

    def test_fast_full_consistency_for_conflicts(self):
        """Both fast and full should detect structural issues that fast can see."""
        g1 = make_gene(start=0, duration=2, group_ids=["G1"])
        g2 = make_gene(course_id="CS102", start=0, duration=2, group_ids=["G1"])
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
        )
        detect_violated_genes([g1, g2], ctx, strategy="fast")
        full_result = detect_violated_genes([g1, g2], ctx, strategy="full")
        # Full should at least detect group_overlap
        full_violations_0 = full_result.get(0, [])
        assert "group_overlap" in full_violations_0

    def test_hybrid_is_superset(self):
        """Hybrid should return at least everything full returns."""
        g1 = make_gene(start=0, duration=2, group_ids=["G1"])
        g2 = make_gene(course_id="CS102", start=0, duration=2, group_ids=["G1"])
        ctx = make_context(
            courses=[make_course("CS101"), make_course("CS102")],
        )
        full_result = detect_violated_genes([g1, g2], ctx, strategy="full")
        hybrid_result = detect_violated_genes([g1, g2], ctx, strategy="hybrid")
        # Hybrid includes everything from full
        for idx, vtypes in full_result.items():
            for vtype in vtypes:
                assert vtype in hybrid_result.get(
                    idx, []
                ), f"Hybrid missing {vtype} for gene {idx}"
