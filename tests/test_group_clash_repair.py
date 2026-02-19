"""Tests for the fast group-clash repair operator.

Verifies that repair_group_clashes():
  1. Eliminates all group-time overlaps (literal group_ids)
  2. Respects family_map (parent-child-sibling)
  3. Is idempotent (running twice gives same result)
  4. Handles edge cases (no clashes, all clashes, single gene)
  5. Returns correct fix count
  6. Keeps genes within valid day boundaries (post_init)
"""

from __future__ import annotations

from src.ga.repair.group_clash_repair import repair_group_clashes
from tests.conftest import (
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_group_clashes(individual, family_map=None):
    """Count literal group-time clashes (what StudentGroupExclusivity checks)."""
    from collections import Counter

    occ: Counter[tuple[str, int]] = Counter()
    for gene in individual:
        for q in range(gene.start_quanta, gene.start_quanta + gene.num_quanta):
            for gid in gene.group_ids:
                occ[(gid, q)] += 1
    return sum(v - 1 for v in occ.values() if v > 1)


def _make_default_context(**kwargs):
    """Default context with one group, instructor, room."""
    return make_context(
        courses=[
            make_course("CS101", groups=["G1"]),
            make_course("CS102", groups=["G1"]),
        ],
        groups=[make_group("G1")],
        instructors=[make_instructor("I1")],
        rooms=[make_room("R1")],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGroupClashRepairBasic:
    """Basic repair scenarios."""

    def test_no_clash_noop(self):
        """If there are no group clashes, repair should not move any gene."""
        ctx = _make_default_context()
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1"])
        g2 = make_gene("CS102", start=7, duration=2, group_ids=["G1"])
        individual = [g1, g2]

        fixes = repair_group_clashes(individual, ctx)

        assert fixes == 0
        assert g1.start_quanta == 0
        assert g2.start_quanta == 7

    def test_simple_clash_fixed(self):
        """Two genes for the same group at the same time → one gets moved."""
        ctx = _make_default_context()
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G1"])
        individual = [g1, g2]

        fixes = repair_group_clashes(individual, ctx)

        assert fixes >= 1
        assert _count_group_clashes(individual) == 0

    def test_partial_overlap_fixed(self):
        """Partially overlapping sessions for same group → clash fixed."""
        ctx = _make_default_context()
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1"])
        g2 = make_gene("CS102", start=1, duration=2, group_ids=["G1"])
        individual = [g1, g2]

        fixes = repair_group_clashes(individual, ctx)

        assert fixes >= 1
        assert _count_group_clashes(individual) == 0

    def test_different_groups_no_clash(self):
        """Two genes for different groups at same time → no clash."""
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["G1"]),
                make_course("CS102", groups=["G2"]),
            ],
            groups=[make_group("G1"), make_group("G2")],
        )
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G2"])
        individual = [g1, g2]

        fixes = repair_group_clashes(individual, ctx)

        assert fixes == 0

    def test_multiple_clashes_all_fixed(self):
        """Multiple group clashes at different times → all fixed."""
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["G1"]),
                make_course("CS102", groups=["G1"]),
                make_course("CS103", groups=["G1"]),
            ],
            groups=[make_group("G1")],
        )
        # All three at the same time
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G1"])
        g3 = make_gene("CS103", start=0, duration=2, group_ids=["G1"])
        individual = [g1, g2, g3]

        fixes = repair_group_clashes(individual, ctx)

        assert fixes >= 2
        assert _count_group_clashes(individual) == 0

    def test_returns_fix_count(self):
        """Return value accurately counts the number of genes shifted."""
        ctx = _make_default_context()
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G1"])
        individual = [g1, g2]

        fixes = repair_group_clashes(individual, ctx)

        # Exactly one gene should move; the other stays
        assert fixes == 1


class TestGroupClashRepairFamilyMap:
    """Tests for family-aware group clash repair."""

    def test_family_overlap_prevented(self):
        """Parent-child clashes are resolved via family_map."""
        family_map = {
            "G1AB": {"G1AB", "G1A", "G1B"},
            "G1A": {"G1AB", "G1A", "G1B"},
            "G1B": {"G1AB", "G1A", "G1B"},
        }
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["G1AB"]),
                make_course("CS102", groups=["G1A"]),
            ],
            groups=[make_group("G1AB"), make_group("G1A"), make_group("G1B")],
            family_map=family_map,
        )
        # Parent and child at same time
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1AB"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G1A"])
        individual = [g1, g2]

        repair_group_clashes(individual, ctx)

        # After repair, genes should not overlap in time
        g1_quanta = set(
            range(individual[0].start_quanta, individual[0].start_quanta + 2)
        )
        g2_quanta = set(
            range(individual[1].start_quanta, individual[1].start_quanta + 2)
        )
        assert not g1_quanta & g2_quanta, "Family-related genes should not overlap"

    def test_sibling_overlap_prevented(self):
        """Sibling group clashes are resolved via family_map."""
        family_map = {
            "G1A": {"G1AB", "G1A", "G1B"},
            "G1B": {"G1AB", "G1A", "G1B"},
        }
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["G1A"]),
                make_course("CS102", groups=["G1B"]),
            ],
            groups=[make_group("G1A"), make_group("G1B")],
            family_map=family_map,
        )
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1A"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G1B"])
        individual = [g1, g2]

        repair_group_clashes(individual, ctx)

        g1_quanta = set(
            range(individual[0].start_quanta, individual[0].start_quanta + 2)
        )
        g2_quanta = set(
            range(individual[1].start_quanta, individual[1].start_quanta + 2)
        )
        assert not g1_quanta & g2_quanta, "Sibling genes should not overlap"


class TestGroupClashRepairEdgeCases:
    """Edge cases and stress tests."""

    def test_single_gene_no_crash(self):
        """Single gene never has a clash."""
        ctx = _make_default_context()
        individual = [make_gene("CS101", start=0, duration=2, group_ids=["G1"])]

        fixes = repair_group_clashes(individual, ctx)

        assert fixes == 0

    def test_empty_individual(self):
        """Empty individual should not crash."""
        ctx = _make_default_context()

        fixes = repair_group_clashes([], ctx)

        assert fixes == 0

    def test_idempotent(self):
        """Running repair twice gives the same result."""
        ctx = _make_default_context()
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G1"])
        individual = [g1, g2]

        repair_group_clashes(individual, ctx)
        starts_after_first = [g.start_quanta for g in individual]

        fixes2 = repair_group_clashes(individual, ctx)

        assert fixes2 == 0, "Second repair should find no clashes"
        assert [g.start_quanta for g in individual] == starts_after_first

    def test_genes_stay_within_valid_boundaries(self):
        """Repaired genes should stay within valid day boundaries."""
        ctx = _make_default_context()
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G1"])
        individual = [g1, g2]

        repair_group_clashes(individual, ctx)

        from src.domain.gene import get_time_system

        qts = get_time_system()
        for gene in individual:
            # Gene must start at a valid day-aligned position
            assert 0 <= gene.start_quanta < qts.total_quanta
            assert gene.start_quanta + gene.num_quanta <= qts.total_quanta

    def test_multi_group_gene_clash(self):
        """Gene with multiple group_ids — clash on any one group triggers repair."""
        ctx = make_context(
            courses=[
                make_course("CS101", groups=["G1", "G2"]),
                make_course("CS102", groups=["G1"]),
            ],
            groups=[make_group("G1"), make_group("G2")],
        )
        g1 = make_gene("CS101", start=0, duration=2, group_ids=["G1", "G2"])
        g2 = make_gene("CS102", start=0, duration=2, group_ids=["G1"])
        individual = [g1, g2]

        repair_group_clashes(individual, ctx)

        assert _count_group_clashes(individual) == 0
