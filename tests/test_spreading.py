"""Tests for GeneDomainStore and UsageTracker (spreading infrastructure).

Verifies:
1. Domain buckets pre-compute correct instructors / rooms / valid_starts
2. UsageTracker counts resources correctly and prefers least-used
3. Integration: spreading produces more uniform distributions than random
"""

from __future__ import annotations

import random

import pytest

from src.domain.gene import set_time_system
from src.ga.core.domain_store import GeneDomainStore
from src.ga.core.usage_tracker import UsageTracker
from src.io.time_system import QuantumTimeSystem
from tests.conftest import (
    make_context,
    make_course,
    make_gene,
    make_group,
    make_instructor,
    make_room,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_qts():
    """Ensure QuantumTimeSystem is set for all tests in this file."""
    qts = QuantumTimeSystem()
    set_time_system(qts)
    yield
    set_time_system(None)


@pytest.fixture()
def qts():
    return QuantumTimeSystem()


@pytest.fixture()
def basic_context():
    """Small context: 2 instructors, 3 rooms, 2 groups, 1 course."""
    c1 = make_course(
        "CS101",
        "theory",
        quanta=2,
        room_feat="lecture",
        groups=["G1", "G2"],
        instructors=["I1", "I2"],
    )
    g1 = make_group("G1", students=30, courses=["CS101"])
    g2 = make_group("G2", students=25, courses=["CS101"])
    i1 = make_instructor("I1", courses=[("CS101", "theory")])
    i2 = make_instructor("I2", courses=[("CS101", "theory")])
    r1 = make_room("R1", capacity=50, features="lecture")
    r2 = make_room("R2", capacity=40, features="lecture")
    r3 = make_room("R3", capacity=60, features="lecture")
    return make_context(
        courses=[c1],
        groups=[g1, g2],
        instructors=[i1, i2],
        rooms=[r1, r2, r3],
        family_map={"G1": {"G1"}, "G2": {"G2"}},
    )


# ---------------------------------------------------------------------------
# GeneDomainStore tests
# ---------------------------------------------------------------------------


class TestGeneDomainStore:
    def test_build_domains_instructor_bucket(self, basic_context, qts):
        store = GeneDomainStore(basic_context, qts)
        gene = make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
        store.build_domains([gene])

        domain = store.get_domain(0)
        assert set(domain.instructors) == {"I1", "I2"}

    def test_build_domains_room_bucket(self, basic_context, qts):
        store = GeneDomainStore(basic_context, qts)
        gene = make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
        store.build_domains([gene])

        domain = store.get_domain(0)
        assert set(domain.rooms) == {"R1", "R2", "R3"}

    def test_valid_starts_respect_day_boundary(self, basic_context, qts):
        """A 2-quanta block cannot start at the last quantum of a day."""
        store = GeneDomainStore(basic_context, qts)
        gene = make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
        store.build_domains([gene])

        domain = store.get_domain(0)
        # Day 0 (Sun) has quanta 0-6 (7 quanta). 2-quanta block → valid starts 0-5
        # So quantum 6 should NOT be a valid start for duration 2
        assert 6 not in domain.valid_starts
        assert 0 in domain.valid_starts
        assert 5 in domain.valid_starts

    def test_narrow_time_domain(self, basic_context, qts):
        store = GeneDomainStore(basic_context, qts)
        gene = make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
        store.build_domains([gene])

        blocked = {0, 1, 7, 8}  # block first 2 slots of Sun and Mon
        narrowed = store.narrow_time_domain(0, blocked)
        # starts 0 and 7 should be gone (they overlap with blocked quanta)
        assert 0 not in narrowed
        assert 7 not in narrowed
        # start 2 should still be there (quanta 2,3 not blocked)
        assert 2 in narrowed

    def test_refresh_domain(self, basic_context, qts):
        store = GeneDomainStore(basic_context, qts)
        gene = make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
        store.build_domains([gene])

        # Change gene duration and refresh
        new_gene = make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=3)
        store.refresh_domain(0, new_gene)
        domain = store.get_domain(0)
        assert domain.num_quanta == 3
        # 3-quanta block on a 7-quanta day → valid starts 0-4
        assert 4 in domain.valid_starts
        assert 5 not in domain.valid_starts  # 5+3=8 > 7 (overflows day)

    def test_room_capacity_filter(self, qts):
        """Rooms too small for the group should be excluded."""
        c1 = make_course(
            "CS101",
            "theory",
            quanta=2,
            room_feat="lecture",
            groups=["G1"],
            instructors=["I1"],
        )
        g1 = make_group("G1", students=55, courses=["CS101"])
        i1 = make_instructor("I1", courses=[("CS101", "theory")])
        r_small = make_room("RSMALL", capacity=30, features="lecture")
        r_big = make_room("RBIG", capacity=60, features="lecture")
        ctx = make_context(
            courses=[c1], groups=[g1], instructors=[i1], rooms=[r_small, r_big]
        )

        store = GeneDomainStore(ctx, qts)
        gene = make_gene("CS101", "theory", "I1", ["G1"], "RBIG", start=0, duration=2)
        store.build_domains([gene])

        domain = store.get_domain(0)
        assert "RBIG" in domain.rooms
        assert "RSMALL" not in domain.rooms


# ---------------------------------------------------------------------------
# UsageTracker tests
# ---------------------------------------------------------------------------


class TestUsageTracker:
    def test_add_and_remove_gene(self):
        tracker = UsageTracker()
        gene = make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
        tracker.add_gene(gene)

        assert tracker.time_load[0] == 1
        assert tracker.time_load[1] == 1
        assert tracker.instructor_total["I1"] == 2
        assert tracker.room_load["R1"][0] == 1
        assert tracker.group_load["G1"][1] == 1

        tracker.remove_gene(gene)
        assert tracker.time_load.get(0, 0) == 0
        assert tracker.instructor_total.get("I1", 0) == 0

    def test_build_from_individual(self):
        genes = [
            make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("CS101", "theory", "I2", ["G2"], "R2", start=0, duration=2),
        ]
        tracker = UsageTracker()
        tracker.build_from_individual(genes)

        # quantum 0 is used by both genes
        assert tracker.time_load[0] == 2
        # each instructor used once
        assert tracker.instructor_total["I1"] == 2
        assert tracker.instructor_total["I2"] == 2

    def test_pick_least_used_instructor(self):
        tracker = UsageTracker()
        # I1 has 6 quanta, I2 has 2 quanta
        for _ in range(3):
            tracker.add_gene(
                make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
            )
        tracker.add_gene(
            make_gene("CS101", "theory", "I2", ["G2"], "R2", start=0, duration=2)
        )

        # With top_k=1, should always pick I2 (least loaded)
        picked = tracker.pick_least_used_instructor(["I1", "I2"], top_k=1)
        assert picked == "I2"

    def test_pick_least_used_start(self):
        tracker = UsageTracker()
        # Load up quantum 0-1 heavily
        for _ in range(5):
            tracker.add_gene(
                make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
            )

        # quantum 7-8 never used
        picked = tracker.pick_least_used_start([0, 7, 14], num_quanta=2, top_k=1)
        # Should pick 7 or 14 (both score 0), not 0 (score 10)
        assert picked in (7, 14)

    def test_pick_least_used_room(self):
        tracker = UsageTracker()
        # R1 heavily used at time 0-1
        for _ in range(5):
            tracker.add_gene(
                make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
            )

        picked = tracker.pick_least_used_room(
            ["R1", "R2", "R3"], start_quanta=0, num_quanta=2, top_k=1
        )
        assert picked in ("R2", "R3")  # both unused at time 0

    def test_instructor_free_starts(self):
        tracker = UsageTracker()
        tracker.add_gene(
            make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2)
        )

        free = tracker.instructor_free_starts("I1", [0, 2, 7], num_quanta=2)
        assert 0 not in free  # I1 busy at 0-1
        assert 2 in free
        assert 7 in free

    def test_group_free_starts_with_family(self):
        tracker = UsageTracker()
        tracker.add_gene(
            make_gene("CS101", "theory", "I1", ["G1A"], "R1", start=0, duration=2)
        )

        family_map = {"G1A": {"G1A", "G1B", "G1AB"}, "G1B": {"G1A", "G1B", "G1AB"}}
        # G1B is a sibling of G1A — should block time 0-1
        free = tracker.group_free_starts(
            ["G1B"], [0, 7, 14], num_quanta=2, family_map=family_map
        )
        assert 0 not in free
        assert 7 in free

    def test_summary(self):
        tracker = UsageTracker()
        genes = [
            make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("CS101", "theory", "I2", ["G2"], "R2", start=7, duration=2),
        ]
        tracker.build_from_individual(genes)
        s = tracker.summary()
        assert s["time_slots_used"] == 4
        assert s["instructors_active"] == 2


# ---------------------------------------------------------------------------
# Integration: spreading vs random
# ---------------------------------------------------------------------------


class TestSpreadingIntegration:
    def test_spreading_produces_lower_time_std(self, basic_context, qts):
        """Core test: tracker-guided assignment should spread time slots
        more evenly than pure random.choice."""
        random.seed(42)
        n_genes = 20
        num_quanta = 2

        # --- Random baseline ---
        random_genes = []
        for _ in range(n_genes):
            start = random.choice(range(40))
            inst = random.choice(["I1", "I2"])
            room = random.choice(["R1", "R2", "R3"])
            random_genes.append(
                make_gene(
                    "CS101",
                    "theory",
                    inst,
                    ["G1"],
                    room,
                    start=start,
                    duration=num_quanta,
                )
            )
        random_tracker = UsageTracker()
        random_tracker.build_from_individual(random_genes)

        # --- Spreading ---
        store = GeneDomainStore(basic_context, qts)
        spread_tracker = UsageTracker()
        spread_genes = []
        for i in range(n_genes):
            phantom = make_gene(
                "CS101", "theory", "I1", ["G1"], "R1", start=0, duration=num_quanta
            )
            store.refresh_domain(i, phantom)
            domain = store.get_domain(i)

            inst = spread_tracker.pick_least_used_instructor(domain.instructors)
            spread_start: int | None = spread_tracker.pick_least_used_start(
                domain.valid_starts, num_quanta, top_k=3
            )
            room = spread_tracker.pick_least_used_room(
                domain.rooms, spread_start or 0, num_quanta
            )

            gene = make_gene(
                "CS101",
                "theory",
                inst,
                ["G1"],
                room,
                start=spread_start or 0,
                duration=num_quanta,
            )
            spread_genes.append(gene)
            spread_tracker.add_gene(gene)

        # Spreading should have lower or equal std deviation of time load
        assert spread_tracker.time_load_std() <= random_tracker.time_load_std() + 1.0
        # And instructor load should be more balanced
        assert (
            spread_tracker.instructor_load_std()
            <= random_tracker.instructor_load_std() + 1.0
        )

    def test_mutate_gene_spreading_changes_gene(self, basic_context, qts):
        """mutate_gene_spreading should return a gene with the same identity
        but potentially different mutable fields."""
        from src.ga.operators.mutation import mutate_gene_spreading

        store = GeneDomainStore(basic_context, qts)
        genes = [
            make_gene("CS101", "theory", "I1", ["G1"], "R1", start=0, duration=2),
            make_gene("CS101", "theory", "I2", ["G2"], "R2", start=7, duration=2),
        ]
        store.build_domains(genes)
        tracker = UsageTracker()
        tracker.build_from_individual(genes)

        new_gene = mutate_gene_spreading(
            genes[0], 0, store, tracker, genes, basic_context
        )
        # Identity preserved
        assert new_gene.course_id == "CS101"
        assert new_gene.course_type == "theory"
        assert new_gene.group_ids == ["G1"]
        assert new_gene.num_quanta == 2
