"""Tests for DataStore: the single source of truth for data loading."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── DataStore.from_json ──────────────────────────────────────────────


class TestDataStoreFromJson:
    """Integration tests loading real data from data/."""

    @pytest.fixture(scope="class")
    def store(self):
        from src.io.data_store import DataStore

        return DataStore.from_json(DATA_DIR, run_preflight=False)

    def test_courses_loaded(self, store):
        assert len(store.courses) > 0, "No courses loaded"

    def test_groups_loaded(self, store):
        assert len(store.groups) > 0, "No groups loaded"

    def test_instructors_loaded(self, store):
        assert len(store.instructors) > 0, "No instructors loaded"

    def test_rooms_loaded(self, store):
        assert len(store.rooms) > 0, "No rooms loaded"

    def test_qts_created(self, store):
        assert store.qts is not None
        assert store.qts.total_quanta > 0

    def test_cohort_pairs_derived(self, store):
        assert isinstance(store.cohort_pairs, list)
        # With real data, should have some pairs
        assert len(store.cohort_pairs) > 0, "No cohort pairs derived"
        # Each pair is a 2-tuple of strings
        for pair in store.cohort_pairs:
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)

    def test_available_quanta(self, store):
        quanta = store.available_quanta
        assert len(quanta) > 0
        assert all(isinstance(q, int) for q in quanta)

    def test_summary(self, store):
        s = store.summary()
        assert "Courses:" in s
        assert "Groups:" in s
        assert "Instructors:" in s
        assert "Rooms:" in s
        assert "Quanta:" in s


# ── DataStore.to_context ─────────────────────────────────────────────


class TestDataStoreToContext:
    """Backward compatibility: SchedulingContext from DataStore."""

    @pytest.fixture(scope="class")
    def store(self):
        from src.io.data_store import DataStore

        return DataStore.from_json(DATA_DIR, run_preflight=False)

    def test_context_type(self, store):
        from src.domain.types import SchedulingContext

        ctx = store.to_context()
        assert isinstance(ctx, SchedulingContext)

    def test_context_has_all_fields(self, store):
        ctx = store.to_context()
        assert ctx.courses is store.courses
        assert ctx.groups is store.groups
        assert ctx.instructors is store.instructors
        assert ctx.rooms is store.rooms
        assert ctx.cohort_pairs == store.cohort_pairs
        assert len(ctx.available_quanta) == len(store.available_quanta)


# ── DataStore serialization ──────────────────────────────────────────


class TestDataStoreSerialization:
    """to_dict/from_dict round-trip for multiprocessing."""

    @pytest.fixture(scope="class")
    def store(self):
        from src.io.data_store import DataStore

        return DataStore.from_json(DATA_DIR, run_preflight=False)

    def test_round_trip(self, store):
        from src.io.data_store import DataStore

        d = store.to_dict()
        store2 = DataStore.from_dict(d)
        assert len(store2.courses) == len(store.courses)
        assert len(store2.groups) == len(store.groups)
        assert len(store2.instructors) == len(store.instructors)
        assert len(store2.rooms) == len(store.rooms)
        assert store2.cohort_pairs == store.cohort_pairs

    def test_dict_keys(self, store):
        d = store.to_dict()
        assert "courses" in d
        assert "groups" in d
        assert "instructors" in d
        assert "rooms" in d
        assert "qts" in d
        assert "cohort_pairs" in d


# ── _merge_cohort_pairs ─────────────────────────────────────────────


class TestMergeCohortPairs:
    """Unit tests for cohort pair deduplication."""

    def test_empty_inputs(self):
        from src.io.data_store import _merge_cohort_pairs

        assert _merge_cohort_pairs([], []) == []

    def test_dedup_same_pair(self):
        from src.io.data_store import _merge_cohort_pairs

        pairs = [("A", "B"), ("A", "B")]
        result = _merge_cohort_pairs(pairs, [])
        assert len(result) == 1

    def test_dedup_reversed_pair(self):
        from src.io.data_store import _merge_cohort_pairs

        result = _merge_cohort_pairs([("A", "B")], [("B", "A")])
        assert len(result) == 1

    def test_case_insensitive_dedup(self):
        from src.io.data_store import _merge_cohort_pairs

        result = _merge_cohort_pairs([("Grp-A", "Grp-B")], [("grp-a", "grp-b")])
        assert len(result) == 1

    def test_strips_whitespace(self):
        from src.io.data_store import _merge_cohort_pairs

        result = _merge_cohort_pairs([("  A ", " B  ")], [])
        assert result == [("A", "B")]

    def test_skips_empty_strings(self):
        from src.io.data_store import _merge_cohort_pairs

        result = _merge_cohort_pairs([("A", ""), ("", "B")], [])
        assert result == []

    def test_merges_derived_and_configured(self):
        from src.io.data_store import _merge_cohort_pairs

        result = _merge_cohort_pairs([("A", "B")], [("C", "D")])
        assert len(result) == 2
