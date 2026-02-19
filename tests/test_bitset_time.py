"""Unit tests for bitset_time.py — T=42 quanta bitset primitives."""

from __future__ import annotations

import numpy as np

from src.pipeline.bitset_time import (
    FULL_MASK,
    ZERO,
    T,
    build_availability_masks,
    clear_interval,
    mask_from_interval,
    mask_from_quanta,
    overlap_count,
    overlaps,
    popcount,
    popcount_array,
    set_interval,
)


class TestMaskFromInterval:
    """Tests for mask_from_interval()."""

    def test_basic_interval(self):
        m = mask_from_interval(0, 3)
        assert m == np.uint64(0b111)
        assert popcount(m) == 3

    def test_offset_interval(self):
        m = mask_from_interval(5, 3)
        # bits 5,6,7 set
        assert popcount(m) == 3
        assert bool(m & (np.uint64(1) << np.uint64(5)))
        assert bool(m & (np.uint64(1) << np.uint64(6)))
        assert bool(m & (np.uint64(1) << np.uint64(7)))
        assert not bool(m & (np.uint64(1) << np.uint64(4)))
        assert not bool(m & (np.uint64(1) << np.uint64(8)))

    def test_single_quantum(self):
        m = mask_from_interval(10, 1)
        assert popcount(m) == 1
        assert bool(m & (np.uint64(1) << np.uint64(10)))

    def test_full_week(self):
        m = mask_from_interval(0, T)
        assert m == FULL_MASK
        assert popcount(m) == T

    def test_zero_duration(self):
        m = mask_from_interval(5, 0)
        assert m == ZERO

    def test_negative_duration(self):
        m = mask_from_interval(5, -1)
        assert m == ZERO

    def test_last_quantum(self):
        m = mask_from_interval(T - 1, 1)
        assert popcount(m) == 1

    def test_boundary_clamp(self):
        # Duration exceeding T should be masked by FULL_MASK
        m = mask_from_interval(40, 5)
        # bits 40, 41 are valid (T=42); bits 42-44 are masked out
        assert popcount(m) == 2


class TestMaskFromQuanta:
    """Tests for mask_from_quanta()."""

    def test_empty(self):
        assert mask_from_quanta([]) == ZERO

    def test_single(self):
        m = mask_from_quanta([7])
        assert popcount(m) == 1
        assert bool(m & (np.uint64(1) << np.uint64(7)))

    def test_multiple(self):
        m = mask_from_quanta([0, 5, 10, 41])
        assert popcount(m) == 4

    def test_set_input(self):
        m = mask_from_quanta({3, 7, 15})
        assert popcount(m) == 3

    def test_equivalent_to_interval(self):
        """mask_from_quanta of a range should equal mask_from_interval."""
        m1 = mask_from_quanta(range(5, 10))
        m2 = mask_from_interval(5, 5)
        assert m1 == m2


class TestOverlaps:
    """Tests for overlaps() and overlap_count()."""

    def test_no_overlap(self):
        a = mask_from_interval(0, 3)
        b = mask_from_interval(5, 3)
        assert not overlaps(a, b)
        assert overlap_count(a, b) == 0

    def test_partial_overlap(self):
        a = mask_from_interval(0, 5)
        b = mask_from_interval(3, 5)
        assert overlaps(a, b)
        assert overlap_count(a, b) == 2  # quanta 3, 4

    def test_full_overlap(self):
        a = mask_from_interval(2, 3)
        b = mask_from_interval(2, 3)
        assert overlaps(a, b)
        assert overlap_count(a, b) == 3

    def test_subset_overlap(self):
        a = mask_from_interval(0, 10)
        b = mask_from_interval(3, 2)
        assert overlaps(a, b)
        assert overlap_count(a, b) == 2

    def test_adjacent_no_overlap(self):
        a = mask_from_interval(0, 5)
        b = mask_from_interval(5, 5)
        assert not overlaps(a, b)


class TestSetClearInterval:
    """Tests for set_interval() and clear_interval()."""

    def test_set_from_zero(self):
        m = set_interval(ZERO, 3, 4)
        assert popcount(m) == 4

    def test_set_idempotent(self):
        m = set_interval(ZERO, 3, 4)
        m2 = set_interval(m, 3, 4)
        assert m == m2

    def test_set_union(self):
        m = set_interval(ZERO, 0, 3)
        m = set_interval(m, 5, 3)
        assert popcount(m) == 6

    def test_clear_removes_bits(self):
        m = mask_from_interval(0, 10)
        m = clear_interval(m, 3, 4)
        assert popcount(m) == 6  # 10 - 4

    def test_clear_noop_if_not_set(self):
        m = mask_from_interval(0, 3)
        m2 = clear_interval(m, 10, 5)
        assert m == m2

    def test_set_then_clear_roundtrip(self):
        m = set_interval(ZERO, 5, 7)
        m = clear_interval(m, 5, 7)
        assert m == ZERO


class TestPopcount:
    """Tests for popcount() and popcount_array()."""

    def test_zero(self):
        assert popcount(ZERO) == 0

    def test_full(self):
        assert popcount(FULL_MASK) == T

    def test_known_value(self):
        assert popcount(np.uint64(0b10101010)) == 4

    def test_array_basic(self):
        arr = np.array([ZERO, FULL_MASK, mask_from_interval(0, 5)], dtype=np.uint64)
        result = popcount_array(arr)
        assert list(result) == [0, T, 5]

    def test_array_matches_scalar(self):
        rng = np.random.default_rng(42)
        vals = rng.integers(0, int(FULL_MASK), size=100).astype(np.uint64)
        batch = popcount_array(vals)
        for i, v in enumerate(vals):
            assert batch[i] == popcount(v), f"Mismatch at index {i}"


class TestBuildAvailabilityMasks:
    """Tests for build_availability_masks()."""

    def test_none_means_full(self):
        masks = build_availability_masks({0: None})
        assert masks[0] == FULL_MASK

    def test_empty_set_means_zero(self):
        masks = build_availability_masks({0: set()})
        assert masks[0] == ZERO

    def test_specific_quanta(self):
        masks = build_availability_masks({0: {1, 5, 10}})
        assert popcount(masks[0]) == 3

    def test_multiple_entities(self):
        masks = build_availability_masks(
            {
                0: None,
                1: {0, 1, 2},
                2: set(range(T)),
            }
        )
        assert masks[0] == FULL_MASK
        assert popcount(masks[1]) == 3
        assert masks[2] == FULL_MASK


class TestConstants:
    """Tests for module-level constants."""

    def test_T_is_42(self):
        assert T == 42

    def test_full_mask_popcount(self):
        assert popcount(FULL_MASK) == 42

    def test_zero_is_zero(self):
        assert np.uint64(0) == ZERO

    def test_full_mask_value(self):
        assert int(FULL_MASK) == (1 << 42) - 1
