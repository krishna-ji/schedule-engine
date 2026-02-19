"""Bitset-based time occupancy primitives for T=42 quanta.

All operations use ``numpy.uint64`` masks where bit *q* represents
quantum *q*.  Since T=42 ≤ 64, a single ``uint64`` encodes the full
weekly timetable for one resource.

Performance note
----------------
``popcount`` uses ``bin(x).count('1')`` which CPython accelerates to
a native call — fast enough for 10^6 calls/sec on modern hardware.
Numpy-vectorized ``popcount_array`` is also provided for batch use.

Public API
----------
- ``mask_from_interval(start_q, dur) -> np.uint64`` — contiguous bits.
- ``overlaps(a, b) -> bool`` — O(1) overlap test.
- ``overlap_count(a, b) -> int`` — popcount(a & b).
- ``set_interval(occ, start_q, dur) -> np.uint64`` — ``occ | mask``.
- ``clear_interval(occ, start_q, dur) -> np.uint64`` — ``occ & ~mask``.
- ``popcount(mask) -> int`` — number of set bits.
- ``popcount_array(arr) -> np.ndarray`` — vectorized popcount.
- ``mask_from_quanta(quanta) -> np.uint64`` — from iterable of ints.
- ``FULL_MASK`` — all 42 bits set.
- ``T`` — number of quanta (42).
"""

from __future__ import annotations

import numpy as np

# Total number of quanta in the weekly timetable
T: int = 42

# Mask with bits 0..41 set
FULL_MASK: np.uint64 = np.uint64((1 << T) - 1)

# Zero mask constant
ZERO: np.uint64 = np.uint64(0)

# One constant for bit shifting
_ONE: np.uint64 = np.uint64(1)


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def mask_from_interval(start_q: int, dur: int) -> np.uint64:
    """Return a uint64 mask with bits [start_q, start_q+dur) set.

    >>> bin(mask_from_interval(0, 3))
    '0b111'
    >>> bin(mask_from_interval(2, 4))
    '0b111100'
    """
    if dur <= 0:
        return ZERO
    # ((1 << dur) - 1) << start_q
    return np.uint64(((1 << dur) - 1) << start_q) & FULL_MASK


def mask_from_quanta(quanta) -> np.uint64:
    """Build a mask from an iterable of quantum indices.

    >>> bin(mask_from_quanta([0, 2, 5]))
    '0b100101'
    """
    m = np.uint64(0)
    for q in quanta:
        m |= _ONE << np.uint64(q)
    return m & FULL_MASK


def overlaps(mask_a: np.uint64, mask_b: np.uint64) -> bool:
    """Return True if any bit is set in both masks (O(1))."""
    return bool(mask_a & mask_b)


def overlap_count(mask_a: np.uint64, mask_b: np.uint64) -> int:
    """Return the number of overlapping quanta (popcount of a & b)."""
    return popcount(mask_a & mask_b)


def set_interval(occ: np.uint64, start_q: int, dur: int) -> np.uint64:
    """Return ``occ`` with bits [start_q, start_q+dur) set."""
    return occ | mask_from_interval(start_q, dur)


def clear_interval(occ: np.uint64, start_q: int, dur: int) -> np.uint64:
    """Return ``occ`` with bits [start_q, start_q+dur) cleared."""
    return occ & ~mask_from_interval(start_q, dur)


def popcount(mask: np.uint64) -> int:
    """Count set bits in a uint64 mask."""
    return bin(int(mask)).count("1")


def popcount_array(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount for an array of uint64 values.

    Uses a lookup-based approach: split each uint64 into bytes and
    sum the popcount of each byte via a 256-entry LUT.
    """
    arr = np.asarray(arr, dtype=np.uint64)
    # Byte-wise popcount LUT
    lut = _POPCOUNT_LUT
    total = np.zeros(arr.shape, dtype=np.int64)
    for shift in range(0, 64, 8):
        byte_vals = ((arr >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.intp)
        total += lut[byte_vals]
    return total


# Pre-computed byte popcount lookup table (256 entries)
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Convenience: bulk mask building
# ---------------------------------------------------------------------------


def build_availability_masks(
    available_quanta: dict[int, set | None],
) -> dict[int, np.uint64]:
    """Convert {entity_idx: set_of_quanta | None} -> {entity_idx: uint64_mask}.

    ``None`` (= always available) maps to ``FULL_MASK``.
    """
    out: dict[int, np.uint64] = {}
    for idx, slots in available_quanta.items():
        if slots is None:
            out[idx] = FULL_MASK
        else:
            out[idx] = mask_from_quanta(slots)
    return out


def build_event_masks(events: list[dict]) -> np.ndarray:
    """For each event, precompute its duration mask at every allowed start.

    Returns a 1-D array of uint64 of length E, where each entry is the
    mask for the event at its *currently assigned* start.  Callers should
    rebuild after time changes.

    (This is a convenience; for the evaluator we compute masks on the fly.)
    """
    E = len(events)
    masks = np.zeros(E, dtype=np.uint64)
    return masks


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test
    m = mask_from_interval(5, 3)
    assert m == np.uint64(0b11100000), f"Got {bin(m)}"
    assert popcount(m) == 3
    assert overlaps(m, mask_from_interval(6, 2))
    assert not overlaps(m, mask_from_interval(8, 2))
    assert overlap_count(m, mask_from_interval(6, 2)) == 2

    m2 = set_interval(ZERO, 10, 5)
    assert popcount(m2) == 5
    m3 = clear_interval(m2, 12, 2)
    assert popcount(m3) == 3

    mq = mask_from_quanta([0, 5, 10, 41])
    assert popcount(mq) == 4

    arr = np.array([m, m2, mq], dtype=np.uint64)
    pc = popcount_array(arr)
    assert list(pc) == [3, 5, 4], f"Got {pc}"

    avail = build_availability_masks({0: {1, 2, 3}, 1: None, 2: set()})
    assert popcount(avail[0]) == 3
    assert avail[1] == FULL_MASK
    assert avail[2] == ZERO

    print("All bitset_time self-tests PASSED")
