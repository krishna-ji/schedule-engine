"""Canonical 3×E interleaved encoding for the scheduling chromosome.

Chromosome layout::

    X = [I0, R0, T0, I1, R1, T1, ..., I_{E-1}, R_{E-1}, T_{E-1}]

Where:
    X[3e + 0] = instructor_idx   for event e
    X[3e + 1] = room_idx         for event e
    X[3e + 2] = start_quanta     for event e

All values are **indices** (not IDs).  Mapping to/from string IDs is done
via the ``instructor_to_idx`` / ``room_to_idx`` dicts stored in the pkl.

Gene domains are event-dependent::

    instructor_idx ∈ allowed_instructors[e]
    room_idx       ∈ allowed_rooms[e]
    start_quanta   ∈ allowed_starts[e]

The structural fields (course_id, course_type, group_ids, num_quanta) are
fixed per event and stored in ``events[e]``.

Schema version and data hash are in the pkl for integrity checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class GeneAssignment(NamedTuple):
    """Assignment for a single event."""

    instructor_idx: int
    room_idx: int
    start_quanta: int


@dataclass(frozen=True, slots=True)
class EncodingSpec:
    """Encoding specification derived from the events pkl data."""

    n_events: int
    n_vars: int  # = 3 * n_events
    allowed_instructors: list[list[int]]
    allowed_rooms: list[list[int]]
    allowed_starts: list[list[int]]

    @classmethod
    def from_pkl_data(cls, pkl_data: dict) -> EncodingSpec:
        """Create from loaded events_with_domains.pkl dict."""
        n = len(pkl_data["events"])
        return cls(
            n_events=n,
            n_vars=3 * n,
            allowed_instructors=pkl_data["allowed_instructors"],
            allowed_rooms=pkl_data["allowed_rooms"],
            allowed_starts=pkl_data["allowed_starts"],
        )

    def xl(self) -> np.ndarray:
        """Lower bounds for pymoo Problem (all zeros)."""
        return np.zeros(self.n_vars, dtype=int)

    def xu(self) -> np.ndarray:
        """Upper bounds for pymoo Problem.

        Note: these are *max index* bounds, not domain-enforced.
        Domain enforcement is via repair/sampling.
        """
        xu = np.zeros(self.n_vars, dtype=int)
        for e in range(self.n_events):
            xu[3 * e + 0] = (
                max(self.allowed_instructors[e]) if self.allowed_instructors[e] else 0
            )
            xu[3 * e + 1] = max(self.allowed_rooms[e]) if self.allowed_rooms[e] else 0
            xu[3 * e + 2] = max(self.allowed_starts[e]) if self.allowed_starts[e] else 0
        return xu


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------


def encode(genes: list[GeneAssignment]) -> np.ndarray:
    """Pack a list of GeneAssignment into a flat 3×E chromosome.

    Args:
        genes: List of E GeneAssignment tuples.

    Returns:
        numpy int array of length 3E in interleaved order.
    """
    E = len(genes)
    x = np.zeros(3 * E, dtype=int)
    for e, g in enumerate(genes):
        x[3 * e + 0] = g.instructor_idx
        x[3 * e + 1] = g.room_idx
        x[3 * e + 2] = g.start_quanta
    return x


def decode(x: np.ndarray) -> list[GeneAssignment]:
    """Unpack a flat 3×E chromosome into a list of GeneAssignment.

    Args:
        x: numpy int array of length 3E.

    Returns:
        List of E GeneAssignment tuples.
    """
    assert x.ndim == 1 and len(x) % 3 == 0, (
        f"Expected 1-D array with len%3==0, got shape={x.shape}"
    )
    E = len(x) // 3
    return [
        GeneAssignment(
            instructor_idx=int(x[3 * e + 0]),
            room_idx=int(x[3 * e + 1]),
            start_quanta=int(x[3 * e + 2]),
        )
        for e in range(E)
    ]


def chromosome_views(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (instructor, room, time) views into interleaved chromosome.

    These are views (not copies), so modifying them modifies x.
    """
    return x[0::3], x[1::3], x[2::3]


# ---------------------------------------------------------------------------
# Roundtrip sanity
# ---------------------------------------------------------------------------


def _roundtrip_test() -> None:
    """Verify encode/decode roundtrip. Called by test suite."""
    import random

    E = 100
    genes = [
        GeneAssignment(
            instructor_idx=random.randint(0, 50),
            room_idx=random.randint(0, 30),
            start_quanta=random.randint(0, 40),
        )
        for _ in range(E)
    ]
    x = encode(genes)
    assert x.shape == (3 * E,)
    decoded = decode(x)
    assert decoded == genes, "Roundtrip failed: decoded != original"

    # Also check views
    inst, room, time = chromosome_views(x)
    for e, g in enumerate(genes):
        assert inst[e] == g.instructor_idx
        assert room[e] == g.room_idx
        assert time[e] == g.start_quanta

    print("Roundtrip test PASSED")


if __name__ == "__main__":
    _roundtrip_test()
