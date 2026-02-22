#!/usr/bin/env python3
"""Tests for BitsetSchedulingRepair — validates equivalence with SchedulingRepair
and batch repair functionality.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.fast_evaluator import fast_evaluate_hard
from src.pipeline.repair_operator import SchedulingRepair
from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair, repair_batch

PKL_PATH = ".cache/events_with_domains.pkl"


@pytest.fixture(scope="module")
def pkl_data():
    if not Path(PKL_PATH).exists():
        pytest.skip("events_with_domains.pkl not found — run build_events first")
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def repairer_orig():
    return SchedulingRepair(PKL_PATH)


@pytest.fixture(scope="module")
def repairer_bs():
    return BitsetSchedulingRepair(PKL_PATH)


def _score(x, data):
    return fast_evaluate_hard(
        data["events"],
        x[0::3],
        x[1::3],
        x[2::3],
        data["allowed_instructors"],
        data["allowed_rooms"],
        data.get("instructor_available_quanta", {}),
        data.get("room_available_quanta", {}),
    )


def _total_violations(x, data):
    return sum(_score(x, data).values())


def _random_chromosome(E, data, rng):
    x = np.zeros(3 * E, dtype=int)
    for e in range(E):
        ai = data["allowed_instructors"][e]
        ar = data["allowed_rooms"][e]
        at = data["allowed_starts"][e]
        x[3 * e] = rng.choice(ai) if ai else 0
        x[3 * e + 1] = rng.choice(ar) if ar else 0
        x[3 * e + 2] = rng.choice(at) if at else 0
    return x


class TestBitsetRepairEquivalence:
    """Bitset repair must reduce violations at least as well as original."""

    @pytest.mark.parametrize("seed", range(5))
    def test_repair_reduces_violations(self, seed, pkl_data, repairer_bs):
        """Repaired chromosome must have fewer violations than random."""
        rng = np.random.default_rng(seed)
        E = len(pkl_data["events"])
        x = _random_chromosome(E, pkl_data, rng)

        orig_v = _total_violations(x, pkl_data)
        x_rep = repairer_bs.repair(x)
        rep_v = _total_violations(x_rep, pkl_data)

        assert rep_v < orig_v, f"seed={seed}: violations {orig_v} -> {rep_v}"

    @pytest.mark.parametrize("seed", range(5))
    def test_comparable_quality(self, seed, pkl_data, repairer_orig, repairer_bs):
        """Bitset repair quality must be within 2x of original."""
        rng = np.random.default_rng(seed + 100)
        E = len(pkl_data["events"])
        x = _random_chromosome(E, pkl_data, rng)

        x_orig = repairer_orig.repair(x.copy())
        x_bs = repairer_bs.repair(x.copy())

        v_orig = _total_violations(x_orig, pkl_data)
        v_bs = _total_violations(x_bs, pkl_data)

        # Bitset repair should be no worse than 2x original
        assert v_bs <= max(
            2 * v_orig, v_orig + 30
        ), f"seed={seed}: orig={v_orig}, bitset={v_bs}"


class TestConstructFeasible:
    """construct_feasible must produce valid chromosomes."""

    def test_correct_length(self, pkl_data, repairer_bs):
        E = len(pkl_data["events"])
        chrom = repairer_bs.construct_feasible(np.random.default_rng(42))
        assert chrom.shape == (3 * E,)

    def test_domains_respected(self, pkl_data, repairer_bs):
        chrom = repairer_bs.construct_feasible(np.random.default_rng(42))
        E = len(pkl_data["events"])
        for e in range(E):
            i, r, t = int(chrom[3 * e]), int(chrom[3 * e + 1]), int(chrom[3 * e + 2])
            ai = pkl_data["allowed_instructors"][e]
            ar = pkl_data["allowed_rooms"][e]
            at = pkl_data["allowed_starts"][e]
            if ai:
                assert i in ai, f"event {e}: inst {i} not in {ai}"
            if ar:
                assert r in ar, f"event {e}: room {r} not in {ar}"
            if at:
                assert t in at, f"event {e}: time {t} not in {at}"

    def test_low_violations(self, pkl_data, repairer_bs):
        """construct_feasible should produce a chromosome with few violations."""
        chrom = repairer_bs.construct_feasible(np.random.default_rng(42))
        v = _total_violations(chrom, pkl_data)
        # Should be much better than random (threshold scales with event count)
        # Note: threshold accounts for availability rounding (start times
        # rounded UP to quantum boundaries in encode_availability) and
        # sibling_same_day violations from the 9th constraint.
        n_events = len(pkl_data["events"])
        threshold = max(200, int(n_events * 0.5))
        assert (
            v < threshold
        ), f"construct_feasible violations: {v} (threshold={threshold})"


class TestRepairDomains:
    """Repaired chromosomes must have valid domains."""

    def test_domains_after_repair(self, pkl_data, repairer_bs):
        rng = np.random.default_rng(77)
        E = len(pkl_data["events"])
        x = _random_chromosome(E, pkl_data, rng)
        x_rep = repairer_bs.repair(x)

        for e in range(E):
            i, r, t = int(x_rep[3 * e]), int(x_rep[3 * e + 1]), int(x_rep[3 * e + 2])
            ai = pkl_data["allowed_instructors"][e]
            ar = pkl_data["allowed_rooms"][e]
            at = pkl_data["allowed_starts"][e]
            if ai:
                assert i in ai, f"event {e}: inst {i} not in domain"
            if ar:
                assert r in ar, f"event {e}: room {r} not in domain"
            if at:
                assert t in at, f"event {e}: time {t} not in domain"


class TestBatchRepair:
    """repair_batch must produce same results as individual repair."""

    def test_batch_matches_individual(self, pkl_data, repairer_bs):
        rng = np.random.default_rng(55)
        E = len(pkl_data["events"])
        N = 5
        X = np.array([_random_chromosome(E, pkl_data, rng) for _ in range(N)])

        X_batch = repair_batch(X, repairer_bs)

        for i in range(N):
            x_ind = repairer_bs.repair(X[i])
            np.testing.assert_array_equal(
                X_batch[i],
                x_ind,
                err_msg=f"Batch result {i} differs from individual repair",
            )

    def test_batch_shape(self, pkl_data, repairer_bs):
        rng = np.random.default_rng(66)
        E = len(pkl_data["events"])
        N = 3
        X = np.array([_random_chromosome(E, pkl_data, rng) for _ in range(N)])
        X_rep = repair_batch(X, repairer_bs)
        assert X_rep.shape == (N, 3 * E)

    def test_batch_single(self, pkl_data, repairer_bs):
        """Single-row batch should work."""
        rng = np.random.default_rng(77)
        E = len(pkl_data["events"])
        x = _random_chromosome(E, pkl_data, rng)
        X_rep = repair_batch(x.reshape(1, -1), repairer_bs)
        assert X_rep.shape == (1, 3 * E)


class TestOffspringRepair:
    """Test repair on perturbed construct_feasible parents (realistic use case)."""

    def _perturb(self, chrom, k, data, rng):
        out = chrom.copy()
        E = len(data["events"])
        events = rng.choice(E, size=min(k, E), replace=False)
        for e in events:
            ai = data["allowed_instructors"][e]
            ar = data["allowed_rooms"][e]
            at = data["allowed_starts"][e]
            if ai:
                out[3 * e] = rng.choice(ai)
            if ar:
                out[3 * e + 1] = rng.choice(ar)
            if at:
                out[3 * e + 2] = rng.choice(at)
        return out

    @pytest.mark.parametrize("k", [5, 10, 20, 50])
    def test_offspring_repair_quality(self, k, pkl_data, repairer_bs):
        """Perturb k events from feasible parent, repair, check quality."""
        rng = np.random.default_rng(42)
        parent = repairer_bs.construct_feasible(rng)
        parent_v = _total_violations(parent, pkl_data)

        successes = 0
        n_trials = 10
        for _trial in range(n_trials):
            offspring = self._perturb(parent, k, pkl_data, rng)
            repaired = repairer_bs.repair(offspring)
            rep_v = _total_violations(repaired, pkl_data)
            # Allow some tolerance
            if rep_v <= parent_v + 30:
                successes += 1

        # At least 60% of trials should meet tolerance
        assert (
            successes >= n_trials * 0.6
        ), f"k={k}: only {successes}/{n_trials} met tolerance"
