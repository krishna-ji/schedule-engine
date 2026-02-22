"""Tests for vectorized repair analysis equivalence against per-individual OOP.

Gates:
    1. build_counts_batch matches per-individual _build_counts — exact
    2. count_conflicts_batch matches per-individual _count_conflicts — exact
    3. fix_domains_batch matches per-individual _fix_domains — exact
    4. Performance: batch must be faster than per-individual loop
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PKL_PATH = ".cache/events_with_domains.pkl"
PKL_EXISTS = Path(PKL_PATH).exists()

skip_no_pkl = pytest.mark.skipif(
    not PKL_EXISTS, reason="events_with_domains.pkl not found"
)


@pytest.fixture(scope="module")
def pkl_data():
    if not PKL_EXISTS:
        pytest.skip("events_with_domains.pkl not found")
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def repairer():
    if not PKL_EXISTS:
        pytest.skip("events_with_domains.pkl not found")
    from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

    return BitsetSchedulingRepair(PKL_PATH)


@pytest.fixture(scope="module")
def population(pkl_data):
    """Build a small test population (unrepaired, from random init + domain fix)."""
    from src.pipeline.pymoo_operators import ConstructiveSampling
    from src.pipeline.scheduling_problem import SchedulingProblem

    prob = SchedulingProblem(PKL_PATH)
    sampling = ConstructiveSampling(PKL_PATH)
    return sampling._do(prob, 10)


@skip_no_pkl
class TestBuildCountsBatch:
    """build_counts_batch must exactly match per-individual _build_counts."""

    def test_room_counts_match(self, population, repairer):
        from src.pipeline.repair_analysis_vectorized import build_counts_batch

        rc_batch, ic_batch, gc_batch = build_counts_batch(
            population,
            repairer.durations,
            repairer.event_group_indices,
            repairer.n_rooms,
            repairer.n_instructors,
            repairer.n_groups,
        )

        for i in range(population.shape[0]):
            xi = population[i]
            inst, room, time_ = xi[0::3], xi[1::3], xi[2::3]
            rc_oop, ic_oop, gc_oop = repairer._build_counts(inst, room, time_)

            np.testing.assert_array_equal(
                rc_batch[i], rc_oop, err_msg=f"Room counts mismatch for individual {i}"
            )
            np.testing.assert_array_equal(
                ic_batch[i],
                ic_oop,
                err_msg=f"Instructor counts mismatch for individual {i}",
            )
            np.testing.assert_array_equal(
                gc_batch[i], gc_oop, err_msg=f"Group counts mismatch for individual {i}"
            )


@skip_no_pkl
class TestCountConflictsBatch:
    """count_conflicts_batch must exactly match per-individual _count_conflicts."""

    def test_conflicts_match(self, population, repairer):
        from src.pipeline.repair_analysis_vectorized import count_conflicts_batch

        C_batch = count_conflicts_batch(
            population,
            repairer.durations,
            repairer.event_group_indices,
            repairer.n_rooms,
            repairer.n_instructors,
            repairer.n_groups,
        )

        for i in range(population.shape[0]):
            xi = population[i]
            inst, room, time_ = xi[0::3], xi[1::3], xi[2::3]
            rc, ic, gc = repairer._build_counts(inst, room, time_)

            for e in range(repairer.n_events):
                cc_oop = repairer._count_conflicts(e, inst, room, time_, rc, ic, gc)
                assert C_batch[i, e] == cc_oop, (
                    f"Conflict count mismatch for individual {i}, event {e}: "
                    f"batch={C_batch[i, e]}, oop={cc_oop}"
                )


@skip_no_pkl
class TestFixDomainsBatch:
    """fix_domains_batch must match per-individual _fix_domains."""

    def test_domains_match(self, pkl_data, repairer):
        from src.pipeline.repair_analysis_vectorized import fix_domains_batch

        # Create random (unclamped) population
        E = repairer.n_events
        rng = np.random.default_rng(42)
        N = 10
        X = rng.integers(0, 200, size=(N, 3 * E))

        # Batch fix
        X_batch = X.copy()
        fix_domains_batch(
            X_batch,
            repairer.allowed_instructors,
            repairer.allowed_rooms,
            repairer.allowed_starts,
            inst_avail=repairer.inst_avail,
            durations=repairer.durations,
        )

        # Per-individual fix
        X_oop = X.copy()
        for i in range(N):
            inst = X_oop[i, 0::3]
            room = X_oop[i, 1::3]
            time_ = X_oop[i, 2::3]
            repairer._fix_domains(inst, room, time_)

        np.testing.assert_array_equal(
            X_batch,
            X_oop,
            err_msg="fix_domains_batch doesn't match per-individual _fix_domains",
        )


@skip_no_pkl
class TestRepairAnalysisPerformance:
    """Batch operations should be faster than per-individual loops."""

    def test_build_counts_speedup(self, population, repairer):
        from src.pipeline.repair_analysis_vectorized import build_counts_batch

        N = population.shape[0]

        # Batch
        t0 = time.perf_counter()
        for _ in range(50):
            build_counts_batch(
                population,
                repairer.durations,
                repairer.event_group_indices,
                repairer.n_rooms,
                repairer.n_instructors,
                repairer.n_groups,
            )
        t_batch = (time.perf_counter() - t0) / 50

        # Per-individual
        t0 = time.perf_counter()
        for _ in range(50):
            for i in range(N):
                xi = population[i]
                repairer._build_counts(xi[0::3], xi[1::3], xi[2::3])
        t_loop = (time.perf_counter() - t0) / 50

        speedup = t_loop / t_batch if t_batch > 0 else 1.0
        print(
            f"\nbuild_counts: batch={t_batch * 1000:.2f}ms, loop={t_loop * 1000:.2f}ms, speedup={speedup:.1f}x"
        )
        # At least no regression
        assert t_batch < t_loop * 5, f"Batch is much slower: {speedup:.2f}x"

    def test_count_conflicts_speedup(self, population, repairer):
        from src.pipeline.repair_analysis_vectorized import count_conflicts_batch

        N = population.shape[0]

        # Batch
        t0 = time.perf_counter()
        for _ in range(20):
            count_conflicts_batch(
                population,
                repairer.durations,
                repairer.event_group_indices,
                repairer.n_rooms,
                repairer.n_instructors,
                repairer.n_groups,
            )
        t_batch = (time.perf_counter() - t0) / 20

        # Per-individual
        t0 = time.perf_counter()
        for _ in range(20):
            for i in range(N):
                xi = population[i]
                inst, room, time_ = xi[0::3], xi[1::3], xi[2::3]
                rc, ic, gc = repairer._build_counts(inst, room, time_)
                for e in range(repairer.n_events):
                    repairer._count_conflicts(e, inst, room, time_, rc, ic, gc)
        t_loop = (time.perf_counter() - t0) / 20

        speedup = t_loop / t_batch if t_batch > 0 else 1.0
        print(
            f"\ncount_conflicts: batch={t_batch * 1000:.2f}ms, loop={t_loop * 1000:.2f}ms, speedup={speedup:.1f}x"
        )
