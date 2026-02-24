#!/usr/bin/env python3
"""Tests for parallel repair infrastructure in ga_experiment.py.

Validates:
1. _repair_single_elite is importable and picklable (ProcessPoolExecutor needs this)
2. _repair_single_elite produces valid repaired chromosomes
3. Parallel dispatch matches sequential results (deterministic seeds)
4. MemeticExperiment callback integrates repair_frequency + parallelism
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PKL_PATH = PROJECT_ROOT / ".cache" / "events_with_domains.pkl"
PKL_EXISTS = PKL_PATH.exists()

SKIP_MSG = "events_with_domains.pkl not found — run build_events first"


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pkl_data():
    if not PKL_EXISTS:
        pytest.skip(SKIP_MSG)
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def random_chromosome(pkl_data):
    """Build a random chromosome for testing."""
    events = pkl_data["events"]
    allowed_instructors = pkl_data["allowed_instructors"]
    allowed_rooms = pkl_data["allowed_rooms"]
    allowed_starts = pkl_data["allowed_starts"]
    rng = np.random.default_rng(42)
    E = len(events)
    X = np.zeros(3 * E, dtype=int)
    inst, room, time = X[0::3], X[1::3], X[2::3]
    for e in range(E):
        ai = allowed_instructors[e]
        ar = allowed_rooms[e]
        at = allowed_starts[e]
        inst[e] = rng.choice(ai) if ai else 0
        room[e] = rng.choice(ar) if ar else 0
        time[e] = rng.choice(at) if at else 0
    return X


# ─── Test 1: Import & Picklability ───────────────────────────────────


class TestWorkerFunction:
    """Test _repair_single_elite is importable and picklable."""

    def test_importable(self):
        from src.experiments.ga_experiment import _repair_single_elite

        assert callable(_repair_single_elite)

    def test_picklable(self):
        """ProcessPoolExecutor requires the target function to be picklable."""
        from src.experiments.ga_experiment import _repair_single_elite

        pickled = pickle.dumps(_repair_single_elite)
        restored = pickle.loads(pickled)
        assert callable(restored)
        assert restored.__name__ == "_repair_single_elite"


# ─── Test 2: Worker produces valid output ────────────────────────────


@pytest.mark.skipif(not PKL_EXISTS, reason=SKIP_MSG)
class TestWorkerOutput:
    """Test _repair_single_elite produces valid repaired chromosomes."""

    def test_output_shape(self, random_chromosome):
        from src.experiments.ga_experiment import _repair_single_elite

        X = random_chromosome.copy()
        result = _repair_single_elite(X, str(PKL_PATH), 2, 1, 0)
        assert result.shape == X.shape
        assert result.dtype == X.dtype

    def test_reduces_conflicts(self, random_chromosome, pkl_data):
        """Repair should reduce or maintain hard violations."""
        from src.experiments.ga_experiment import _repair_single_elite
        from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

        repairer = BitsetSchedulingRepair(str(PKL_PATH))
        X = random_chromosome.copy()

        def _count_conflicts(x):
            """Count total conflicts using the repairer's internal machinery."""
            inst, room, time = x[0::3], x[1::3], x[2::3]
            rc, ic, gc = repairer._build_counts(inst, room, time)
            total = 0
            for e in range(repairer.n_events):
                total += repairer._count_conflicts(e, inst, room, time, rc, ic, gc)
            return total

        before = _count_conflicts(X)
        result = _repair_single_elite(X, str(PKL_PATH), 4, 1, 0)
        after = _count_conflicts(result)
        assert after <= before, f"Repair worsened: {before} -> {after}"

    def test_deterministic_with_same_seed(self, random_chromosome):
        """Same (gen, idx) seed → identical output."""
        from src.experiments.ga_experiment import _repair_single_elite

        X = random_chromosome.copy()
        r1 = _repair_single_elite(X.copy(), str(PKL_PATH), 4, gen=10, idx=3)
        r2 = _repair_single_elite(X.copy(), str(PKL_PATH), 4, gen=10, idx=3)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self, random_chromosome):
        """Different (gen, idx) seeds → different outputs (with high probability)."""
        from src.experiments.ga_experiment import _repair_single_elite

        X = random_chromosome.copy()
        r1 = _repair_single_elite(X.copy(), str(PKL_PATH), 4, gen=1, idx=0)
        r2 = _repair_single_elite(X.copy(), str(PKL_PATH), 4, gen=2, idx=0)
        # Not strictly guaranteed, but overwhelmingly likely for 790 events
        assert not np.array_equal(r1, r2), "Different seeds produced identical output"

    def test_does_not_mutate_input(self, random_chromosome):
        """Worker must not modify the input array."""
        from src.experiments.ga_experiment import _repair_single_elite

        X = random_chromosome.copy()
        X_saved = X.copy()
        _repair_single_elite(X, str(PKL_PATH), 2, 1, 0)
        np.testing.assert_array_equal(X, X_saved)


# ─── Test 3: Parallel dispatch matches sequential ────────────────────


@pytest.mark.skipif(not PKL_EXISTS, reason=SKIP_MSG)
class TestParallelDispatch:
    """Test ProcessPoolExecutor dispatch produces correct results."""

    def test_parallel_matches_sequential(self, random_chromosome):
        """4 individuals repaired in parallel should match sequential."""
        from src.experiments.ga_experiment import _repair_single_elite

        rng = np.random.default_rng(99)
        n_elites = 4
        gen = 5
        repair_iters = 3

        # Create slightly different chromosomes
        X_list = []
        for i in range(n_elites):
            X = random_chromosome.copy()
            # Perturb a few genes
            for _ in range(10):
                pos = rng.integers(len(X))
                X[pos] = rng.integers(0, 100)
            X_list.append(X)

        # Sequential
        seq_results = []
        for i in range(n_elites):
            r = _repair_single_elite(
                X_list[i].copy(), str(PKL_PATH), repair_iters, gen, i
            )
            seq_results.append(r)

        # Parallel
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _repair_single_elite,
                    X_list[i].copy(),
                    str(PKL_PATH),
                    repair_iters,
                    gen,
                    i,
                )
                for i in range(n_elites)
            ]
            par_results = [f.result() for f in futures]

        for i in range(n_elites):
            np.testing.assert_array_equal(
                seq_results[i],
                par_results[i],
                err_msg=f"Mismatch at elite {i}",
            )

    def test_pool_does_not_crash(self, random_chromosome):
        """Smoke test: pool with max_workers=2 completes without error."""
        from src.experiments.ga_experiment import _repair_single_elite

        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _repair_single_elite,
                    random_chromosome.copy(),
                    str(PKL_PATH),
                    1,
                    1,
                    i,
                )
                for i in range(3)
            ]
            results = [f.result() for f in futures]

        assert len(results) == 3
        for r in results:
            assert r.shape == random_chromosome.shape


# ─── Test 4: MemeticExperiment integration ────────────────────────────


class TestMemeticIntegration:
    """Test MemeticExperiment wiring (no pkl needed)."""

    def test_repair_frequency_parameter(self):
        """MemeticExperiment accepts repair_frequency."""
        from src.experiments.ga_experiment import MemeticExperiment

        exp = MemeticExperiment(
            seed=1,
            pop_size=10,
            ngen=5,
            elite_pct=0.2,
            repair_iters=1,
            repair_frequency=3,
            data_dir=str(PROJECT_ROOT / "data"),
        )
        assert exp.repair_frequency == 3
        assert exp.repair_iters == 1
        assert exp.elite_pct == 0.2

    def test_default_repair_frequency(self):
        """Default repair_frequency is 5."""
        from src.experiments.ga_experiment import MemeticExperiment

        exp = MemeticExperiment(
            seed=1,
            pop_size=10,
            ngen=5,
            data_dir=str(PROJECT_ROOT / "data"),
        )
        assert exp.repair_frequency == 5

    def test_worker_function_signature(self):
        """_repair_single_elite has the expected parameters."""
        import inspect

        from src.experiments.ga_experiment import _repair_single_elite

        sig = inspect.signature(_repair_single_elite)
        params = list(sig.parameters.keys())
        assert params == ["X", "pkl_path", "repair_iters", "gen", "idx"]
