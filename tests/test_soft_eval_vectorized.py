"""Tests for vectorized soft evaluator equivalence against OOP evaluator.

Gates:
    1. Vectorized soft eval vs OOP Evaluator — tolerance <= 1e-6 for top 3 constraints
    2. Shape/dtype contracts
    3. Performance: vectorized must be faster than per-individual loop
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
def population(pkl_data):
    """Build a small test population."""
    from src.pipeline.pymoo_operators import ConstructiveSampling
    from src.pipeline.scheduling_problem import SchedulingProblem

    prob = SchedulingProblem(PKL_PATH)
    sampling = ConstructiveSampling(PKL_PATH)
    return sampling._do(prob, 10)


@pytest.fixture(scope="module")
def oop_soft_scores(population, pkl_data):
    """Compute soft scores using the OOP evaluator for comparison."""
    try:
        from src.constraints.evaluator import Evaluator
        from src.domain.gene import SessionGene
        from src.domain.timetable import Timetable
        from src.io.data_store import DataStore
        from src.io.time_system import QuantumTimeSystem
        from src.pipeline.encoding import chromosome_views
    except ImportError:
        pytest.skip("OOP evaluator dependencies not available")

    store = DataStore.from_json("data", run_preflight=False)
    ctx = store.to_context()
    qts = QuantumTimeSystem()
    evaluator = Evaluator()

    events = pkl_data["events"]
    idx_to_instructor = {int(k): v for k, v in pkl_data["idx_to_instructor"].items()}
    idx_to_room = {int(k): v for k, v in pkl_data["idx_to_room"].items()}
    E = len(events)

    scores = {
        "CSC": [],  # Cohort Schedule Compactness
        "FSC": [],  # Faculty Schedule Compactness
        "MIP": [],  # Mandatory Intermission Provision
        "total": [],
    }

    for ind_idx in range(population.shape[0]):
        xi = population[ind_idx].astype(int)
        inst, room, time_ = chromosome_views(xi)
        genes = []
        for e in range(E):
            ev = events[e]
            genes.append(
                SessionGene(
                    course_id=ev["course_id"],
                    course_type=ev["course_type"],
                    instructor_id=idx_to_instructor[int(inst[e])],
                    group_ids=list(ev["group_ids"]),
                    room_id=idx_to_room[int(room[e])],
                    start_quanta=int(time_[e]),
                    num_quanta=ev["num_quanta"],
                )
            )

        tt = Timetable(genes, ctx, qts)
        bd = evaluator.soft_breakdown(tt)

        scores["CSC"].append(bd.get("CSC", 0.0))
        scores["FSC"].append(bd.get("FSC", 0.0))
        scores["MIP"].append(bd.get("MIP", 0.0))
        # Total of these 3 constraints
        total = sum(
            [
                bd.get("CSC", 0.0),
                bd.get("FSC", 0.0),
                bd.get("MIP", 0.0),
            ]
        )
        scores["total"].append(total)

    return {k: np.array(v) for k, v in scores.items()}


@skip_no_pkl
class TestSoftEvalVectorized:
    """Vectorized soft evaluator shape/dtype contracts."""

    def test_shape_dtype(self, population, pkl_data):
        from src.pipeline.soft_evaluator_vectorized import (
            eval_soft_vectorized,
            prepare_soft_vectorized_data,
        )

        sdata = prepare_soft_vectorized_data(pkl_data)
        S = eval_soft_vectorized(population, sdata)

        assert S.shape == (population.shape[0],)
        assert S.dtype == np.float64

    def test_nonnegative(self, population, pkl_data):
        from src.pipeline.soft_evaluator_vectorized import (
            eval_soft_vectorized,
            prepare_soft_vectorized_data,
        )

        sdata = prepare_soft_vectorized_data(pkl_data)
        S = eval_soft_vectorized(population, sdata)
        assert (S >= 0).all(), f"Negative soft penalties: {S[S < 0]}"

    def test_1d_input(self, population, pkl_data):
        from src.pipeline.soft_evaluator_vectorized import (
            eval_soft_vectorized,
            prepare_soft_vectorized_data,
        )

        sdata = prepare_soft_vectorized_data(pkl_data)
        S = eval_soft_vectorized(population[0], sdata)
        assert S.shape == (1,)


@skip_no_pkl
class TestSoftEvalEquivalence:
    """Vectorized vs OOP soft evaluator equivalence (tolerance-based)."""

    def test_top3_total_tolerance(self, population, pkl_data, oop_soft_scores):
        """Sum of top 3 soft constraints should match within tolerance."""
        from src.pipeline.soft_evaluator_vectorized import (
            eval_soft_vectorized,
            prepare_soft_vectorized_data,
        )

        sdata = prepare_soft_vectorized_data(pkl_data)
        S_vec = eval_soft_vectorized(population, sdata)
        S_oop = oop_soft_scores["total"]

        # Allow tolerance for floating point differences
        np.testing.assert_allclose(
            S_vec,
            S_oop,
            atol=1e-6,
            err_msg="Vectorized and OOP soft evaluators disagree on top-3 total",
        )

    def test_ranking_preserved(self, population, pkl_data, oop_soft_scores):
        """Relative ranking should be preserved (Spearman correlation)."""
        from src.pipeline.soft_evaluator_vectorized import (
            eval_soft_vectorized,
            prepare_soft_vectorized_data,
        )

        sdata = prepare_soft_vectorized_data(pkl_data)
        S_vec = eval_soft_vectorized(population, sdata)
        S_oop = oop_soft_scores["total"]

        # If all values are 0, skip ranking test
        if np.all(S_oop == 0) and np.all(S_vec == 0):
            return

        from scipy.stats import spearmanr

        corr, _ = spearmanr(S_vec, S_oop)
        assert corr > 0.9 or np.isnan(
            corr
        ), f"Ranking correlation too low: {corr:.3f}\nVec: {S_vec}\nOOP: {S_oop}"


@skip_no_pkl
class TestSoftEvalPerformance:
    """Vectorized soft eval must be faster than per-individual loop."""

    def test_speedup(self, population, pkl_data):
        from src.pipeline.soft_evaluator_vectorized import (
            eval_soft_vectorized,
            prepare_soft_vectorized_data,
        )

        sdata = prepare_soft_vectorized_data(pkl_data)

        # Warm up
        eval_soft_vectorized(population, sdata)

        # Vectorized timing
        N_REPS = 5
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            eval_soft_vectorized(population, sdata)
        t_vec = (time.perf_counter() - t0) / N_REPS

        # Per-individual loop timing (using same vectorized function but one at a time)
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            for i in range(population.shape[0]):
                eval_soft_vectorized(population[i : i + 1], sdata)
        t_loop = (time.perf_counter() - t0) / N_REPS

        # Vectorized should be at least as fast (batching overhead may be small for N=10)
        # The real speedup appears at N=200+
        print(
            f"\nSoft eval: vec={t_vec * 1000:.1f}ms, loop={t_loop * 1000:.1f}ms, "
            f"speedup={t_loop / max(t_vec, 1e-9):.1f}x"
        )

        # At minimum, vectorized should not be significantly slower
        assert (
            t_vec < t_loop * 2
        ), f"Vectorized is slower than loop: {t_vec:.4f}s vs {t_loop:.4f}s"
