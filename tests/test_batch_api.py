"""Tests for the canonical batch API and vectorized evaluator equivalence.

Gates:
    1. batch vs scalar hard evaluator — exact match
    2. batch API shape/dtype contracts
    3. metrics_batch returns valid results
"""

from __future__ import annotations

import pickle
import sys
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


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def pkl_data():
    if not PKL_EXISTS:
        pytest.skip("events_with_domains.pkl not found")
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def population(pkl_data):
    """Build a small test population via constructive sampling."""
    from src.pipeline.pymoo_operators import ConstructiveSampling
    from src.pipeline.scheduling_problem import SchedulingProblem

    prob = SchedulingProblem(PKL_PATH)
    sampling = ConstructiveSampling(PKL_PATH)
    X = sampling._do(prob, 10)
    return X


@pytest.fixture(scope="module")
def batch_ctx():
    from src.pipeline.batch_api import BatchContext

    return BatchContext(PKL_PATH)


# ------------------------------------------------------------------
# Phase A: Hard evaluator equivalence
# ------------------------------------------------------------------


@skip_no_pkl
class TestHardEvaluatorEquivalence:
    """Vectorized hard evaluator must exactly match the batch bitset evaluator."""

    def test_exact_match(self, pkl_data, population):
        from src.pipeline.fast_evaluator_batch import (
            fast_evaluate_hard_batch,
            prepare_batch_data,
        )
        from src.pipeline.fast_evaluator_vectorized import (
            fast_evaluate_hard_vectorized,
            prepare_vectorized_data,
        )

        batch_data = prepare_batch_data(pkl_data)
        vec_data = prepare_vectorized_data(pkl_data)

        G_batch = fast_evaluate_hard_batch(population, batch_data)
        G_vec = fast_evaluate_hard_vectorized(population, vec_data)

        # Batch evaluator does not compute sibling_same_day (col 8 = 0).
        # Compare the shared first 8 columns for equivalence.
        np.testing.assert_array_equal(
            G_batch[:, :8],
            G_vec[:, :8],
            err_msg="Vectorized and batch hard evaluators disagree",
        )

    def test_single_individual(self, pkl_data, population):
        """Single individual should also work."""
        from src.pipeline.fast_evaluator_vectorized import (
            fast_evaluate_hard_vectorized,
            prepare_vectorized_data,
        )

        vec_data = prepare_vectorized_data(pkl_data)
        G = fast_evaluate_hard_vectorized(population[0:1], vec_data)
        assert G.shape == (1, 8)
        assert G.dtype == np.int64

    def test_1d_input(self, pkl_data, population):
        """1-D input (single chromosome) should be auto-reshaped."""
        from src.pipeline.fast_evaluator_vectorized import (
            fast_evaluate_hard_vectorized,
            prepare_vectorized_data,
        )

        vec_data = prepare_vectorized_data(pkl_data)
        G = fast_evaluate_hard_vectorized(population[0], vec_data)
        assert G.shape == (1, 8)


# ------------------------------------------------------------------
# Batch API shape/dtype contracts
# ------------------------------------------------------------------


@skip_no_pkl
class TestBatchAPIContract:
    """Verify the canonical batch API shapes and dtypes."""

    def test_eval_hard_batch_shape(self, population, batch_ctx):
        from src.pipeline.batch_api import eval_hard_batch

        G = eval_hard_batch(population, batch_ctx)
        N = population.shape[0]
        assert G.shape == (N, 8)
        assert G.dtype == np.int64
        assert (G >= 0).all(), "Violation counts must be non-negative"

    def test_eval_soft_batch_shape(self, population, batch_ctx):
        from src.pipeline.batch_api import eval_soft_batch

        S = eval_soft_batch(population, batch_ctx)
        N = population.shape[0]
        assert S.shape == (N,)
        assert S.dtype == np.float64

    def test_repair_batch_shape(self, population, batch_ctx):
        from src.pipeline.batch_api import repair_batch

        X_repaired = repair_batch(population[:3], batch_ctx)
        assert X_repaired.shape == (3, population.shape[1])
        assert X_repaired.dtype == np.int64

    def test_repair_reduces_violations(self, population, batch_ctx):
        """Repaired individuals should have fewer/equal violations."""
        from src.pipeline.batch_api import eval_hard_batch, repair_batch

        # Mangle some chromosomes to create violations
        X_bad = population[:5].copy()
        X_bad[:, 2::3] = 0  # set all times to 0 (massive conflicts)

        G_before = eval_hard_batch(X_bad, batch_ctx)
        X_fixed = repair_batch(X_bad, batch_ctx)
        G_after = eval_hard_batch(X_fixed, batch_ctx)

        # Repair should reduce total violations
        before_total = G_before.sum(axis=1)
        after_total = G_after.sum(axis=1)
        assert (
            after_total <= before_total
        ).all(), f"Repair made things worse: {before_total} -> {after_total}"

    def test_metrics_batch(self, population, batch_ctx):
        from src.pipeline.batch_api import eval_hard_batch, metrics_batch

        G = eval_hard_batch(population, batch_ctx)
        F = np.column_stack([G.sum(axis=1), np.zeros(len(G))])
        F = F.astype(np.float64)

        m = metrics_batch(F)
        assert "hv" in m
        assert "spacing" in m
        assert "igd" in m
        assert "n_fronts" in m
        assert isinstance(m["hv"], float)
        assert isinstance(m["n_fronts"], int)
        assert m["hv"] >= 0
        assert m["n_fronts"] >= 1

    def test_single_individual_api(self, population, batch_ctx):
        """1-D input should auto-reshape to (1, n_var)."""
        from src.pipeline.batch_api import eval_hard_batch

        G = eval_hard_batch(population[0], batch_ctx)
        assert G.shape == (1, 8)


# ------------------------------------------------------------------
# SchedulingProblem uses vectorized-only path
# ------------------------------------------------------------------


@skip_no_pkl
class TestSchedulingProblemCanonical:
    """Verify SchedulingProblem always uses vectorized evaluator."""

    def test_no_vectorized_flag(self):
        """The vectorized kwarg should no longer exist."""
        import inspect

        from src.pipeline.scheduling_problem import SchedulingProblem

        sig = inspect.signature(SchedulingProblem.__init__)
        assert "vectorized" not in sig.parameters, "vectorized flag should be removed"

    def test_evaluate_uses_vectorized(self, population):
        from src.pipeline.scheduling_problem import SchedulingProblem

        prob = SchedulingProblem(PKL_PATH)
        out = {}
        prob._evaluate(population, out)

        assert "F" in out
        assert "G" in out
        assert out["F"].shape == (population.shape[0], 2)
        assert out["G"].shape == (population.shape[0], 8)
