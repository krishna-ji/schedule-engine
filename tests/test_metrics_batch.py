"""Tests for metrics_batch — pure numpy input/output, pymoo/scipy internals.

Gates:
    1. Shape/dtype contracts
    2. Known-answer test (simple 2D front)
    3. Edge cases (single individual, identical objectives)
"""

from __future__ import annotations

import numpy as np


class TestMetricsBatch:
    """Metrics batch API shape/dtype and known-answer tests."""

    def test_shape_dtype(self):
        from src.pipeline.batch_api import metrics_batch

        F = np.array(
            [
                [10.0, 200.0],
                [5.0, 300.0],
                [15.0, 150.0],
                [3.0, 400.0],
                [20.0, 100.0],
            ]
        )
        result = metrics_batch(F)

        assert "hv" in result
        assert "igd" in result
        assert "spacing" in result
        assert "n_fronts" in result
        assert isinstance(result["hv"], float)
        assert isinstance(result["igd"], float)
        assert isinstance(result["spacing"], float)
        assert isinstance(result["n_fronts"], int)

    def test_known_pareto_front(self):
        from src.pipeline.batch_api import metrics_batch

        # All non-dominated (Pareto front = all 3)
        F = np.array(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
            ]
        )
        result = metrics_batch(F)
        assert result["n_fronts"] == 1
        assert result["hv"] > 0
        assert result["spacing"] >= 0

    def test_dominated_individuals(self):
        from src.pipeline.batch_api import metrics_batch

        F = np.array(
            [
                [1.0, 1.0],  # dominates next
                [2.0, 2.0],  # dominated
                [3.0, 3.0],  # dominated
            ]
        )
        result = metrics_batch(F)
        assert result["n_fronts"] >= 2  # at least 2 fronts

    def test_single_individual(self):
        from src.pipeline.batch_api import metrics_batch

        F = np.array([[5.0, 10.0]])
        result = metrics_batch(F)
        assert result["n_fronts"] == 1
        assert result["hv"] > 0
        assert result["spacing"] == 0.0  # single point, no spacing

    def test_hv_with_ref_point(self):
        from src.pipeline.batch_api import metrics_batch

        F = np.array(
            [
                [1.0, 3.0],
                [3.0, 1.0],
            ]
        )
        ref = np.array([10.0, 10.0])
        result = metrics_batch(F, ref_point=ref)
        assert result["hv"] > 0

    def test_igd_nonnegative(self):
        from src.pipeline.batch_api import metrics_batch

        rng = np.random.default_rng(42)
        F = rng.random((20, 2)) * 100
        result = metrics_batch(F)
        assert result["igd"] >= 0
