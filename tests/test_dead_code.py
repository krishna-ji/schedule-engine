"""Tests for dead code removal: verify deleted modules are gone."""

from __future__ import annotations

import pytest


class TestDeletedModules:
    """Verify deleted modules are no longer importable."""

    def test_shared_course_analyzer_deleted(self):
        with pytest.raises(ImportError):
            import schedule_engine.ga.shared_course_analyzer  # noqa: F401

    def test_behavioral_archive_deleted(self):
        with pytest.raises(ImportError):
            from schedule_engine.metrics.behavioral_archive import (
                BehavioralArchive,
            )  # noqa: F401

    def test_behavioral_features_deleted(self):
        with pytest.raises(ImportError):
            from schedule_engine.metrics.behavioral_features import (
                extract_behavioral_features,
            )  # noqa: F401

    def test_rl_rewards_deleted(self):
        with pytest.raises(ImportError):
            import schedule_engine.rl.rewards  # noqa: F401


class TestMetricsPackageClean:
    """Metrics package should work without deleted behavioral modules."""

    def test_metrics_import(self):
        from schedule_engine.metrics import (
            average_pairwise_diversity,
            calculate_hypervolume,
            calculate_spacing,
            compute_novelty,
            k_nearest_neighbors,
        )

        assert callable(calculate_hypervolume)
        assert callable(average_pairwise_diversity)
        assert callable(compute_novelty)
        assert callable(k_nearest_neighbors)
        assert callable(calculate_spacing)

    def test_no_behavioral_in_metrics_all(self):
        import schedule_engine.metrics as m

        assert "BehavioralArchive" not in m.__all__
        assert "extract_behavioral_features" not in m.__all__


class TestIOPackageExports:
    """Verify DataStore is properly exported from io package."""

    def test_data_store_in_io(self):
        from schedule_engine.io import DataStore

        assert DataStore is not None
