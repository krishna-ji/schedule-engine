"""Tests for dead code removal: verify deleted modules are gone."""

from __future__ import annotations

import pytest


class TestDeletedModules:
    """Verify deleted modules are no longer importable."""

    def test_shared_course_analyzer_deleted(self):
        with pytest.raises(ImportError):
            import src.shared_course_analyzer  # noqa: F401

    def test_rl_rewards_deleted(self):
        with pytest.raises(ImportError):
            import src.rl_rewards  # noqa: F401

    def test_metrics_package_deleted(self):
        """Top-level metrics/ submodules are deleted - use ga.metrics."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from src.metrics import calculate_hypervolume  # noqa: F401

    def test_heuristics_package_deleted(self):
        """Top-level heuristics/ is deleted."""
        with pytest.raises(ImportError):
            import src.heuristics  # noqa: F401

    def test_ga_heuristics_deleted(self):
        """ga.heuristics is deleted (superseded by rl.actions)."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from src.ga.heuristics import get_all_heuristics  # noqa: F401


class TestGAMetricsPackage:
    """ga.metrics is the canonical location for all metrics."""

    def test_ga_metrics_exports(self):
        """ga.metrics exports all necessary metrics functions."""
        from src.ga.metrics import (
            ViolationHeatmap,
            average_pairwise_diversity,
            calculate_hypervolume,
            calculate_spacing,
        )

        assert callable(calculate_hypervolume)
        assert callable(average_pairwise_diversity)
        assert callable(calculate_spacing)
        assert ViolationHeatmap is not None


class TestIOPackageExports:
    """Verify DataStore is properly exported from io package."""

    def test_data_store_in_io(self):
        from src.io import DataStore

        assert DataStore is not None


class TestNewPackageStructure:
    """Test the restructured package organization."""

    def test_core_package_exports(self):
        """domain/ provides unified access to domain models."""
        from src.constraints import Evaluator
        from src.domain import (
            Course,
            Group,
            Instructor,
            Room,
            SchedulingContext,
            SessionGene,
        )

        assert Course is not None
        assert Group is not None
        assert Instructor is not None
        assert Room is not None
        assert SessionGene is not None
        assert SchedulingContext is not None
        assert Evaluator is not None

    def test_ga_metrics_package(self):
        """ga.metrics is the canonical location for GA metrics."""
        from src.ga.metrics import (
            ViolationHeatmap,
            average_pairwise_diversity,
            calculate_hypervolume,
            calculate_spacing,
        )

        assert callable(calculate_hypervolume)
        assert callable(average_pairwise_diversity)
        assert callable(calculate_spacing)
        assert ViolationHeatmap is not None

    def test_output_package_deleted(self):
        """experiments.output was removed with the experiments package."""
        with pytest.raises(ImportError):
            import src.experiments.output
        with pytest.raises(ImportError):
            import src.experiments.output.plots
        with pytest.raises(ImportError):
            import src.experiments.output.tables  # noqa: F401

    def test_output_plots_ga(self):
        """io.export has GA plotting functions."""
        from src.io.export.plotdiversity import plot_diversity_trend
        from src.io.export.plotpareto import plot_pareto_front_from_F

        assert callable(plot_pareto_front_from_F)
        assert callable(plot_diversity_trend)

    def test_output_plots_rl_deleted(self):
        """rl.training.visualizer was removed (orphan module)."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from src.rl.training.visualizer import plot_training_curves  # noqa: F401
