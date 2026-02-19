"""Tests for dead code removal: verify deleted modules are gone."""

from __future__ import annotations

import pytest


class TestDeletedModules:
    """Verify deleted modules are no longer importable."""

    def test_shared_course_analyzer_deleted(self):
        with pytest.raises(ImportError):
            pass

    def test_rl_rewards_deleted(self):
        with pytest.raises(ImportError):
            pass

    def test_metrics_package_deleted(self):
        """Entire metrics/ package is deleted - use ga.metrics."""
        with pytest.raises(ImportError):
            pass

    def test_heuristics_package_deleted(self):
        """Top-level heuristics/ is deleted - use ga.heuristics."""
        with pytest.raises(ImportError):
            pass


class TestGAHeuristicsPackage:
    """ga.heuristics is the canonical location for heuristics."""

    def test_ga_heuristics_available(self):
        """ga.heuristics provides all heuristic functions."""
        from src.ga.heuristics import get_all_heuristics
        from src.ga.repair.lns.repair import lns_repair

        assert callable(get_all_heuristics)
        assert callable(lns_repair)


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
            pass
        with pytest.raises(ImportError):
            pass
        with pytest.raises(ImportError):
            pass

    def test_output_plots_ga(self):
        """io.export has GA plotting functions."""
        from src.io.export.plotdiversity import plot_diversity_trend
        from src.io.export.plotpareto import plot_pareto_front

        assert callable(plot_pareto_front)
        assert callable(plot_diversity_trend)

    def test_output_plots_rl(self):
        """rl.training.visualizer has RL visualization functions."""
        from src.rl.training.visualizer import (
            load_tensorboard_data,
            plot_training_curves,
        )

        assert callable(load_tensorboard_data)
        assert callable(plot_training_curves)
