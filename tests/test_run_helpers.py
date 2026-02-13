"""Tests for run_helpers: dead code removal and import cleanliness."""

from __future__ import annotations

import pytest


class TestRunHelpersDeadCodeRemoval:
    """Verify dead code was properly removed."""

    def test_create_detailed_evaluator_removed(self):
        """create_detailed_evaluator should no longer exist."""
        import schedule_engine.ga.run_helpers as rh

        assert not hasattr(rh, "create_detailed_evaluator")

    def test_create_detailed_evaluator_not_importable(self):
        with pytest.raises(ImportError):
            from schedule_engine.ga.run_helpers import (
                create_detailed_evaluator,
            )  # noqa: F401


class TestRunHelpersPublicAPI:
    """All expected public symbols should still be importable."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "NotebookData",
            "EvolutionConfig",
            "EvolutionStats",
            "track_nsga_metrics",
            "stats_to_ga_metrics",
            "load_data",
            "create_random_individual",
            "create_evaluator",
            "course_aware_crossover",
            "smart_mutation",
            "setup_deap",
            "get_constraint_breakdown",
            "run_nsga2",
            "get_best_individual",
            "print_constraint_details",
        ],
    )
    def test_symbol_exists(self, symbol):
        import schedule_engine.ga.run_helpers as rh

        assert hasattr(rh, symbol), f"{symbol} missing from run_helpers"


class TestRunHelpersWrappers:
    """Thin wrappers should delegate correctly."""

    def test_course_aware_crossover_delegates(self):
        """course_aware_crossover should call crossover_course_group_aware."""
        import inspect

        from schedule_engine.ga.run_helpers import course_aware_crossover

        source = inspect.getsource(course_aware_crossover)
        assert "crossover_course_group_aware" in source

    def test_smart_mutation_delegates(self):
        """smart_mutation should call mutate_individual."""
        import inspect

        from schedule_engine.ga.run_helpers import smart_mutation

        source = inspect.getsource(smart_mutation)
        assert "mutate_individual" in source
