"""Tests for soft constraint module: setters, no global config dependency."""

from __future__ import annotations

import pytest


class TestSoftConstraintSetters:
    """Verify the module-level setter pattern for QTS and cohort pairs."""

    def test_set_qts_and_get_qts(self):
        from schedule_engine.constraints.soft import _get_qts, set_qts
        from schedule_engine.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        set_qts(qts)
        assert _get_qts() is qts

    def test_get_qts_lazy_default(self):
        """If _QTS is None, _get_qts should lazily create a default."""
        import schedule_engine.constraints.soft as soft_mod

        # Save and clear
        old = soft_mod._QTS
        soft_mod._QTS = None
        try:
            qts = soft_mod._get_qts()
            assert qts is not None
            assert qts.total_quanta > 0
        finally:
            soft_mod._QTS = old

    def test_set_cohort_pairs(self):
        import schedule_engine.constraints.soft as soft_mod
        from schedule_engine.constraints.soft import set_cohort_pairs

        pairs = [("A", "B"), ("C", "D")]
        set_cohort_pairs(pairs)
        assert soft_mod._COHORT_PAIRS == pairs

    def test_set_cohort_pairs_empty(self):
        import schedule_engine.constraints.soft as soft_mod
        from schedule_engine.constraints.soft import set_cohort_pairs

        set_cohort_pairs([])
        assert soft_mod._COHORT_PAIRS == []


class TestSoftConstraintNoConfigDependency:
    """Verify soft.py has no dependency on get_config_or_default."""

    def test_no_get_config_import(self):
        """The module source should not contain get_config_or_default."""
        import inspect

        import schedule_engine.constraints.soft as soft_mod

        source = inspect.getsource(soft_mod)
        assert "get_config_or_default" not in source

    def test_all_constraint_functions_callable(self):
        """All 6 soft constraint functions should be importable."""
        from schedule_engine.constraints.soft import (
            break_placement_compliance,
            instructor_schedule_compactness,
            paired_cohort_practical_alignment,
            session_continuity,
            student_lunch_break,
            student_schedule_compactness,
        )

        assert callable(student_schedule_compactness)
        assert callable(instructor_schedule_compactness)
        assert callable(student_lunch_break)
        assert callable(session_continuity)
        assert callable(break_placement_compliance)
        assert callable(paired_cohort_practical_alignment)


class TestSoftConstraintFunctions:
    """Smoke tests for soft constraint functions with empty sessions."""

    @pytest.fixture(autouse=True)
    def setup_qts(self):
        """Ensure QTS is set before running constraint functions."""
        from schedule_engine.constraints.soft import set_cohort_pairs, set_qts
        from schedule_engine.io.time_system import QuantumTimeSystem

        set_qts(QuantumTimeSystem())
        set_cohort_pairs([])

    def test_student_compactness_empty(self):
        from schedule_engine.constraints.soft import student_schedule_compactness

        assert student_schedule_compactness([]) == 0

    def test_instructor_compactness_empty(self):
        from schedule_engine.constraints.soft import instructor_schedule_compactness

        assert instructor_schedule_compactness([]) == 0

    def test_student_lunch_break_empty(self):
        from schedule_engine.constraints.soft import student_lunch_break

        assert student_lunch_break([]) == 0

    def test_session_continuity_empty(self):
        from schedule_engine.constraints.soft import session_continuity

        assert session_continuity([]) == 0

    def test_break_placement_empty(self):
        from schedule_engine.constraints.soft import break_placement_compliance

        assert break_placement_compliance([]) == 0

    def test_cohort_alignment_empty(self):
        from schedule_engine.constraints.soft import paired_cohort_practical_alignment

        assert paired_cohort_practical_alignment([], {}) == 0
