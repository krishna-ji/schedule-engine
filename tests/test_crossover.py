"""Tests for crossover operators: no global config dependency."""

from __future__ import annotations

import pytest

from src.domain.gene import SessionGene


def _make_gene(course: str, groups: list[str], start: int = 0) -> SessionGene:
    """Helper to create a SessionGene for testing."""
    return SessionGene(
        course_id=course,
        course_type="theory",
        group_ids=groups,
        instructor_id="I1",
        room_id="R1",
        start_quanta=start,
        num_quanta=2,
    )


class TestCrossoverNoConfigDependency:
    """Verify crossover.py has no get_config_or_default dependency."""

    def test_no_get_config_import(self):
        import inspect

        import src.ga.operators.crossover as cx_mod

        source = inspect.getsource(cx_mod)
        assert "get_config_or_default" not in source

    def test_validate_param_exists(self):
        import inspect

        from src.ga.operators.crossover import crossover_course_group_aware

        sig = inspect.signature(crossover_course_group_aware)
        assert "validate" in sig.parameters


class TestCrossoverValidation:
    """Test the validate parameter on crossover."""

    def test_matching_individuals_pass(self):
        from src.ga.operators.crossover import crossover_course_group_aware

        ind1 = [_make_gene("C1", ["G1"], start=0)]
        ind2 = [_make_gene("C1", ["G1"], start=5)]
        # Should not raise
        child1, child2 = crossover_course_group_aware(ind1, ind2, validate=True)
        assert len(child1) == 1
        assert len(child2) == 1

    def test_mismatched_individuals_raise_with_validate(self):
        from src.ga.operators.crossover import crossover_course_group_aware

        ind1 = [_make_gene("C1", ["G1"])]
        ind2 = [_make_gene("C2", ["G2"])]
        with pytest.raises(ValueError, match="CROSSOVER ERROR"):
            crossover_course_group_aware(ind1, ind2, validate=True)

    def test_mismatched_individuals_no_raise_without_validate(self):
        from src.ga.operators.crossover import crossover_course_group_aware

        ind1 = [_make_gene("C1", ["G1"])]
        ind2 = [_make_gene("C2", ["G2"])]
        # Should not raise — uses intersection of keys
        child1, child2 = crossover_course_group_aware(ind1, ind2, validate=False)
        assert len(child1) == 1
        assert len(child2) == 1


class TestConstraintAwareCrossoverNoConfig:
    """Verify constraint_aware.py has no get_config_or_default."""

    def test_no_get_config_import(self):
        import inspect

        import src.ga.operators.constraint_aware as cao_mod

        source = inspect.getsource(cao_mod)
        assert "get_config_or_default" not in source
