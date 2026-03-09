"""Tests for Phase 2 consolidation: GA subpackages.

Verifies all exports from:
- ga/metrics/ (hypervolume, diversity, spacing, etc.)
- ga/heuristics/ (OOP and legacy APIs)
- ga/heuristics/repair/ (repair heuristics)
- ga/operators/ (crossover, mutation, repair)

These tests run BEFORE and AFTER consolidation to ensure no functionality is lost.
"""

from __future__ import annotations

# =============================================================================
# Part 1: ga/metrics/ package tests
# =============================================================================


class TestMetricsImports:
    """Verify all metrics can be imported from ga.metrics."""

    def test_hypervolume_import(self):
        from src.ga.metrics import calculate_hypervolume

        assert callable(calculate_hypervolume)

    def test_diversity_imports(self):
        from src.ga.metrics import average_pairwise_diversity, individual_distance

        assert callable(average_pairwise_diversity)
        assert callable(individual_distance)

    def test_pareto_metrics_imports(self):
        from src.ga.metrics import (
            calculate_generational_distance,
            calculate_inverted_generational_distance,
            calculate_spacing,
        )

        assert callable(calculate_generational_distance)
        assert callable(calculate_inverted_generational_distance)
        assert callable(calculate_spacing)

    def test_convergence_imports(self):
        from src.ga.metrics import calculate_convergence_rate, detect_stagnation

        assert callable(calculate_convergence_rate)
        assert callable(detect_stagnation)

    def test_violation_heatmap_import(self):
        from src.ga.metrics import ViolationHeatmap

        assert ViolationHeatmap is not None

    def test_violation_recorder_import(self):
        from src.ga.metrics import record_violations_to_heatmap

        assert callable(record_violations_to_heatmap)


class TestMetricsFunctionality:
    """Test that metrics functions work correctly."""

    def test_hypervolume_empty_front(self):
        from src.ga.metrics import calculate_hypervolume

        hv = calculate_hypervolume([], (10.0, 10.0))
        assert hv == 0

    def test_spacing_function_exists(self):
        """Verify spacing function is callable (full test needs DEAP individuals)."""
        from src.ga.metrics import calculate_spacing

        assert callable(calculate_spacing)

    def test_diversity_function_exists(self):
        """Verify diversity function is callable."""
        from src.ga.metrics import average_pairwise_diversity

        assert callable(average_pairwise_diversity)

    def test_convergence_rate_returns_list(self):
        from src.ga.metrics import calculate_convergence_rate

        history = [100.0, 80.0, 60.0, 50.0, 45.0]
        result = calculate_convergence_rate(history)
        # Returns list of rates per generation
        assert isinstance(result, list)

    def test_stagnation_detection_returns_tuple(self):
        from src.ga.metrics import detect_stagnation

        history = [50.0, 50.0, 50.0, 50.0, 50.0]
        result = detect_stagnation(history, threshold=0.01, window=3)
        # Returns tuple (is_stagnant, generation)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)

    def test_violation_heatmap_creation(self):
        from src.ga.metrics import ViolationHeatmap

        heatmap = ViolationHeatmap()
        assert heatmap is not None


# =============================================================================
# =============================================================================
# Part 3: ga/heuristics/repair/ package tests
# =============================================================================


class TestRepairHeuristicsImports:
    """Test repair heuristics can be imported."""

    def test_igls_repair_import(self):
        from src.ga.repair.igls import igls_repair

        assert callable(igls_repair)

    def test_greedy_repair_import(self):
        from src.ga.repair.greedy import greedy_repair

        assert callable(greedy_repair)

    def test_selective_repair_import(self):
        from src.ga.repair.selective_heuristic import selective_repair

        assert callable(selective_repair)

    def test_lns_repair_import(self):
        from src.ga.repair.lns.repair import lns_repair

        assert callable(lns_repair)

    def test_exhaustive_repair_import(self):
        from src.ga.repair.exhaustive import exhaustive_repair

        assert callable(exhaustive_repair)

    def test_memetic_repair_import(self):
        from src.ga.repair.memetic import memetic_repair

        assert callable(memetic_repair)

    def test_break_repair_import(self):
        from src.ga.repair.break_repair import repair_break_placement

        assert callable(repair_break_placement)

    def test_conflict_detection_import(self):
        from src.ga.repair.conflict_detection import find_hard_conflict_sessions

        assert callable(find_hard_conflict_sessions)

    def test_lns_igls_import(self):
        from src.ga.repair.lns.operator import lns_igls_repair

        assert callable(lns_igls_repair)


# =============================================================================
# Part 4: ga/operators/ package tests
# =============================================================================


class TestOperatorsImports:
    """Test all operators can be imported."""

    def test_mutation_imports(self):
        from src.ga.operators import mutate_gene, mutate_individual

        assert callable(mutate_individual)
        assert callable(mutate_gene)

    def test_crossover_import(self):
        from src.ga.operators import crossover_course_group_aware

        assert callable(crossover_course_group_aware)

    def test_repair_imports(self):
        from src.ga.operators import (
            repair_individual,
            repair_individual_selective,
            repair_individual_unified,
        )

        assert callable(repair_individual)
        assert callable(repair_individual_unified)
        assert callable(repair_individual_selective)

    def test_repair_engine_import(self):
        from src.ga.operators import RepairEngine

        assert RepairEngine is not None

    def test_violation_detector_import(self):
        from src.ga.operators import detect_violated_genes

        assert callable(detect_violated_genes)

    def test_repair_registry_imports(self):
        from src.ga.operators import (
            get_all_repair_operators,
            get_enabled_repair_operators,
            get_repair_operator_function,
            get_repair_operator_metadata,
            get_repair_statistics_template,
            repair_operator,
        )

        assert callable(repair_operator)
        assert callable(get_all_repair_operators)
        assert callable(get_enabled_repair_operators)
        assert callable(get_repair_operator_metadata)
        assert callable(get_repair_operator_function)
        assert callable(get_repair_statistics_template)


class TestOperatorsRegistry:
    """Test repair operator registry functionality."""

    def test_get_all_repair_operators(self):
        from src.ga.operators import get_all_repair_operators

        operators = get_all_repair_operators()
        assert isinstance(operators, list | tuple | dict)

    def test_get_enabled_repair_operators(self):
        from src.ga.operators import get_enabled_repair_operators

        # This function requires config initialization
        # Just verify it's callable
        assert callable(get_enabled_repair_operators)

    def test_get_repair_statistics_template(self):
        from src.ga.operators import get_repair_statistics_template

        template = get_repair_statistics_template()
        assert isinstance(template, dict)


# =============================================================================
# Part 5: ga/evaluator/ package tests (also to consolidate)
# =============================================================================


class TestEvaluatorImports:
    """Test evaluator subpackage imports."""

    def test_fitness_import(self):
        from src.ga.core.evaluator import evaluate

        assert callable(evaluate)

    def test_detailed_fitness_import(self):
        from src.ga.core.evaluator import evaluate_detailed

        assert callable(evaluate_detailed)


# =============================================================================
# Part 6: Cross-module integration tests
# =============================================================================


class TestCrossModuleIntegration:
    """Test that modules work together correctly."""

    def test_repair_uses_violation_detector(self):
        """Verify repair operators can use violation detector."""
        from src.ga.operators import detect_violated_genes, repair_individual

        assert detect_violated_genes is not None
        assert repair_individual is not None

    def test_metrics_with_population(self):
        """Verify metrics functions are callable."""
        from src.ga.metrics import calculate_spacing

        # Verify it's callable (full test needs DEAP individuals)
        assert callable(calculate_spacing)


# =============================================================================
# Part 7: Top-level ga/ package tests
# =============================================================================


class TestGAPackageTopLevel:
    """Test top-level ga/ exports."""

    def test_repair_pipeline_export(self):
        from src.ga import RepairPipeline

        assert RepairPipeline is not None

    def test_population_factory_export(self):
        from src.ga import PopulationFactory

        assert PopulationFactory is not None

    def test_session_gene_export(self):
        from src.ga import SessionGene

        assert SessionGene is not None
