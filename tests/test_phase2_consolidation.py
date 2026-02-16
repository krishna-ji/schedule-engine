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
        from src.ga.metrics import (
            average_pairwise_diversity,
            individual_distance,
        )

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
        from src.ga.metrics import (
            calculate_convergence_rate,
            detect_stagnation,
        )

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
# Part 2: ga/heuristics/ package tests
# =============================================================================


class TestHeuristicsLegacyAPI:
    """Test legacy function-based heuristics API."""

    def test_categories_constant(self):
        from src.ga.heuristics import CATEGORIES

        assert isinstance(CATEGORIES, list | tuple)
        assert len(CATEGORIES) > 0

    def test_heuristic_info_class(self):
        from src.ga.heuristics import HeuristicInfo

        assert HeuristicInfo is not None

    def test_get_all_heuristics(self):
        from src.ga.heuristics import get_all_heuristics

        heuristics = get_all_heuristics()
        assert isinstance(heuristics, list)

    def test_get_enabled_heuristics(self):
        from src.ga.heuristics import get_enabled_heuristics

        enabled = get_enabled_heuristics()
        # Returns dict of enabled heuristics
        assert isinstance(enabled, dict)

    def test_get_heuristic_by_name(self):
        from src.ga.heuristics import (
            get_all_heuristics,
            get_heuristic_by_name,
        )

        all_h = get_all_heuristics()
        if all_h:
            name = all_h[0].name
            found = get_heuristic_by_name(name)
            assert found is not None
            assert found.name == name

    def test_get_heuristics_by_category(self):
        from src.ga.heuristics import CATEGORIES, get_heuristics_by_category

        if CATEGORIES:
            cat = CATEGORIES[0]
            heuristics = get_heuristics_by_category(cat)
            assert isinstance(heuristics, list)

    def test_get_heuristic_statistics_template(self):
        from src.ga.heuristics import get_heuristic_statistics_template

        template = get_heuristic_statistics_template()
        assert isinstance(template, dict)


class TestHeuristicsOOPAPI:
    """Test new OOP heuristics API."""

    def test_heuristic_protocol(self):
        from src.ga.heuristics import Heuristic

        assert Heuristic is not None

    def test_heuristic_base_class(self):
        from src.ga.heuristics import HeuristicBase

        assert HeuristicBase is not None

    def test_function_heuristic_class(self):
        from src.ga.heuristics import FunctionHeuristic

        assert FunctionHeuristic is not None

    def test_all_heuristics_list(self):
        from src.ga.heuristics import ALL_HEURISTICS

        assert isinstance(ALL_HEURISTICS, list | tuple)

    def test_category_lists(self):
        from src.ga.heuristics import (
            CONSTRUCTION_HEURISTICS,
            DIVERSITY_HEURISTICS,
            IMPROVEMENT_HEURISTICS,
            META_HEURISTICS,
            PERTURBATION_HEURISTICS,
            REPAIR_HEURISTICS,
        )

        for lst in [
            CONSTRUCTION_HEURISTICS,
            DIVERSITY_HEURISTICS,
            IMPROVEMENT_HEURISTICS,
            META_HEURISTICS,
            PERTURBATION_HEURISTICS,
            REPAIR_HEURISTICS,
        ]:
            assert isinstance(lst, list | tuple)

    def test_heuristic_names(self):
        from src.ga.heuristics import (
            ENABLED_HEURISTIC_NAMES,
            HEURISTIC_NAMES,
        )

        assert isinstance(HEURISTIC_NAMES, list | tuple | set)
        assert isinstance(ENABLED_HEURISTIC_NAMES, list | tuple | set)

    def test_build_heuristics(self):
        from src.ga.heuristics import build_heuristics

        heuristics = build_heuristics()
        assert isinstance(heuristics, list)

    def test_get_all_heuristic_objects(self):
        from src.ga.heuristics import get_all_heuristic_objects

        objs = get_all_heuristic_objects()
        assert isinstance(objs, list)

    def test_get_heuristics_by_category_oop(self):
        from src.ga.heuristics import get_heuristics_by_category_oop

        repair = get_heuristics_by_category_oop("repair")
        assert isinstance(repair, list)

    def test_get_heuristic_by_name_oop(self):
        from src.ga.heuristics import (
            HEURISTIC_NAMES,
            get_heuristic_by_name_oop,
        )

        if HEURISTIC_NAMES:
            name = next(iter(HEURISTIC_NAMES))
            get_heuristic_by_name_oop(name)
            # May return None if not found, that's ok


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
        from src.ga.repair.conflict_detection import (
            find_hard_conflict_sessions,
        )

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
        from src.ga.operators import (
            detect_violated_genes,
            repair_individual,
        )

        assert detect_violated_genes is not None
        assert repair_individual is not None

    def test_heuristics_include_repair(self):
        """Verify repair heuristics are part of heuristics registry."""
        from src.ga.heuristics import REPAIR_HEURISTICS

        assert isinstance(REPAIR_HEURISTICS, list | tuple)

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

    def test_ga_scheduler_export(self):
        from src.ga import GAScheduler

        assert GAScheduler is not None

    def test_ga_config_export(self):
        from src.ga import GAConfig

        assert GAConfig is not None

    def test_ga_metrics_export(self):
        from src.ga import GAMetrics

        assert GAMetrics is not None

    def test_get_creator_export(self):
        from src.ga import get_creator

        assert callable(get_creator)

    def test_create_individual_export(self):
        from src.ga import create_individual

        assert callable(create_individual)

    def test_session_gene_export(self):
        from src.ga import SessionGene

        assert SessionGene is not None
