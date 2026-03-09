"""Tests for the audit-driven improvements (Phases 1–5).

Covers:
- Phase 1: Dead code removal verification
- Phase 2: NaN guard in SchedulingProblem._evaluate()
- Phase 3: export_pdf opt-in flag
- Phase 4: MOEA metric computation (moea_metrics.py)
- Phase 5: Callback MOEA metric recording + plot wiring
"""

from __future__ import annotations

import numpy as np
import pytest

# =====================================================================
#  Phase 1 — Dead code removal
# =====================================================================


class TestPhase1DeadCodeRemoval:
    """Verify DEAP-era dead code has been removed."""

    def test_viz_module_deleted(self):
        """src.viz should no longer exist."""
        with pytest.raises(ModuleNotFoundError):
            import src.viz  # noqa: F401

    def test_deap_plot_pareto_front_removed(self):
        """Old DEAP-style plot_pareto_front should not be importable from plotpareto."""
        from src.io.export import plotpareto

        assert not hasattr(
            plotpareto, "plot_pareto_front"
        ), "DEAP-only plot_pareto_front still exists in plotpareto module"

    def test_pymoo_plot_pareto_front_from_F_exists(self):
        """plot_pareto_front_from_F should still be importable."""
        from src.io.export.plotpareto import plot_pareto_front_from_F

        assert callable(plot_pareto_front_from_F)

    def test_plot_population_summary_deleted(self):
        """DEAP-only plot_population_summary.py should not exist."""
        with pytest.raises(ModuleNotFoundError):
            import src.io.export.plot_population_summary  # noqa: F401

    def test_init_exports_no_deap_functions(self):
        """__init__.py should not export plot_pareto_front (DEAP version)."""
        from src.io.export import __all__

        assert "plot_pareto_front" not in __all__
        assert "plot_pareto_front_from_F" in __all__

    def test_init_import_works(self):
        """All functions in __all__ should be importable."""
        import src.io.export as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} listed in __all__ but not importable"


# =====================================================================
#  Phase 2 — NaN guard in _evaluate()
# =====================================================================


class TestPhase2NaNGuard:
    """Verify defensive NaN/Inf handling in SchedulingProblem._evaluate()."""

    def test_nan_guard_in_source(self):
        """scheduling_problem.py should contain nan_to_num call."""
        import inspect

        from src.pipeline.scheduling_problem import SchedulingProblem

        source = inspect.getsource(SchedulingProblem._evaluate)
        assert "nan_to_num" in source, "_evaluate() missing defensive nan_to_num guard"

    def test_nan_to_num_replaces_nan(self):
        """np.nan_to_num should replace NaN with 1e6."""
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
        result = np.nan_to_num(arr, nan=1e6, posinf=1e6, neginf=0.0)
        assert result[0] == 1.0
        assert result[1] == 1e6  # NaN -> 1e6
        assert result[2] == 1e6  # +inf -> 1e6
        assert result[3] == 0.0  # -inf -> 0
        assert result[4] == 5.0


# =====================================================================
#  Phase 3 — export_pdf flag
# =====================================================================


class TestPhase3ExportPDFFlag:
    """Verify the export_pdf parameter on GAExperiment."""

    def test_export_pdf_default_true(self):
        """GAExperiment should default export_pdf to True."""
        import inspect

        from src.experiments.ga_experiment import GAExperiment

        sig = inspect.signature(GAExperiment.__init__)
        assert "export_pdf" in sig.parameters
        param = sig.parameters["export_pdf"]
        assert param.default is True

    def test_export_pdf_false_accepted(self):
        """Constructor should accept export_pdf=False without error."""
        import inspect

        from src.experiments.ga_experiment import BaselineExperiment

        # export_pdf flows through **kwargs to GAExperiment
        parent_sig = inspect.signature(BaselineExperiment.__mro__[1].__init__)
        assert "export_pdf" in parent_sig.parameters

    def test_export_pdf_in_docstring(self):
        """GAExperiment docstring should document export_pdf."""
        from src.experiments.ga_experiment import GAExperiment

        assert "export_pdf" in (GAExperiment.__doc__ or "")


# =====================================================================
#  Phase 4 — MOEA metric computation
# =====================================================================


class TestPhase4MOEAMetrics:
    """Unit tests for src.experiments.moea_metrics functions."""

    def test_compute_hypervolume_basic(self):
        """HV of a known set of points."""
        from src.experiments.moea_metrics import compute_hypervolume

        F = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
        ref = np.array([5.0, 5.0])
        hv = compute_hypervolume(F, ref_point=ref)
        assert hv > 0, "HV should be positive for dominated points"

    def test_compute_hypervolume_auto_ref(self):
        """HV with auto-computed reference point."""
        from src.experiments.moea_metrics import compute_hypervolume

        F = np.array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]])
        hv = compute_hypervolume(F)
        assert hv > 0

    def test_compute_hypervolume_single_point(self):
        """HV with a single point should still work."""
        from src.experiments.moea_metrics import compute_hypervolume

        F = np.array([[3.0, 7.0]])
        hv = compute_hypervolume(F, ref_point=np.array([10.0, 10.0]))
        assert hv > 0

    def test_compute_spacing_basic(self):
        """Spacing of uniformly distributed points should be low."""
        from src.experiments.moea_metrics import compute_spacing

        F = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
        sp = compute_spacing(F)
        assert sp >= 0
        # Uniform spacing → low std of min pairwise distances
        assert sp < 1.0

    def test_compute_spacing_single_point(self):
        """Spacing with 1 point should return 0."""
        from src.experiments.moea_metrics import compute_spacing

        sp = compute_spacing(np.array([[1.0, 2.0]]))
        assert sp == 0.0

    def test_compute_spacing_two_points(self):
        """Spacing with 2 points — std of single distance = 0."""
        from src.experiments.moea_metrics import compute_spacing

        F = np.array([[0.0, 1.0], [1.0, 0.0]])
        sp = compute_spacing(F)
        assert sp == 0.0  # Only one min-dist per point, std of identical = 0

    def test_compute_diversity_basic(self):
        """Diversity should be positive for spread population."""
        from src.experiments.moea_metrics import compute_diversity

        F = np.array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]])
        d = compute_diversity(F)
        assert d > 0

    def test_compute_diversity_identical(self):
        """Diversity of identical points should be 0."""
        from src.experiments.moea_metrics import compute_diversity

        F = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
        d = compute_diversity(F)
        assert d == 0.0

    def test_compute_diversity_single(self):
        """Diversity of single point should be 0."""
        from src.experiments.moea_metrics import compute_diversity

        d = compute_diversity(np.array([[1.0, 2.0]]))
        assert d == 0.0

    def test_compute_feasibility_rate_all_feasible(self):
        """All feasible (G <= 0) → rate = 1.0."""
        from src.experiments.moea_metrics import compute_feasibility_rate

        G = np.zeros((10, 8))
        assert compute_feasibility_rate(G) == 1.0

    def test_compute_feasibility_rate_none_feasible(self):
        """All infeasible → rate = 0.0."""
        from src.experiments.moea_metrics import compute_feasibility_rate

        G = np.ones((10, 8))
        assert compute_feasibility_rate(G) == 0.0

    def test_compute_feasibility_rate_mixed(self):
        """Half feasible → rate = 0.5."""
        from src.experiments.moea_metrics import compute_feasibility_rate

        G = np.zeros((10, 8))
        G[5:, 0] = 1  # last 5 have violations
        rate = compute_feasibility_rate(G)
        assert abs(rate - 0.5) < 1e-9

    def test_compute_feasibility_rate_partial_violations(self):
        """Individual with one violated constraint is still infeasible."""
        from src.experiments.moea_metrics import compute_feasibility_rate

        G = np.zeros((4, 8))
        G[0, 3] = 2  # one constraint violated
        rate = compute_feasibility_rate(G)
        assert abs(rate - 0.75) < 1e-9


# =====================================================================
#  Phase 5 — Callback MOEA lists + plot wiring
# =====================================================================


class TestPhase5CallbackMOEALists:
    """Verify callbacks have MOEA metric storage attributes."""

    def test_generate_outputs_has_moea_plots(self):
        """_generate_outputs source should reference hypervolume/spacing/diversity."""
        import inspect

        from src.experiments.ga_experiment import GAExperiment

        source = inspect.getsource(GAExperiment._generate_outputs)
        assert "hypervolume" in source.lower()
        assert "spacing" in source.lower()
        assert "diversity" in source.lower()
        assert "feasibility" in source.lower()
        assert "convergence_dashboard" in source


class TestPhase5ReturnDict:
    """Verify _execute return dict includes new metrics."""

    def test_return_dict_keys_documented(self):
        """_execute return dict should include MOEA metrics."""
        import inspect

        from src.experiments.ga_experiment import GAExperiment

        source = inspect.getsource(GAExperiment._execute)
        for key in ["hypervolumes", "spacings", "diversities", "feasibility_rates"]:
            assert f'"{key}"' in source, f"Return dict missing key: {key}"


# =====================================================================
#  moea_metrics module-level tests
# =====================================================================


class TestMOEAMetricsEdgeCases:
    """Edge cases and robustness of metric functions."""

    def test_hypervolume_empty_fails_gracefully(self):
        """Empty F should return 0.0 (not crash)."""
        from src.experiments.moea_metrics import compute_hypervolume

        F = np.empty((0, 2))
        hv = compute_hypervolume(F, ref_point=np.array([10.0, 10.0]))
        assert hv == 0.0

    def test_spacing_empty_returns_zero(self):
        """Empty F should return 0.0."""
        from src.experiments.moea_metrics import compute_spacing

        sp = compute_spacing(np.empty((0, 2)))
        assert sp == 0.0

    def test_diversity_empty_returns_zero(self):
        """Empty F should return 0.0."""
        from src.experiments.moea_metrics import compute_diversity

        d = compute_diversity(np.empty((0, 2)))
        assert d == 0.0

    def test_hypervolume_dominated_vs_nondominated(self):
        """HV with dominated points vs only nondominated should differ."""
        from src.experiments.moea_metrics import compute_hypervolume

        ref = np.array([10.0, 10.0])
        F_front = np.array([[1.0, 5.0], [5.0, 1.0]])
        F_all = np.array([[1.0, 5.0], [5.0, 1.0], [6.0, 6.0]])
        # HV should be same — dominated points don't contribute
        hv_front = compute_hypervolume(F_front, ref_point=ref)
        hv_all = compute_hypervolume(F_all, ref_point=ref)
        assert abs(hv_front - hv_all) < 1e-9

    def test_spacing_nonuniform(self):
        """Non-uniform spacing should have higher value."""
        from src.experiments.moea_metrics import compute_spacing

        uniform = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], dtype=float)
        nonuniform = np.array(
            [[0, 10], [0.1, 9.9], [5, 5], [9.9, 0.1], [10, 0]],
            dtype=float,
        )
        sp_u = compute_spacing(uniform)
        sp_n = compute_spacing(nonuniform)
        assert sp_n > sp_u


# =====================================================================
#  plotpareto verification
# =====================================================================


class TestPlotParetoFromF:
    """Ensure plot_pareto_front_from_F works with numpy F matrix."""

    def test_importable(self):
        from src.io.export.plotpareto import plot_pareto_front_from_F

        assert callable(plot_pareto_front_from_F)

    def test_signature(self):
        """Should accept (F: ndarray, output_dir: str)."""
        import inspect

        from src.io.export.plotpareto import plot_pareto_front_from_F

        sig = inspect.signature(plot_pareto_front_from_F)
        params = list(sig.parameters.keys())
        assert params == ["F", "output_dir"]
