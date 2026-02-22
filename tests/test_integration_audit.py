"""Integration tests for the audit improvements.

These tests run actual (tiny) GA experiments to verify end-to-end
correctness of:
- MOEA metric collection in callbacks
- Plot file generation (HV, spacing, diversity, feasibility)
- export_pdf=False skipping PDFs
- NaN guard in _evaluate

Requires: events_with_domains.pkl + data/ directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Skip all tests if pkl or data doesn't exist
_PKL = PROJECT_ROOT / "events_with_domains.pkl"
_DATA = PROJECT_ROOT / "data"
_SKIP_MSG = "Requires events_with_domains.pkl and data/ directory"
pytestmark = pytest.mark.skipif(
    not (_PKL.exists() and _DATA.exists()), reason=_SKIP_MSG
)


# =====================================================================
#  Helpers
# =====================================================================


def _run_baseline(
    tmp_path: Path, *, ngen: int = 12, pop_size: int = 10, export_pdf: bool = False
) -> dict:
    """Run a tiny BaselineExperiment and return results dict."""
    from src.experiments.ga_experiment import BaselineExperiment

    exp = BaselineExperiment(
        pop_size=pop_size,
        ngen=ngen,
        seed=42,
        output_dir=str(tmp_path),
        export_pdf=export_pdf,
        verbose=False,
    )
    return exp.run()


# =====================================================================
#  Integration: MOEA Metric Collection
# =====================================================================


class TestIntegrationMOEAMetrics:
    """End-to-end: callback collects MOEA metrics during optimization."""

    def test_moea_metrics_populated(self, tmp_path):
        """After 12 gens, metrics should have at least 1 entry (interval=10)."""
        result = _run_baseline(tmp_path, ngen=12)

        # HV/spacing/diversity/feasibility should each have ≥1 entry
        assert len(result["hypervolumes"]) >= 1, "No hypervolume recorded"
        assert len(result["spacings"]) >= 1, "No spacing recorded"
        assert len(result["diversities"]) >= 1, "No diversity recorded"
        assert len(result["feasibility_rates"]) >= 1, "No feasibility_rate recorded"

    def test_moea_metrics_types(self, tmp_path):
        """MOEA metric values should be floats (may be nan for feasible-only)."""
        result = _run_baseline(tmp_path, ngen=12)

        # Feasibility rates are always finite (computed on ALL individuals)
        for val in result["feasibility_rates"]:
            assert isinstance(val, float), "feasibility_rates value is not float"
            assert np.isfinite(val), f"feasibility_rates not finite: {val}"

        # HV/spacing/diversity may be nan when no feasible solutions exist
        for key in ["hypervolumes", "spacings", "diversities"]:
            for val in result[key]:
                assert isinstance(val, float), f"{key} value is not float"

    def test_feasibility_rate_range(self, tmp_path):
        """Feasibility rates should be in [0, 1]."""
        result = _run_baseline(tmp_path, ngen=12)
        for rate in result["feasibility_rates"]:
            assert 0.0 <= rate <= 1.0, f"Feasibility rate out of [0,1]: {rate}"

    def test_hypervolume_nonnegative_or_nan(self, tmp_path):
        """HV should be >= 0 or nan (no feasible solutions)."""
        result = _run_baseline(tmp_path, ngen=12)
        for hv in result["hypervolumes"]:
            assert hv >= 0 or np.isnan(hv), f"Unexpected HV: {hv}"

    def test_convergence_series_length(self, tmp_path):
        """Convergence lists should have exactly ngen entries."""
        ngen = 12
        result = _run_baseline(tmp_path, ngen=ngen)
        assert len(result["convergence_hard"]) == ngen
        assert len(result["convergence_soft"]) == ngen
        assert len(result["convergence_constraints"]) == ngen


# =====================================================================
#  Integration: export_pdf Flag
# =====================================================================


class TestIntegrationExportPDF:
    """End-to-end: export_pdf=False should skip PDFs."""

    def test_no_pdf_when_disabled(self, tmp_path):
        """With export_pdf=False, no PDF files should exist."""
        _run_baseline(tmp_path, ngen=5, export_pdf=False)

        pdf_files = list(tmp_path.rglob("*.pdf"))
        # Plot PDFs (from matplotlib save_figure) are still generated for
        # convergence/pareto plots.  But schedule PDFs (calendar, instructor,
        # room) should NOT be generated.
        schedule_pdfs = [
            p
            for p in pdf_files
            if any(
                kw in p.name.lower()
                for kw in [
                    "calendar",
                    "instructor_schedule",
                    "room_schedule",
                    "group_schedule",
                ]
            )
        ]
        assert (
            len(schedule_pdfs) == 0
        ), f"Schedule PDFs generated despite export_pdf=False: {schedule_pdfs}"

    def test_plots_still_generated_when_pdf_disabled(self, tmp_path):
        """Convergence/pareto plots should still be generated."""
        _run_baseline(tmp_path, ngen=5, export_pdf=False)

        # Check that at least some plot files were generated
        all_plots = list(tmp_path.rglob("*.pdf")) + list(tmp_path.rglob("*.png"))
        assert len(all_plots) > 0, "No plot files generated at all"


# =====================================================================
#  Integration: Plot File Generation
# =====================================================================


class TestIntegrationPlotFiles:
    """End-to-end: verify expected plot files are created."""

    def test_pareto_front_csv_created(self, tmp_path):
        """pareto_front.csv + population_fitness.csv should exist."""
        _run_baseline(tmp_path, ngen=5, export_pdf=False)

        csv_files = list(tmp_path.rglob("*.csv"))
        csv_names = {p.name for p in csv_files}
        assert (
            "pareto_front.csv" in csv_names
        ), f"pareto_front.csv not found in {csv_names}"
        assert (
            "population_fitness.csv" in csv_names
        ), f"population_fitness.csv not found in {csv_names}"

    def test_results_json_has_all_keys(self, tmp_path):
        """results.json should contain the expected keys."""
        _run_baseline(tmp_path, ngen=5, export_pdf=False)

        results_path = tmp_path / "results.json"
        assert results_path.exists(), "results.json not found"
        with open(results_path) as f:
            results = json.load(f)

        expected_keys = {
            "solver",
            "mode",
            "version",
            "config",
            "best_hard",
            "best_soft",
            "best_cv",
            "n_feasible",
            "elapsed_s",
            "sec_per_gen",
            "convergence_hard",
            "convergence_soft",
            "convergence_constraints",
            "hypervolumes",
            "spacings",
            "diversities",
            "feasibility_rates",
            "igds",
        }
        for key in expected_keys:
            assert key in results, f"Missing key in results.json: {key}"


# =====================================================================
#  Integration: NaN Guard (evaluate)
# =====================================================================


class TestIntegrationNaNGuard:
    """Test NaN guard through actual SchedulingProblem evaluation."""

    def test_evaluate_produces_finite_F(self, tmp_path):
        """_evaluate should never produce NaN/Inf in F."""
        from src.io.data_store import DataStore
        from src.io.time_system import QuantumTimeSystem
        from src.pipeline.scheduling_problem import create_problem

        pkl_path = str(_PKL)
        store = DataStore.from_json(str(_DATA), run_preflight=False)
        ctx = store.to_context()
        qts = QuantumTimeSystem()

        prob = create_problem(pkl_path, ctx=ctx, qts=qts)

        # Create a random population
        rng = np.random.default_rng(42)
        x = rng.integers(prob.xl, prob.xu + 1, size=(20, prob.n_var))

        out = {}
        prob._evaluate(x, out)

        F = out["F"]
        assert np.all(np.isfinite(F)), f"Non-finite values in F:\n{F[~np.isfinite(F)]}"
        assert F.shape == (20, 2)
        assert F[:, 0].min() >= 0, "Hard violations should be non-negative"

    def test_evaluate_G_integer_violations(self, tmp_path):
        """_evaluate should produce integer constraint violations >= 0."""
        from src.io.data_store import DataStore
        from src.io.time_system import QuantumTimeSystem
        from src.pipeline.scheduling_problem import create_problem

        pkl_path = str(_PKL)
        store = DataStore.from_json(str(_DATA), run_preflight=False)
        ctx = store.to_context()
        qts = QuantumTimeSystem()

        prob = create_problem(pkl_path, ctx=ctx, qts=qts)

        rng = np.random.default_rng(123)
        x = rng.integers(prob.xl, prob.xu + 1, size=(5, prob.n_var))

        out = {}
        prob._evaluate(x, out)

        G = out["G"]
        assert G.shape[1] == 8, "Should have 8 hard constraint columns"
        assert np.all(G >= 0), "Constraint violations should be non-negative"


# =====================================================================
#  Integration: Ruff lint check
# =====================================================================


class TestIntegrationLint:
    """Verify changed files pass linting."""

    def test_ruff_on_changed_files(self):
        """Run ruff on all files we modified."""
        import subprocess

        files = [
            "src/experiments/ga_experiment.py",
            "src/experiments/moea_metrics.py",
            "src/pipeline/scheduling_problem.py",
            "src/io/export/__init__.py",
            "src/io/export/plotpareto.py",
            "src/io/export/plot_igd.py",
            "tests/test_audit_improvements.py",
            "tests/test_dead_code.py",
            "tests/test_moea_quality.py",
        ]
        existing = [str(PROJECT_ROOT / f) for f in files if (PROJECT_ROOT / f).exists()]

        result = subprocess.run(
            ["python", "-m", "ruff", "check", "--select=E,F,W", *existing],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Ruff errors:\n{result.stdout}\n{result.stderr}"
