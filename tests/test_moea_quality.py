"""Tests for MOEA metric quality fixes (Phase 6).

Covers:
- Adaptive HV reference point (update_ref_point_max)
- Feasible-only metric policy (filter_feasible)
- IGD computation and reference front loading
- encode_availability quantum-boundary rounding
- IGD plot module existence
"""

from __future__ import annotations

import numpy as np
import pytest

# =====================================================================
#  moea_metrics: adaptive ref_point
# =====================================================================


class TestAdaptiveRefPoint:
    """Verify update_ref_point_max tracks and scales correctly."""

    def test_first_call_sets_running_max(self):
        from src.experiments.moea_metrics import update_ref_point_max

        F = np.array([[10.0, 200.0], [5.0, 300.0]])
        rmax, rp = update_ref_point_max(None, F, margin=1.1)
        np.testing.assert_array_equal(rmax, [10.0, 300.0])
        np.testing.assert_allclose(rp, [11.0, 330.0])

    def test_running_max_accumulates(self):
        from src.experiments.moea_metrics import update_ref_point_max

        F1 = np.array([[10.0, 200.0]])
        rmax, _ = update_ref_point_max(None, F1)
        F2 = np.array([[5.0, 400.0]])
        rmax, rp = update_ref_point_max(rmax, F2, margin=1.1)
        # max should be element-wise: [10, 400]
        np.testing.assert_array_equal(rmax, [10.0, 400.0])
        np.testing.assert_allclose(rp, [11.0, 440.0])

    def test_zero_column_gets_safe_value(self):
        from src.experiments.moea_metrics import update_ref_point_max

        F = np.array([[0.0, 100.0], [0.0, 50.0]])
        _, rp = update_ref_point_max(None, F, margin=1.1)
        # First column max = 0 → safe_max = 1.0 → ref = 1.1
        assert rp[0] == pytest.approx(1.1)
        assert rp[1] == pytest.approx(110.0)

    def test_hv_with_adaptive_ref_is_positive(self):
        from src.experiments.moea_metrics import (
            compute_hypervolume,
            update_ref_point_max,
        )

        F = np.array([[1.0, 10.0], [2.0, 5.0], [3.0, 3.0]])
        _, rp = update_ref_point_max(None, F, margin=1.1)
        hv = compute_hypervolume(F, ref_point=rp)
        assert hv > 0


# =====================================================================
#  moea_metrics: feasible-only filtering
# =====================================================================


class TestFeasibleOnlyPolicy:
    """Verify filter_feasible correctly selects feasible rows."""

    def test_all_feasible(self):
        from src.experiments.moea_metrics import filter_feasible

        F = np.array([[1.0, 2.0], [3.0, 4.0]])
        G = np.array([[0, 0, 0], [0, 0, 0]])
        result = filter_feasible(F, G)
        assert result is not None
        assert result.shape == (2, 2)

    def test_none_feasible(self):
        from src.experiments.moea_metrics import filter_feasible

        F = np.array([[1.0, 2.0], [3.0, 4.0]])
        G = np.array([[1, 0, 0], [0, 2, 0]])
        result = filter_feasible(F, G)
        assert result is None

    def test_partial_feasible(self):
        from src.experiments.moea_metrics import filter_feasible

        F = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        G = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        result = filter_feasible(F, G)
        assert result is not None
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1.0, 2.0], [5.0, 6.0]])

    def test_feasibility_rate_uses_all(self):
        """feasibility_rate should use ALL individuals, not just feasible."""
        from src.experiments.moea_metrics import compute_feasibility_rate

        G = np.array([[0, 0], [1, 0], [0, 0], [0, 2]])
        rate = compute_feasibility_rate(G)
        assert rate == pytest.approx(0.5)


# =====================================================================
#  moea_metrics: IGD
# =====================================================================


class TestIGD:
    """Verify IGD computation and reference front loading."""

    def test_igd_basic(self):
        from src.experiments.moea_metrics import compute_igd

        ref = np.array([[0.0, 0.0], [1.0, 1.0]])
        approx = np.array([[0.1, 0.1], [0.9, 0.9]])
        igd = compute_igd(approx, ref)
        assert igd > 0
        assert np.isfinite(igd)

    def test_igd_perfect(self):
        from src.experiments.moea_metrics import compute_igd

        ref = np.array([[1.0, 2.0], [3.0, 4.0]])
        igd = compute_igd(ref, ref)
        assert igd == pytest.approx(0.0, abs=1e-10)

    def test_load_reference_front_npy(self, tmp_path):
        from src.experiments.moea_metrics import load_reference_front

        F = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "ref.npy"
        np.save(path, F)
        loaded = load_reference_front(path)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, F)

    def test_load_reference_front_csv(self, tmp_path):
        from src.experiments.moea_metrics import load_reference_front

        F = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "ref.csv"
        np.savetxt(path, F, delimiter=",")
        loaded = load_reference_front(path)
        assert loaded is not None
        np.testing.assert_allclose(loaded, F)

    def test_load_reference_front_missing(self, tmp_path):
        from src.experiments.moea_metrics import load_reference_front

        loaded = load_reference_front(tmp_path / "nonexistent.npy")
        assert loaded is None

    def test_load_reference_front_bad_ext(self, tmp_path):
        from src.experiments.moea_metrics import load_reference_front

        path = tmp_path / "ref.json"
        path.write_text("{}")
        loaded = load_reference_front(path)
        assert loaded is None


# =====================================================================
#  moea_metrics: edge cases
# =====================================================================


class TestMetricsEdgeCases:
    """Edge cases for HV/spacing/diversity with nan handling."""

    def test_hv_fallback_ref_point(self):
        """compute_hypervolume without explicit ref_point still works."""
        from src.experiments.moea_metrics import compute_hypervolume

        F = np.array([[1.0, 10.0], [2.0, 5.0]])
        hv = compute_hypervolume(F)
        assert hv > 0

    def test_spacing_single_point(self):
        from src.experiments.moea_metrics import compute_spacing

        F = np.array([[1.0, 2.0]])
        assert compute_spacing(F) == 0.0

    def test_diversity_single_point(self):
        from src.experiments.moea_metrics import compute_diversity

        F = np.array([[1.0, 2.0]])
        assert compute_diversity(F) == 0.0


# =====================================================================
#  ga_experiment: callback MOEA lists include IGD
# =====================================================================


class TestCallbackMOEALists:
    """Verify callback MOEA lists include IGD."""

    def test_result_dict_includes_igds(self):
        """GAExperiment._execute return dict should contain igds key."""
        # Just verify the key exists in the expected attributes
        # The return dict is built inside _execute; we verify the code
        # accesses callback.igds by reading the source
        import inspect

        from src.experiments.ga_experiment import GAExperiment

        src = inspect.getsource(GAExperiment._execute)
        assert '"igds"' in src


# =====================================================================
#  encode_availability: quantum-boundary rounding
# =====================================================================


class TestEncodeAvailabilityRounding:
    """Verify start times round UP, end times round DOWN."""

    def test_aligned_times_unchanged(self):
        """Times exactly on quantum boundaries should produce same result."""
        from src.io.data_loader import encode_availability
        from src.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        # Pick a day and its operating hours
        # Find an operational day
        op_day = None
        for d in qts.DAY_NAMES:
            if qts.operating_hours.get(d) is not None:
                op_day = d
                break
        assert op_day is not None

        start_str, end_str = qts.operating_hours[op_day]
        avail = {op_day: [{"start": start_str, "end": end_str}]}
        quanta = encode_availability(avail, qts)

        # Should include all quanta for that day
        expected = set(
            range(
                qts.day_quanta_offset[op_day],
                qts.day_quanta_offset[op_day] + qts.day_quanta_count[op_day],
            )
        )
        assert quanta == expected

    def test_midquantum_start_rounds_up(self):
        """Start at 13:45 with 60-min quanta should round up to 14:00 quantum."""
        from src.io.data_loader import encode_availability
        from src.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        assert qts.QUANTUM_MINUTES == 60, "Test assumes 60-min quanta"

        # Find a day with operating hours that include 13:00-16:00
        op_day = None
        for d in qts.DAY_NAMES:
            hours = qts.operating_hours.get(d)
            if hours is None:
                continue
            sh, _ = map(int, hours[0].split(":"))
            eh, _ = map(int, hours[1].split(":"))
            if sh <= 13 and eh >= 16:
                op_day = d
                break

        if op_day is None:
            pytest.skip("No day with 13:00-16:00 in operating hours")

        avail = {op_day: [{"start": "13:45", "end": "16:00"}]}
        quanta = encode_availability(avail, qts)

        # With ceiling, 13:45 → next quantum boundary at 14:00
        # So the available set should NOT include the 13:00 quantum
        start_minutes = int(qts.operating_hours[op_day][0].split(":")[0]) * 60
        q_13 = qts.day_quanta_offset[op_day] + (13 * 60 - start_minutes) // 60
        assert (
            q_13 not in quanta
        ), "Quantum at 13:00 should NOT be included for 13:45 start"

        # But 14:00 and 15:00 should be included
        q_14 = q_13 + 1
        q_15 = q_13 + 2
        assert q_14 in quanta
        assert q_15 in quanta

    def test_midquantum_end_rounds_down(self):
        """End at 15:30 with 60-min quanta: only quanta fully covered are included."""
        from src.io.data_loader import encode_availability
        from src.io.time_system import QuantumTimeSystem

        qts = QuantumTimeSystem()
        if qts.QUANTUM_MINUTES != 60:
            pytest.skip("Test assumes 60-min quanta")

        op_day = None
        for d in qts.DAY_NAMES:
            hours = qts.operating_hours.get(d)
            if hours is None:
                continue
            sh, _ = map(int, hours[0].split(":"))
            eh, _ = map(int, hours[1].split(":"))
            if sh <= 13 and eh >= 16:
                op_day = d
                break

        if op_day is None:
            pytest.skip("No suitable day")

        avail = {op_day: [{"start": "13:00", "end": "15:30"}]}
        quanta = encode_availability(avail, qts)

        # 13:00 start on boundary → included
        # 15:30 end: floor(15:30-boundary) → quantum at 15:00 NOT included
        #   because 15:30 // 60 = 15 from start, end_q = that quantum (exclusive)
        start_minutes = int(qts.operating_hours[op_day][0].split(":")[0]) * 60
        q_13 = qts.day_quanta_offset[op_day] + (13 * 60 - start_minutes) // 60
        q_14 = q_13 + 1
        q_15 = q_13 + 2
        assert q_13 in quanta
        assert q_14 in quanta
        # 15:30 → end_q = quantum at 15:00 (floor of 15.5h from day start)
        # So range(start_q, end_q) = range(q_13, q_15) = [q_13, q_14]
        assert q_15 not in quanta


# =====================================================================
#  IGD plot module
# =====================================================================


class TestIGDPlotModule:
    """Verify plot_igd module loads and function is callable."""

    def test_plot_igd_importable(self):
        from src.io.export.plot_igd import plot_igd_trend

        assert callable(plot_igd_trend)

    def test_plot_igd_in_export_init(self):
        from src.io.export import plot_igd_trend

        assert callable(plot_igd_trend)

    def test_plot_igd_empty_list(self, tmp_path):
        """Empty history should not crash."""
        from src.io.export.plot_igd import plot_igd_trend

        plot_igd_trend([], str(tmp_path))  # should return silently

    def test_plot_igd_all_nan(self, tmp_path):
        """All-NaN history should not crash."""
        from src.io.export.plot_igd import plot_igd_trend

        plot_igd_trend([float("nan")] * 5, str(tmp_path))
