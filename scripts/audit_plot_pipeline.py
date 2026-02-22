#!/usr/bin/env python
"""Audit script: measures plot/export vs eval/repair cost in a micro-run.

Runs 2 gens, pop=10 and records:
  - Number of plot/export calls
  - Number of decoder / Timetable builds
  - Total time in plotting/export vs evaluation vs repair
  - Per-function cProfile hotspots

Usage:
    python scripts/audit_plot_pipeline.py
"""
from __future__ import annotations

import cProfile
import functools
import io
import logging
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.logging_config import quick_setup

logger = logging.getLogger(__name__)

# ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── monkey-patch counters ─────────────────────────────────────────────
_CALL_LOG: dict[str, list[float]] = defaultdict(list)


def _wrap(module_path: str, func_name: str):
    """Dynamically patch a function to record call count + wall time."""
    parts = module_path.rsplit(".", 1)
    mod = __import__(parts[0], fromlist=[parts[1]])
    obj = getattr(mod, parts[1])
    original = getattr(obj, func_name) if hasattr(obj, func_name) else None
    if original is None:
        original = getattr(mod, func_name, None)
        obj = mod
    if original is None:
        return

    @functools.wraps(original)
    def wrapper(*a: Any, **kw: Any) -> Any:
        t0 = time.perf_counter()
        result = original(*a, **kw)
        dt = time.perf_counter() - t0
        key = f"{module_path}.{func_name}"
        _CALL_LOG[key].append(dt)
        return result

    setattr(obj, func_name, wrapper)


# Patch plot / export functions
_wrap("src.io.export.plothard", "plot_hard_constraint_violation_over_generation")
_wrap("src.io.export.plotsoft", "plot_soft_constraint_violation_over_generation")
_wrap("src.io.export.plot_detailed_constraints", "plot_individual_hard_constraints")
_wrap("src.io.export.plot_convergence", "plot_convergence_rate")
_wrap("src.io.export.plotpareto", "plot_pareto_front_from_F")
_wrap("src.io.export.exporter", "export_everything")
_wrap("src.io.export.schedule_views", "generate_instructor_schedules_pdf")
_wrap("src.io.export.schedule_views", "generate_room_schedules_pdf")
_wrap("src.io.export.violation_reporter", "generate_violation_report")

# Patch decoder
_wrap("src.io.decoder", "decode_individual")

# Patch evaluation
_wrap("src.pipeline.fast_evaluator_vectorized", "fast_evaluate_hard_vectorized")
_wrap("src.pipeline.soft_evaluator_vectorized", "eval_soft_vectorized")


def main() -> None:
    import matplotlib as mpl

    mpl.use("Agg")

    from src.experiments.ga_experiment import BaselineExperiment

    out_dir = str(ROOT / "output" / "_audit_temp")

    # ── cProfile the full run ─────────────────────────────────────
    pr = cProfile.Profile()
    pr.enable()
    t_total_start = time.perf_counter()

    exp = BaselineExperiment(pop_size=10, ngen=2, seed=42, output_dir=out_dir)
    results = exp.run()

    t_total = time.perf_counter() - t_total_start
    pr.disable()

    # ── Print call log ────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info(" AUDIT: PLOT / EXPORT PIPELINE COST")
    logger.info("=" * 72)

    total_plot = 0.0
    total_eval = 0.0
    total_decode = 0.0
    total_other = 0.0

    logger.info("%-65s %5s %9s %9s", "Function", "Calls", "Total(s)", "Avg(ms)")
    logger.info("%s", "-" * 92)

    for key in sorted(_CALL_LOG):
        calls = _CALL_LOG[key]
        n = len(calls)
        tot = sum(calls)
        avg_ms = (tot / n * 1000) if n else 0

        if (
            "plot" in key
            or "export" in key
            or "schedule_views" in key
            or "violation" in key
        ):
            total_plot += tot
            tag = "[PLOT]"
        elif "evaluate" in key or "eval_soft" in key:
            total_eval += tot
            tag = "[EVAL]"
        elif "decode" in key:
            total_decode += tot
            tag = "[DEC]"
        else:
            total_other += tot
            tag = "[?]"

        short = (
            key.replace("src.io.export.", "")
            .replace("src.pipeline.", "")
            .replace("src.io.", "")
        )
        logger.info("  %s %-58s %5d %9.3f %9.1f", tag, short, n, tot, avg_ms)

    logger.info("%s", "-" * 92)
    logger.info("  Total plotting/export:  %8.3fs", total_plot)
    logger.info("  Total evaluation:       %8.3fs", total_eval)
    logger.info("  Total decoding:         %8.3fs", total_decode)
    logger.info("  Wall clock:             %8.3fs", t_total)
    pct_plot = (total_plot / t_total * 100) if t_total > 0 else 0
    pct_eval = (total_eval / t_total * 100) if t_total > 0 else 0
    logger.info("  Plot/export fraction:   %7.1f%%", pct_plot)
    logger.info("  Eval fraction:          %7.1f%%", pct_eval)

    # Decoder calls
    decode_calls = len(_CALL_LOG.get("src.io.decoder.decode_individual", []))
    logger.info("  decode_individual calls: %d", decode_calls)
    logger.info("  (should be 1 — only best solution after run)")

    # ── cProfile top-20 ──────────────────────────────────────────
    logger.info("=" * 72)
    logger.info(" cProfile TOP-20 (cumulative)")
    logger.info("=" * 72)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(20)
    logger.info("%s", s.getvalue())

    # ── Cleanup ──────────────────────────────────────────────────
    import shutil

    shutil.rmtree(out_dir, ignore_errors=True)

    logger.info("Done. Temp output cleaned up.")


if __name__ == "__main__":
    quick_setup()
    main()
