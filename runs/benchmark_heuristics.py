#!/usr/bin/env python3
r"""Benchmark Heuristics — Brutal mathematical interrogation of the Elite 8.

For each of the 8 operators in the action space:
1. Create a **highly broken** population (random init, zero repair).
2. Evaluate baseline Hard / Soft / per-constraint violations.
3. Apply the operator exclusively for **1 iteration**.
4. Re-evaluate and compute the exact $\Delta$.
5. Print a strict Markdown table with PASS/FAIL verdicts.

Usage::

    python runs/benchmark_heuristics.py

Output:  Markdown table to stdout + ``output/benchmark_heuristics.md``.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_heuristics")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PKL_PATH = ".cache/events_with_domains.pkl"
POP_SIZE = 60          # smaller pop for speed; still statistically valid
SEED = 12345


def _create_broken_population(problem, pop_size: int, seed: int) -> np.ndarray:
    """Create a maximally broken population via raw random domain sampling.

    No repair is applied — every individual is essentially noise within
    the valid domain ranges.
    """
    from src.pipeline.pymoo_operators import RandomDomainSampling

    sampler = RandomDomainSampling(PKL_PATH)
    from pymoo.core.population import Population
    pop = sampler.do(problem, pop_size)
    return pop.get("X").astype(np.int64)


def _evaluate(problem, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a population matrix through the scheduling problem.

    Returns (F, G) where F=(N,2), G=(N,8).
    """
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population
    pop = Population.new("X", X)
    Evaluator().eval(problem, pop)
    F = pop.get("F")
    G = pop.get("G")
    if G is None:
        G = np.zeros((F.shape[0], 8), dtype=np.float64)
    return F, G


def main() -> None:
    from src.pipeline.scheduling_problem import create_problem
    from src.rl.actions.vectorized_ops import (
        VECTORIZED_ACTION_SPACE,
        ACTION_NAMES,
        NUM_ACTIONS,
    )
    from src.rl.gym_env.fast_state_encoder import (
        HARD_CONSTRAINT_NAMES,
        SOFT_CONSTRAINT_NAMES,
    )

    print(f"Creating scheduling problem from {PKL_PATH} ...")
    problem = create_problem(PKL_PATH)

    print(f"Generating broken population (N={POP_SIZE}, seed={SEED}) ...")
    X_base = _create_broken_population(problem, POP_SIZE, SEED)

    print("Evaluating baseline ...")
    F_base, G_base = _evaluate(problem, X_base)
    soft_bd_base = getattr(problem, "_last_soft_breakdown", None)

    base_hard = float(F_base[:, 0].mean())
    base_soft = float(F_base[:, 1].mean())

    # Per-hard-constraint baseline means
    base_cv_hard = {}
    for i, name in enumerate(HARD_CONSTRAINT_NAMES):
        if i < G_base.shape[1]:
            base_cv_hard[name] = float(G_base[:, i].mean())
        else:
            base_cv_hard[name] = 0.0

    # Per-soft-constraint baseline means
    base_cv_soft = {}
    for name in SOFT_CONSTRAINT_NAMES:
        if soft_bd_base and name in soft_bd_base:
            base_cv_soft[name] = float(np.asarray(soft_bd_base[name]).mean())
        else:
            base_cv_soft[name] = 0.0

    print(f"Baseline: Hard={base_hard:.1f}  Soft={base_soft:.1f}\n")

    # ──────────────────────────────────────────────────────────────
    # Interrogate each operator
    # ──────────────────────────────────────────────────────────────
    results: list[dict] = []

    for action_id in range(NUM_ACTIONS):
        cls = VECTORIZED_ACTION_SPACE[action_id]
        op = cls(PKL_PATH)
        name = ACTION_NAMES[action_id]

        # Copy and apply
        X_copy = X_base.copy()
        t0 = time.perf_counter()
        repaired = op._do(problem, X_copy)
        dt = time.perf_counter() - t0

        # Re-evaluate
        F_after, G_after = _evaluate(problem, repaired)
        soft_bd_after = getattr(problem, "_last_soft_breakdown", None)

        after_hard = float(F_after[:, 0].mean())
        after_soft = float(F_after[:, 1].mean())

        # Per-constraint deltas
        delta_cv_hard = {}
        for i, cname in enumerate(HARD_CONSTRAINT_NAMES):
            if i < G_after.shape[1]:
                delta_cv_hard[cname] = float(G_after[:, i].mean()) - base_cv_hard.get(cname, 0.0)
            else:
                delta_cv_hard[cname] = 0.0

        delta_cv_soft = {}
        for cname in SOFT_CONSTRAINT_NAMES:
            if soft_bd_after and cname in soft_bd_after:
                delta_cv_soft[cname] = float(np.asarray(soft_bd_after[cname]).mean()) - base_cv_soft.get(cname, 0.0)
            else:
                delta_cv_soft[cname] = 0.0

        delta_hard = after_hard - base_hard
        delta_soft = after_soft - base_soft

        # Verdict logic — refined per operator role
        #   0-2 : targeted hard repairs  → must reduce ΔHard
        #   3   : soft-constraint sync   → must reduce ΔSSCP
        #   4   : full pipeline (stochastic) → must show *some* reduction
        #   5-6 : perturbations          → any measurable Δ
        #   7   : soft optimization      → must improve soft metrics
        if action_id <= 2:
            # Pure hard-constraint repairs
            verdict = "PASS" if delta_hard < -0.5 else "**FAILED**"
        elif action_id == 3:
            # SymmetricSubcohortSync targets SSCP (soft)
            verdict = "PASS" if delta_cv_soft.get("SSCP", 0.0) < -0.5 else "**FAILED**"
        elif action_id == 4:
            # UniversalFeasibilityProjection — stochastic full pipeline
            # Must show meaningful total delta (either hard or soft)
            total_delta = abs(delta_hard) + abs(delta_soft)
            verdict = "PASS" if total_delta > 1.0 else "**FAILED**"
        elif action_id == 7:
            # MeridianCompaction — targets MIP/CSC soft metrics
            soft_improved = (delta_cv_soft.get("MIP", 0.0) < -0.1
                             or delta_cv_soft.get("CSC", 0.0) < -0.1)
            verdict = "PASS" if soft_improved else "**FAILED**"
        else:
            # Perturbations (5-6): any measurable change = PASS
            any_change = abs(delta_hard) > 0.1 or abs(delta_soft) > 0.1
            verdict = "PASS" if any_change else "**FAILED**"

        results.append({
            "id": action_id,
            "name": name,
            "delta_hard": delta_hard,
            "delta_soft": delta_soft,
            "delta_SRE": delta_cv_hard.get("SRE", 0.0),
            "delta_FTE": delta_cv_hard.get("FTE", 0.0),
            "delta_CTE": delta_cv_hard.get("CTE", 0.0),
            "delta_SSCP": delta_cv_soft.get("SSCP", 0.0),
            "delta_MIP": delta_cv_soft.get("MIP", 0.0),
            "delta_CSC": delta_cv_soft.get("CSC", 0.0),
            "time_ms": dt * 1000,
            "verdict": verdict,
        })

        print(f"  [{action_id}] {name:40s} ΔHard={delta_hard:+8.1f}  ΔSoft={delta_soft:+8.1f}  {verdict}  ({dt*1000:.0f}ms)")

    # ──────────────────────────────────────────────────────────────
    # Generate Markdown table
    # ──────────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("")
    lines.append("# Elite 8 Operator Benchmark")
    lines.append("")
    lines.append(f"Population: N={POP_SIZE}, baseline Hard={base_hard:.1f}, Soft={base_soft:.1f}")
    lines.append("")
    lines.append("| ID | Operator | ΔHard | ΔSoft | ΔSRE | ΔFTE | ΔCTE | ΔSSCP | ΔMIP | ΔCSC | Time(ms) | Verdict |")
    lines.append("|:--:|:---------|------:|------:|-----:|-----:|-----:|------:|-----:|-----:|---------:|:-------:|")

    for r in results:
        lines.append(
            f"| {r['id']} "
            f"| {r['name']:40s} "
            f"| {r['delta_hard']:+8.1f} "
            f"| {r['delta_soft']:+8.1f} "
            f"| {r['delta_SRE']:+7.1f} "
            f"| {r['delta_FTE']:+7.1f} "
            f"| {r['delta_CTE']:+7.1f} "
            f"| {r['delta_SSCP']:+7.1f} "
            f"| {r['delta_MIP']:+7.1f} "
            f"| {r['delta_CSC']:+7.1f} "
            f"| {r['time_ms']:8.0f} "
            f"| {r['verdict']} |"
        )

    lines.append("")
    lines.append("**Verdict criteria**:")
    lines.append("- Repairs (0–2): ΔHard < −0.5")
    lines.append("- Sync (3): ΔSSCP < −0.5")
    lines.append("- Pipeline (4): |ΔHard| + |ΔSoft| > 1.0")
    lines.append("- Perturbations (5–6): |Δ| > 0.1")
    lines.append("- Optimization (7): ΔMIP or ΔCSC < −0.1")
    lines.append("")

    md = "\n".join(lines)
    print(md)

    # Write to file
    out_path = PROJECT_ROOT / "output" / "benchmark_heuristics.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
