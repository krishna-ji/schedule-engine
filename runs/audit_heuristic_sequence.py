#!/usr/bin/env python3
r"""Heuristic Sequence Destruction Test — prove the Whack-A-Mole effect.

Creates a broken population, then applies a carefully chosen sequence of
operators and logs the full Hard/Soft/per-constraint state after each step.

Test sequences (each starts from the same broken population):

**Sequence A — Hard→Soft→Hard (the destroyer)**
  1. Action 0 (SpatialResourceProjection) — fix room clashes
  2. Action 7 (MeridianCompaction) — optimise lunch/compaction
  3. Action 0 again — re-fix rooms

**Sequence B — All 3 hard repairs chained**
  1. Action 0 (SpatialResourceProjection)
  2. Action 1 (FacultyTemporalProjection)
  3. Action 2 (CohortTemporalProjection)

**Sequence C — Sync then hard repair**
  1. Action 3 (SymmetricSubcohortSync) — sync practicals
  2. Action 0 (SpatialResourceProjection) — fix the collateral

**Sequence D — Full pipeline then compaction**
  1. Action 4 (UniversalFeasibilityProjection) — nuclear repair
  2. Action 7 (MeridianCompaction) — soft optimisation

Goal: Mathematically prove whether soft heuristics completely undo
the topological structure built by hard heuristics.

Usage::

    python runs/audit_heuristic_sequence.py

Output:  Markdown table to stdout + ``output/sequence_destruction_test.md``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PKL_PATH = ".cache/events_with_domains.pkl"
POP_SIZE = 60
SEED = 12345


def _create_broken_population(problem, pop_size: int, seed: int) -> np.ndarray:
    from pymoo.core.population import Population

    from src.pipeline.pymoo_operators import RandomDomainSampling

    sampler = RandomDomainSampling(PKL_PATH)
    pop = sampler.do(problem, pop_size)
    return pop.get("X").astype(np.int64)


def _evaluate(problem, X: np.ndarray) -> dict:
    """Full evaluation returning a flat metrics dict."""
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population

    from src.rl.gym_env.fast_state_encoder import (
        HARD_CONSTRAINT_NAMES,
        SOFT_CONSTRAINT_NAMES,
    )

    pop = Population.new("X", X)
    Evaluator().eval(problem, pop)
    F = pop.get("F")
    G = pop.get("G")
    if G is None:
        G = np.zeros((F.shape[0], 8), dtype=np.float64)

    soft_bd = getattr(problem, "_last_soft_breakdown", None)

    m: dict[str, float] = {
        "best_hard": float(F[:, 0].min()),
        "mean_hard": float(F[:, 0].mean()),
        "best_soft": float(F[:, 1].min()),
        "mean_soft": float(F[:, 1].mean()),
    }
    for i, name in enumerate(HARD_CONSTRAINT_NAMES):
        m[name] = float(G[:, i].mean()) if i < G.shape[1] else 0.0
    for name in SOFT_CONSTRAINT_NAMES:
        if soft_bd and name in soft_bd:
            m[name] = float(np.asarray(soft_bd[name]).mean())
        else:
            m[name] = 0.0

    return m


def _compute_reward(
    prev_hard: float, prev_soft: float, cur_hard: float, cur_soft: float
) -> dict:
    """Replicate the exact reward formula from PymooHyperHeuristicEnv."""
    delta_hard = prev_hard - cur_hard  # >0 means improvement
    delta_soft = prev_soft - cur_soft
    norm_hard = max(prev_hard, 1.0)
    norm_soft = max(prev_soft, 1.0)
    hard_reward = delta_hard / norm_hard
    soft_reward = delta_soft / norm_soft
    raw = hard_reward + 0.1 * soft_reward
    reward = float(np.clip(raw, -5.0, 5.0))
    return {
        "delta_hard": delta_hard,
        "delta_soft": delta_soft,
        "hard_reward": hard_reward,
        "soft_reward": soft_reward,
        "raw_reward": raw,
        "clipped_reward": reward,
    }


def _run_sequence(
    problem,
    X_base: np.ndarray,
    sequence: list[tuple[int, str]],
    action_space: dict,
    seq_name: str,
) -> str:
    """Run a sequence of operators and return Markdown report."""
    from src.rl.gym_env.fast_state_encoder import (
        HARD_CONSTRAINT_NAMES,
        SOFT_CONSTRAINT_NAMES,
    )

    lines: list[str] = []
    lines.append(f"### {seq_name}")
    lines.append("")

    # Column keys for the table
    constraint_keys = ["SRE", "FTE", "CTE", "SSCP", "MIP", "CSC"]
    header = (
        "| Step | Action | BestHard | BestSoft | Delta_Hard | Delta_Soft | Reward | "
        + " | ".join(constraint_keys)
        + " |"
    )
    sep = (
        "|-----:|:-------|--------:|---------:|------:|------:|-------:|"
        + "|".join(["------:" for _ in constraint_keys])
        + "|"
    )
    lines.append(header)
    lines.append(sep)

    X = X_base.copy()
    m = _evaluate(problem, X)

    # Baseline row
    cv_vals = " | ".join(f"{m.get(k, 0):.1f}" for k in constraint_keys)
    lines.append(
        f"| 0 | *baseline* | {m['best_hard']:.0f} | {m['best_soft']:.1f} "
        f"| — | — | — | {cv_vals} |"
    )

    prev_hard = m["best_hard"]
    prev_soft = m["best_soft"]
    steps_detail: list[dict] = [{"step": 0, "action": "baseline", **m}]

    for step_num, (action_id, action_label) in enumerate(sequence, 1):
        cls = action_space[action_id]
        op = cls(PKL_PATH)
        X = op._do(problem, X)
        m = _evaluate(problem, X)
        r = _compute_reward(prev_hard, prev_soft, m["best_hard"], m["best_soft"])

        cv_vals = " | ".join(f"{m.get(k, 0):.1f}" for k in constraint_keys)
        dh = m["best_hard"] - prev_hard
        ds = m["best_soft"] - prev_soft
        lines.append(
            f"| {step_num} | {action_label} | {m['best_hard']:.0f} | {m['best_soft']:.1f} "
            f"| {dh:+.1f} | {ds:+.1f} | {r['clipped_reward']:+.4f} | {cv_vals} |"
        )

        steps_detail.append({"step": step_num, "action": action_label, **m, **r})
        prev_hard = m["best_hard"]
        prev_soft = m["best_soft"]

    # ---- Summary verdict ----
    total_dh = steps_detail[-1]["best_hard"] - steps_detail[0]["best_hard"]
    total_ds = steps_detail[-1]["best_soft"] - steps_detail[0]["best_soft"]
    lines.append("")
    lines.append(
        f"**Net effect**: Delta_Hard = {total_dh:+.1f}, Delta_Soft = {total_ds:+.1f}"
    )

    if total_dh > 0 and total_ds < 0:
        lines.append(
            "**Diagnosis**: Classic Whack-A-Mole — soft improvement destroys hard structure."
        )
    elif total_dh < 0 and total_ds > 0:
        lines.append(
            "**Diagnosis**: Hard improvement at soft cost — acceptable trade if hard-dominant."
        )
    elif total_dh > 0 and total_ds > 0:
        lines.append("**Diagnosis**: CATASTROPHIC — both objectives degraded.")
    else:
        lines.append("**Diagnosis**: Pareto-improving — both objectives improved.")

    lines.append("")
    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    from src.pipeline.scheduling_problem import create_problem
    from src.rl.actions.vectorized_ops import ACTION_NAMES, VECTORIZED_ACTION_SPACE

    print(f"Creating scheduling problem from {PKL_PATH} ...")
    problem = create_problem(PKL_PATH)

    print(f"Generating broken population (N={POP_SIZE}, seed={SEED}) ...")
    X_base = _create_broken_population(problem, POP_SIZE, SEED)

    # ---- Define sequences ----
    sequences = [
        (
            "Sequence A — Hard→Soft→Hard (Room Repair → Compaction → Room Repair)",
            [
                (0, "A0:SpatialResourceProjection"),
                (7, "A7:MeridianCompaction"),
                (0, "A0:SpatialResourceProjection"),
            ],
        ),
        (
            "Sequence B — All 3 Hard Repairs Chained",
            [
                (0, "A0:SpatialResourceProjection"),
                (1, "A1:FacultyTemporalProjection"),
                (2, "A2:CohortTemporalProjection"),
            ],
        ),
        (
            "Sequence C — Sync Then Hard Repair",
            [
                (3, "A3:SymmetricSubcohortSync"),
                (0, "A0:SpatialResourceProjection"),
            ],
        ),
        (
            "Sequence D — Full Pipeline Then Compaction",
            [
                (4, "A4:UniversalFeasibilityProjection"),
                (7, "A7:MeridianCompaction"),
            ],
        ),
        (
            "Sequence E — Triple Hard Then Compaction (Real-World Likely Sequence)",
            [
                (0, "A0:SpatialResourceProjection"),
                (1, "A1:FacultyTemporalProjection"),
                (2, "A2:CohortTemporalProjection"),
                (7, "A7:MeridianCompaction"),
            ],
        ),
    ]

    report_parts: list[str] = []
    report_parts.append("# Heuristic Sequence Destruction Test")
    report_parts.append("")
    report_parts.append(f"Population: N={POP_SIZE}, seed={SEED}")
    report_parts.append("")

    for seq_name, seq in sequences:
        print(f"\nRunning: {seq_name}")
        result = _run_sequence(problem, X_base, seq, VECTORIZED_ACTION_SPACE, seq_name)
        report_parts.append(result)
        print(result)

    report = "\n".join(report_parts)

    out_path = PROJECT_ROOT / "output" / "sequence_destruction_test.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
