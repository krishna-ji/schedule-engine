#!/usr/bin/env python3
r"""Trajectory Autopsy — diagnose Soft-constraint degradation in RL training.

Parses ``step_log.csv`` and ``evaluation_trajectory.csv`` from the latest
(or specified) RL run.  Identifies:

1. **Soft spikes** — timesteps where ``best_soft`` *increased* (degraded).
2. **Hard spikes** — timesteps where ``best_hard`` *increased*.
3. **Per-action collateral profile** — for every action, the mean
   $\Delta\text{Hard}$ and $\Delta\text{Soft}$ it produced.

Usage::

    python runs/audit_rl_trajectory.py                         # latest run
    python runs/audit_rl_trajectory.py output/rl_vectorized/20260225_015107

Output:  Markdown table to stdout + ``output/trajectory_autopsy.md``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _find_latest_run() -> Path:
    """Return the most recent rl_vectorized run directory."""
    base = PROJECT_ROOT / "output" / "rl_vectorized"
    runs = sorted(base.iterdir()) if base.exists() else []
    if not runs:
        print("ERROR: No rl_vectorized runs found.", file=sys.stderr)
        sys.exit(1)
    return runs[-1]


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ======================================================================
# 1.  Step-log autopsy  (fine-grained, 100k+ rows)
# ======================================================================


def autopsy_step_log(run_dir: Path) -> str:
    step_csv = run_dir / "step_log.csv"
    if not step_csv.exists():
        return f"[WARNING] step_log.csv not found in {run_dir}\n"

    rows = _load_csv(step_csv)
    n = len(rows)

    # Parse into arrays
    ts = np.array([int(r["timestep"]) for r in rows])
    actions = np.array([int(r["action"]) for r in rows])
    rewards = np.array([float(r["reward"]) for r in rows])
    hard = np.array([float(r["best_hard"]) for r in rows])
    soft = np.array([float(r["best_soft"]) for r in rows])

    # Compute deltas (step i relative to step i-1)
    d_hard = np.diff(hard, prepend=hard[0])
    d_soft = np.diff(soft, prepend=soft[0])

    unique_actions = sorted(set(actions.tolist()))

    # ---- Per-action collateral profile ----
    lines: list[str] = []
    lines.append("## 1. Per-Action Collateral Profile (step_log.csv)")
    lines.append("")
    lines.append(f"Total timesteps: {n}")
    lines.append("")
    lines.append(
        "| Action | Count | Mean Delta_Hard | Mean Delta_Soft | Pct Hard+ | Pct Soft+ | Mean Reward |"
    )
    lines.append(
        "|:------:|------:|-----------:|-----------:|----------:|----------:|------------:|"
    )

    action_stats: dict[int, dict] = {}
    for a in unique_actions:
        mask = actions == a
        cnt = int(mask.sum())
        dh = d_hard[mask]
        ds = d_soft[mask]
        mean_dh = float(dh.mean())
        mean_ds = float(ds.mean())
        pct_hard_up = float((dh > 0).sum() / max(cnt, 1) * 100)
        pct_soft_up = float((ds > 0).sum() / max(cnt, 1) * 100)
        mean_r = float(rewards[mask].mean())

        action_stats[a] = {
            "count": cnt,
            "mean_dh": mean_dh,
            "mean_ds": mean_ds,
            "pct_hard_up": pct_hard_up,
            "pct_soft_up": pct_soft_up,
            "mean_r": mean_r,
        }
        lines.append(
            f"| {a} | {cnt} | {mean_dh:+.2f} | {mean_ds:+.2f} "
            f"| {pct_hard_up:.1f}% | {pct_soft_up:.1f}% | {mean_r:+.4f} |"
        )

    # ---- Worst soft spikes ----
    lines.append("")
    lines.append("## 2. Top 20 Worst Soft-Constraint Spikes")
    lines.append("")
    lines.append(
        "| Rank | Timestep | Action | Delta_Soft | Delta_Hard | Reward | Hard_before | Soft_before |"
    )
    lines.append(
        "|-----:|---------:|:------:|------:|------:|-------:|------------:|------------:|"
    )

    # Find indices where soft increased most
    spike_idx = np.argsort(-d_soft)[:20]  # descending by soft increase
    for rank, idx in enumerate(spike_idx, 1):
        if idx == 0:
            continue
        lines.append(
            f"| {rank} | {ts[idx]} | {actions[idx]} "
            f"| {d_soft[idx]:+.1f} | {d_hard[idx]:+.1f} "
            f"| {rewards[idx]:+.4f} "
            f"| {hard[idx - 1]:.1f} | {soft[idx - 1]:.1f} |"
        )

    # ---- Worst hard spikes ----
    lines.append("")
    lines.append("## 3. Top 20 Worst Hard-Constraint Spikes")
    lines.append("")
    lines.append(
        "| Rank | Timestep | Action | Delta_Hard | Delta_Soft | Reward | Hard_before | Soft_before |"
    )
    lines.append(
        "|-----:|---------:|:------:|------:|------:|-------:|------------:|------------:|"
    )

    spike_h_idx = np.argsort(-d_hard)[:20]
    for rank, idx in enumerate(spike_h_idx, 1):
        if idx == 0:
            continue
        lines.append(
            f"| {rank} | {ts[idx]} | {actions[idx]} "
            f"| {d_hard[idx]:+.1f} | {d_soft[idx]:+.1f} "
            f"| {rewards[idx]:+.4f} "
            f"| {hard[idx - 1]:.1f} | {soft[idx - 1]:.1f} |"
        )

    # ---- Oscillation detection ----
    lines.append("")
    lines.append("## 4. Oscillation Detection")
    lines.append("")

    # Count sign-changes in d_hard and d_soft
    hard_sign = np.sign(d_hard[1:])
    soft_sign = np.sign(d_soft[1:])
    hard_flips = int((np.diff(hard_sign) != 0).sum())
    soft_flips = int((np.diff(soft_sign) != 0).sum())

    lines.append(
        f"- Hard-penalty sign-flips: **{hard_flips}** / {n - 2} steps ({hard_flips / max(n - 2, 1) * 100:.1f}%)"
    )
    lines.append(
        f"- Soft-penalty sign-flips: **{soft_flips}** / {n - 2} steps ({soft_flips / max(n - 2, 1) * 100:.1f}%)"
    )
    lines.append("")

    # Net progress
    lines.append(
        f"- Hard: start={hard[0]:.1f} -> end={hard[-1]:.1f} (net Delta={hard[-1] - hard[0]:+.1f})"
    )
    lines.append(
        f"- Soft: start={soft[0]:.1f} -> end={soft[-1]:.1f} (net Delta={soft[-1] - soft[0]:+.1f})"
    )
    lines.append("")

    return "\n".join(lines)


# ======================================================================
# 2.  Evaluation trajectory autopsy  (per-gen, ~50 rows)
# ======================================================================


def autopsy_eval_trajectory(run_dir: Path) -> str:
    eval_csv = run_dir / "evaluation_trajectory.csv"
    if not eval_csv.exists():
        return f"[WARNING] evaluation_trajectory.csv not found in {run_dir}\n"

    rows = _load_csv(eval_csv)

    lines: list[str] = []
    lines.append("## 5. Evaluation Trajectory Per-Generation Detail")
    lines.append("")

    # Build constraint columns list
    constraint_cols = [c for c in rows[0].keys() if c.startswith("cv_")]
    hard_cols = [
        c
        for c in constraint_cols
        if c.split("_")[1] in ("CTE", "FTE", "SRE", "FPC", "FFC", "FCA", "CQF", "ICTD")
    ]
    soft_cols = [
        c for c in constraint_cols if c.split("_")[1] in ("CSC", "FSC", "MIP", "SSCP")
    ]

    lines.append(
        "| Gen | Action | BestHard | BestSoft | Delta_Hard | Delta_Soft | Reward |"
    )
    lines.append("|----:|:-------|--------:|---------:|------:|------:|-------:|")

    prev_hard = float(rows[0]["best_hard"])
    prev_soft = float(rows[0]["best_soft"])

    for r in rows:
        gen = r["generation"]
        act = r.get("action_name", r.get("action_id", "?"))
        bh = float(r["best_hard"])
        bs = float(r["best_soft"])
        dh = bh - prev_hard
        ds = bs - prev_soft
        rew = float(r["reward"])
        lines.append(
            f"| {gen} | {act} | {bh:.0f} | {bs:.1f} | {dh:+.1f} | {ds:+.1f} | {rew:+.4f} |"
        )
        prev_hard = bh
        prev_soft = bs

    # ---- Per-constraint trajectory (soft only — where damage is) ----
    if soft_cols:
        lines.append("")
        lines.append("### Soft Constraint Trajectory (Evaluation)")
        lines.append("")
        header = (
            "| Gen | Action | "
            + " | ".join(c.replace("cv_", "") for c in soft_cols)
            + " |"
        )
        sep = "|----:|:-------|" + "|".join(["------:" for _ in soft_cols]) + "|"
        lines.append(header)
        lines.append(sep)
        for r in rows:
            vals = " | ".join(f"{float(r.get(c, 0)):.1f}" for c in soft_cols)
            lines.append(
                f"| {r['generation']} | {r.get('action_name', '?')} | {vals} |"
            )

    lines.append("")
    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = _find_latest_run()

    print(f"Auditing run: {run_dir.name}")
    print("=" * 60)

    report_parts: list[str] = []
    report_parts.append("# RL Trajectory Autopsy")
    report_parts.append("")
    report_parts.append(f"**Run**: `{run_dir.name}`")
    report_parts.append("")

    report_parts.append(autopsy_step_log(run_dir))
    report_parts.append(autopsy_eval_trajectory(run_dir))

    report = "\n".join(report_parts)
    print(report)

    out_path = PROJECT_ROOT / "output" / "trajectory_autopsy.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
