#!/usr/bin/env python3
r"""Phase 56 — Static Baseline Comparison + LLH Differentiation Check.

Runs all 6 LLHs for 50 generations (pop=120, seed=42) and produces:
1. Per-generation trajectory log
2. Checkpoint comparison table at gen 5, 25, 50
3. Explicit answers to the 3 critical differentiation questions

Usage::

    python -m runs.rl_02_llh_differentiation
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ======================================================================
# Configuration
# ======================================================================

POP_SIZE = 120
MAX_GEN = 50
PKL_PATH = ".cache/events_with_domains.pkl"
SEED = 42
CHECKPOINTS = [5, 25, 50]

ACTION_SHORT = {
    0: "Conservative",
    1: "Aggressive",
    2: "Memetic",
    3: "SoftFocus",
    4: "Destructive",
    5: "Intensified",
}


def run_one(action_id: int) -> list[dict]:
    """Run one static baseline for MAX_GEN generations."""
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GEN,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
    )
    obs, info = env.reset()

    rows = [
        {
            "gen": 1,
            "best_hard": info["best_hard"],
            "best_soft": info["best_soft"],
            "mean_hard": info["mean_hard"],
            "mean_soft": info["mean_soft"],
            "step_time": 0.0,
            "n_repaired": 0,
        }
    ]

    for g in range(MAX_GEN - 1):
        obs, reward, done, trunc, info = env.step(action_id)
        rows.append(
            {
                "gen": info["generation"],
                "best_hard": info["best_hard"],
                "best_soft": info["best_soft"],
                "mean_hard": info["mean_hard"],
                "mean_soft": info["mean_soft"],
                "step_time": info.get("step_time_s", 0.0),
                "n_repaired": info.get("n_repaired", 0),
            }
        )
        if done or trunc:
            break

    env.close()
    return rows


def main():
    from src.rl.actions.vectorized_ops import ACTION_NAMES, NUM_ACTIONS

    print("=" * 80)
    print("  Phase 56: Static Baseline Comparison + LLH Differentiation Check")
    print(f"  pop_size={POP_SIZE}  max_gen={MAX_GEN}  seed={SEED}")
    print("=" * 80)

    # ---- Run all 6 baselines ----
    all_data: dict[int, list[dict]] = {}
    total_t0 = time.perf_counter()

    for aid in range(NUM_ACTIONS):
        name = ACTION_NAMES[aid]
        short = ACTION_SHORT[aid]
        print(f"\n  Running Action {aid} ({short})...", end="", flush=True)
        t0 = time.perf_counter()
        rows = run_one(aid)
        dt = time.perf_counter() - t0
        all_data[aid] = rows
        final = rows[-1]
        print(
            f" hard={final['best_hard']:.0f}  soft={final['best_soft']:.0f}  "
            f"mean_hard={final['mean_hard']:.0f}  {dt:.1f}s"
        )

    total_time = time.perf_counter() - total_t0
    print(f"\n  Total time: {total_time:.1f}s")

    # ---- Build lookup: aid -> gen -> row ----
    lookup: dict[int, dict[int, dict]] = {}
    for aid, rows in all_data.items():
        lookup[aid] = {r["gen"]: r for r in rows}

    # ---- Per-generation trajectory ----
    print("\n\n" + "=" * 80)
    print("  PER-GENERATION TRAJECTORY (best_hard)")
    print("=" * 80)
    header = f"{'Gen':>4}"
    for aid in range(NUM_ACTIONS):
        header += f"  {ACTION_SHORT[aid]:>12}"
    print(header)
    print("-" * len(header))

    for g in range(1, MAX_GEN + 1):
        line = f"{g:>4}"
        for aid in range(NUM_ACTIONS):
            row = lookup[aid].get(g)
            val = row["best_hard"] if row else float("nan")
            line += f"  {val:>12.0f}"
        # Mark the winner
        vals = []
        for aid in range(NUM_ACTIONS):
            row = lookup[aid].get(g)
            vals.append(row["best_hard"] if row else float("inf"))
        winner = int(np.argmin(vals))
        line += f"  <- {ACTION_SHORT[winner]}"
        print(line)

    # ---- Checkpoint comparison table ----
    print("\n\n" + "=" * 100)
    print("  CHECKPOINT COMPARISON TABLE")
    print("=" * 100)

    col_headers = ["LLH"]
    for cp in CHECKPOINTS:
        col_headers += [f"Hard@{cp}", f"Soft@{cp}"]
    col_headers += ["TotalTime", "s/gen"]

    fmt = f"{'LLH':<16}"
    for cp in CHECKPOINTS:
        fmt += f"  {'Hard@'+str(cp):>8}  {'Soft@'+str(cp):>8}"
    fmt += f"  {'TotalTime':>10}  {'s/gen':>6}"
    print(fmt)
    print("-" * len(fmt))

    for aid in range(NUM_ACTIONS):
        short = ACTION_SHORT[aid]
        rows = all_data[aid]
        total_t = sum(r["step_time"] for r in rows)
        n_steps = sum(1 for r in rows if r["step_time"] > 0)
        spg = total_t / max(n_steps, 1)

        line = f"{short:<16}"
        for cp in CHECKPOINTS:
            row = lookup[aid].get(cp)
            if row:
                line += f"  {row['best_hard']:>8.0f}  {row['best_soft']:>8.0f}"
            else:
                line += f"  {'N/A':>8}  {'N/A':>8}"
        line += f"  {total_t:>10.1f}  {spg:>6.1f}"
        print(line)

    # ---- Find best-ever hard per LLH ----
    print("\n\n" + "=" * 80)
    print("  BEST-EVER HARD PENALTY PER LLH")
    print("=" * 80)
    print(
        f"{'LLH':<16}  {'BestHard':>8}  {'AtGen':>6}  {'BestSoft':>8}  {'SoftAtBestHard':>14}"
    )
    print("-" * 60)

    best_ever_per_llh = {}
    for aid in range(NUM_ACTIONS):
        short = ACTION_SHORT[aid]
        rows = all_data[aid]
        best_hard = min(r["best_hard"] for r in rows)
        best_gen = next(r["gen"] for r in rows if r["best_hard"] == best_hard)
        best_row = next(r for r in rows if r["gen"] == best_gen)
        best_ever_per_llh[aid] = (best_hard, best_gen)
        # Also find overall best soft
        best_soft = min(r["best_soft"] for r in rows)
        print(
            f"{short:<16}  {best_hard:>8.0f}  {best_gen:>6}  {best_soft:>8.0f}  {best_row['best_soft']:>14.0f}"
        )

    # ---- Soft constraint trajectory ----
    print("\n\n" + "=" * 80)
    print("  PER-GENERATION TRAJECTORY (best_soft)")
    print("=" * 80)
    header = f"{'Gen':>4}"
    for aid in range(NUM_ACTIONS):
        header += f"  {ACTION_SHORT[aid]:>12}"
    print(header)
    print("-" * len(header))

    for g in [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        line = f"{g:>4}"
        for aid in range(NUM_ACTIONS):
            row = lookup[aid].get(g)
            val = row["best_soft"] if row else float("nan")
            line += f"  {val:>12.0f}"
        print(line)

    # ---- Critical Questions ----
    print("\n\n" + "=" * 80)
    print("  CRITICAL DIFFERENTIATION ANALYSIS")
    print("=" * 80)

    # Q1: Is there any generation where a non-Conservative LLH wins?
    print("\n  Q1: Is there a generation where a non-Conservative LLH wins on hard?")
    conservative_wins = 0
    non_conservative_wins = {}
    ties = 0

    for g in range(1, MAX_GEN + 1):
        vals = {}
        for aid in range(NUM_ACTIONS):
            row = lookup[aid].get(g)
            if row:
                vals[aid] = row["best_hard"]
        if not vals:
            continue

        min_val = min(vals.values())
        winners = [aid for aid, v in vals.items() if v == min_val]

        if 0 in winners and len(winners) == 1:
            conservative_wins += 1
        elif 0 in winners:
            ties += 1
        else:
            for w in winners:
                non_conservative_wins.setdefault(w, []).append(g)

    print(f"    Conservative sole winner: {conservative_wins} / {MAX_GEN} generations")
    print(f"    Ties including Conservative: {ties} / {MAX_GEN} generations")
    if non_conservative_wins:
        print("    Non-Conservative wins:")
        for aid, gens in sorted(non_conservative_wins.items()):
            short = ACTION_SHORT[aid]
            gen_ranges = _summarize_ranges(gens)
            print(f"      {short}: {len(gens)} gens ({gen_ranges})")
        q1_answer = "YES"
    else:
        print("    No generation where a non-Conservative LLH wins alone.")
        q1_answer = "NO"
    print(f"    >>> ANSWER: {q1_answer}")

    # Q2: Do LLHs converge to different soft constraint values?
    print("\n  Q2: Do LLHs converge to different soft constraint values?")
    final_softs = {}
    for aid in range(NUM_ACTIONS):
        rows = all_data[aid]
        # Average of last 5 generations for stability
        last5 = [r["best_soft"] for r in rows[-5:]]
        final_softs[aid] = np.mean(last5)
        print(
            f"    {ACTION_SHORT[aid]:<16}: avg_soft(gen46-50) = {final_softs[aid]:.1f}"
        )

    soft_vals = list(final_softs.values())
    soft_range = max(soft_vals) - min(soft_vals)
    soft_cv = np.std(soft_vals) / max(np.mean(soft_vals), 1.0)
    print(f"    Range: {soft_range:.1f}  CV: {soft_cv:.3f}")
    q2_answer = "YES" if soft_range > 50 or soft_cv > 0.05 else "NO"
    print(f"    >>> ANSWER: {q2_answer} (range={soft_range:.1f}, threshold=50)")

    # Q3: Does any LLH escape a plateau others get stuck on?
    print("\n  Q3: Does any LLH escape a plateau that others get stuck on?")

    # Detect stagnation: hard doesn't improve for 10+ consecutive gens
    stagnation_info = {}
    for aid in range(NUM_ACTIONS):
        rows = all_data[aid]
        best_seen = rows[0]["best_hard"]
        stag_start = None
        max_stag = 0
        stag_at = None

        for r in rows[1:]:
            if r["best_hard"] < best_seen:
                best_seen = r["best_hard"]
                stag_start = None
            else:
                if stag_start is None:
                    stag_start = r["gen"]
                stag_len = r["gen"] - stag_start
                if stag_len > max_stag:
                    max_stag = stag_len
                    stag_at = (stag_start, r["gen"], best_seen)

        stagnation_info[aid] = (max_stag, stag_at)
        if stag_at:
            print(
                f"    {ACTION_SHORT[aid]:<16}: max stagnation = {max_stag} gens "
                f"(gen {stag_at[0]}-{stag_at[1]}, hard={stag_at[2]:.0f})"
            )
        else:
            print(f"    {ACTION_SHORT[aid]:<16}: no stagnation detected")

    # Check if any LLH broke through where others stalled
    # Compare: at the generation where most LLHs are stagnating,
    # does any LLH have a significantly lower hard?
    print("\n    Late-game comparison (gen 40-50 best_hard):")
    late_bests = {}
    for aid in range(NUM_ACTIONS):
        rows = all_data[aid]
        late = [r["best_hard"] for r in rows if r["gen"] >= 40]
        late_bests[aid] = min(late) if late else float("inf")
        print(
            f"      {ACTION_SHORT[aid]:<16}: min_hard(gen40-50) = {late_bests[aid]:.0f}"
        )

    late_vals = list(late_bests.values())
    best_late = min(late_vals)
    worst_late = max(late_vals)
    q3_answer = "YES" if (worst_late - best_late) > 10 else "NO"
    print(
        f"    Range: {worst_late - best_late:.0f} (best={best_late:.0f}, worst={worst_late:.0f})"
    )
    print(f"    >>> ANSWER: {q3_answer} (threshold: >10 hard difference)")

    # ---- Final verdict ----
    print("\n\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)
    any_yes = q1_answer == "YES" or q2_answer == "YES" or q3_answer == "YES"
    print(f"    Q1 (non-Conservative wins?):   {q1_answer}")
    print(f"    Q2 (different soft values?):    {q2_answer}")
    print(f"    Q3 (plateau escape?):           {q3_answer}")
    print()
    if any_yes:
        print(
            "    VERDICT: LLHs show differentiation. "
            "RL experiment is VIABLE. Proceed to PPO training."
        )
    else:
        print(
            "    VERDICT: Conservative dominates everywhere. "
            "LLH parameter space needs REDESIGN before PPO training."
        )
    print("=" * 80)


def _summarize_ranges(gens: list[int]) -> str:
    """Summarize a list of generation numbers into ranges."""
    if not gens:
        return ""
    gens = sorted(gens)
    ranges = []
    start = gens[0]
    prev = gens[0]
    for g in gens[1:]:
        if g == prev + 1:
            prev = g
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = g
            prev = g
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")
    return ", ".join(ranges)


if __name__ == "__main__":
    main()
