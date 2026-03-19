"""Comprehensive analysis of ALL runs from the other VM."""

import csv
import json
import os
import statistics
import sys

BASE = r"C:\Users\Administrator\Desktop\main-sch-engine\output\fromanothervm"


def read_stats(path):
    """Read stats.json if present."""
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def read_metadata(path):
    """Read metadata.json if present."""
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def read_feasibility(path):
    """Read feasibility.log if present."""
    if os.path.isfile(path):
        with open(path) as f:
            return f.read()
    return None


def find_best_run(ga_dir):
    """Find the most complete run in a GA directory (has stats.json + best_individual.json)."""
    if not os.path.isdir(ga_dir):
        return None
    best = None
    for d in sorted(os.listdir(ga_dir)):
        rpath = os.path.join(ga_dir, d)
        if os.path.isdir(rpath) and os.path.isfile(os.path.join(rpath, "stats.json")):
            best = rpath  # take the latest complete one
    return best


print("=" * 90)
print("  COMPREHENSIVE ANALYSIS OF ALL RUNS FROM OTHER VM")
print("=" * 90)

# ────────────────────────────────────────────────────────
# 1. GA BASELINES
# ────────────────────────────────────────────────────────
ga_variants = [
    ("ga_01_baseline", "GA-01: Baseline GA"),
    ("ga_02_memetic", "GA-02: Memetic GA"),
    ("ga_03_repair_sequential", "GA-03: Sequential Repair"),
    ("ga_04_repair_bandit", "GA-04: Bandit Repair"),
    ("ga_06_ultimate", "GA-06: Ultimate GA"),
]

print("\n" + "─" * 90)
print("  SECTION 1: GA BASELINES COMPARISON")
print("─" * 90)

ga_results = []
for folder, label in ga_variants:
    ga_dir = os.path.join(BASE, folder)
    run_path = find_best_run(ga_dir)
    if run_path is None:
        print(f"\n  {label}: No complete run found")
        continue

    stats = read_stats(os.path.join(run_path, "stats.json"))
    meta = read_metadata(os.path.join(run_path, "metadata.json"))
    feas_text = read_feasibility(os.path.join(run_path, "feasibility.log"))

    print(f"\n  ┌─ {label} ({os.path.basename(run_path)})")

    if meta:
        for k in ["algorithm", "pop_size", "generations", "elapsed_sec", "n_cores"]:
            if k in meta:
                val = meta[k]
                if k == "elapsed_sec":
                    val = f"{val:.0f}s ({val/60:.1f}m)"
                print(f"  │  {k}: {val}")

    if stats:
        print("  │")
        # Look for key metrics
        for k in [
            "best_hard_violations",
            "best_soft_penalty",
            "best_fitness",
            "final_hard_violations",
            "final_soft_penalty",
            "hard_violations",
            "soft_penalty",
            "feasible",
            "feasible_count",
        ]:
            if k in stats:
                print(f"  │  {k}: {stats[k]}")

        # Generational data
        if "generations" in stats and isinstance(stats["generations"], list):
            gens = stats["generations"]
            print(f"  │  Total generations logged: {len(gens)}")
            if gens:
                last = gens[-1]
                if isinstance(last, dict):
                    for k, v in last.items():
                        print(f"  │  Final gen: {k}={v}")

        # Print all top-level keys for visibility
        print(f"  │  Stats keys: {list(stats.keys())}")

        ga_results.append(
            {
                "label": label,
                "stats": stats,
                "meta": meta,
            }
        )

    if feas_text:
        # Print last 10 lines of feasibility log
        lines = feas_text.strip().split("\n")
        print("  │")
        print("  │  Feasibility log (last 5 lines):")
        for line in lines[-5:]:
            print(f"  │    {line.strip()}")

    print("  └─")


# ────────────────────────────────────────────────────────
# 2. RL TITAN V4 SOTA
# ────────────────────────────────────────────────────────
print("\n" + "─" * 90)
print("  SECTION 2: RL TITAN V4 SOTA")
print("─" * 90)

titan_base = os.path.join(BASE, "rl_titan_v4_sota", "20260315_193825")
if os.path.isdir(titan_base):
    # Training curve
    tc_path = os.path.join(titan_base, "training_curve.csv")
    if os.path.isfile(tc_path):
        with open(tc_path) as f:
            rows = list(csv.DictReader(f))
        rewards = [float(r["episode_reward"]) for r in rows]
        print(f"\n  Episodes: {len(rows)}")
        print(
            f"  Reward: min={min(rewards):.1f}, max={max(rewards):.1f}, mean={statistics.mean(rewards):.1f}"
        )
        print(f"  First 10 avg: {statistics.mean(rewards[:10]):.1f}")
        print(f"  Last 10 avg:  {statistics.mean(rewards[-10:]):.1f}")
        print(
            f"  Improvement:  {statistics.mean(rewards[-10:]) - statistics.mean(rewards[:10]):+.1f} ({(statistics.mean(rewards[-10:])/statistics.mean(rewards[:10])-1)*100:+.0f}%)"
        )


# ────────────────────────────────────────────────────────
# 3. PPO EVAL BASELINES
# ────────────────────────────────────────────────────────
print("\n" + "─" * 90)
print("  SECTION 3: PPO EVALUATION BASELINES")
print("─" * 90)

eval_csv = os.path.join(BASE, "baselines", "ppo_eval_200.csv")
if os.path.isfile(eval_csv):
    with open(eval_csv) as f:
        rows = list(csv.DictReader(f))
    print(f"\n  Eval rows: {len(rows)}")
    print(f"  Columns: {list(rows[0].keys())}")
    print("\n  All rows:")
    for r in rows:
        print(f"    {dict(r)}")
else:
    print("\n  ppo_eval_200.csv not found")


# ────────────────────────────────────────────────────────
# 4. EVAL BASELINES LOG
# ────────────────────────────────────────────────────────
print("\n" + "─" * 90)
print("  SECTION 4: EVAL BASELINES LOG")
print("─" * 90)

eval_log = os.path.join(BASE, "eval_baselines.log")
if os.path.isfile(eval_log):
    with open(eval_log) as f:
        text = f.read()
    lines = text.strip().split("\n")
    print(f"\n  Log lines: {len(lines)}")
    # Print key sections
    for line in lines[-40:]:
        print(f"  {line.rstrip()}")
else:
    print("\n  eval_baselines.log not found")


# ────────────────────────────────────────────────────────
# 5. MODE A BASELINE
# ────────────────────────────────────────────────────────
print("\n" + "─" * 90)
print("  SECTION 5: MODE A BASELINE (many runs)")
print("─" * 90)

mode_a = os.path.join(BASE, "mode_a_baseline")
if os.path.isdir(mode_a):
    complete_runs = []
    for d in sorted(os.listdir(mode_a)):
        rpath = os.path.join(mode_a, d)
        sp = os.path.join(rpath, "stats.json")
        if os.path.isfile(sp):
            stats = read_stats(sp)
            complete_runs.append((d, stats))

    print(f"\n  Complete runs with stats.json: {len(complete_runs)}")
    if complete_runs:
        # Sample first one to see structure
        first_stats = complete_runs[0][1]
        print(f"  Stats keys (sample): {list(first_stats.keys())}")
        for d, s in complete_runs:
            summary_parts = [f"{d}:"]
            for k in [
                "best_hard_violations",
                "hard_violations",
                "best_soft_penalty",
                "soft_penalty",
                "feasible",
            ]:
                if k in s:
                    summary_parts.append(f"{k}={s[k]}")
            print(f'    {" ".join(summary_parts)}')
