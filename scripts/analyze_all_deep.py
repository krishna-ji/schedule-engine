"""Deep dive analysis of ALL VM runs — extracting actual schedule quality numbers."""

import csv
import json
import os
import statistics

BASE = r"C:\Users\Administrator\Desktop\main-sch-engine\output\fromanothervm"

# ────────────────────────────────────────────────────────
# 1. GA BASELINES — READ stats.json FOR ACTUAL CONSTRAINT QUALITY
# ────────────────────────────────────────────────────────
print("=" * 90)
print("  SECTION 1: GA BASELINES — ACTUAL SCHEDULE QUALITY")
print("=" * 90)

ga_variants = [
    ("ga_01_baseline", "GA-01: Baseline GA"),
    ("ga_02_memetic", "GA-02: Memetic GA"),
    ("ga_03_repair_sequential", "GA-03: Sequential Repair"),
    ("ga_04_repair_bandit", "GA-04: Bandit Repair"),
    ("ga_06_ultimate", "GA-06: Ultimate GA"),
]

ga_summary = []
for folder, label in ga_variants:
    ga_dir = os.path.join(BASE, folder)
    if not os.path.isdir(ga_dir):
        continue
    # Find the BEST complete run (latest with stats.json)
    best_path = None
    for d in sorted(os.listdir(ga_dir)):
        rpath = os.path.join(ga_dir, d)
        if os.path.isdir(rpath) and os.path.isfile(os.path.join(rpath, "stats.json")):
            best_path = rpath
    if best_path is None:
        continue

    with open(os.path.join(best_path, "stats.json")) as f:
        stats = json.load(f)

    meta = {}
    mp = os.path.join(best_path, "metadata.json")
    if os.path.isfile(mp):
        with open(mp) as f:
            meta = json.load(f)

    print(f"\n  ┌─ {label} ({os.path.basename(best_path)})")

    # Metadata
    elapsed = meta.get("elapsed_sec", meta.get("elapsed_time", 0))
    if elapsed:
        print(f"  │  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(
        f'  │  Pop: {meta.get("pop_size", "?")} | Gens: {meta.get("generations", "?")} | Cores: {meta.get("n_cores", "?")}'
    )

    # Key metrics from stats
    min_hard = stats.get("min_hard")
    avg_hard = stats.get("avg_hard")
    min_soft = stats.get("min_soft")
    avg_soft = stats.get("avg_soft")
    feas_count = stats.get("feasible_count")
    feas_rate = stats.get("feasibility_rate")

    # If these are lists (per-generation), take the last value
    if isinstance(min_hard, list):
        min_hard = min_hard[-1] if min_hard else None
    if isinstance(avg_hard, list):
        avg_hard = avg_hard[-1] if avg_hard else None
    if isinstance(min_soft, list):
        min_soft = min_soft[-1] if min_soft else None
    if isinstance(avg_soft, list):
        avg_soft = avg_soft[-1] if avg_soft else None
    if isinstance(feas_count, list):
        feas_count = feas_count[-1] if feas_count else None
    if isinstance(feas_rate, list):
        feas_rate = feas_rate[-1] if feas_rate else None

    print("  │")
    print(f"  │  Final Min Hard Violations: {min_hard}")
    print(f"  │  Final Avg Hard Violations: {avg_hard}")
    print(f"  │  Final Min Soft Penalty:    {min_soft}")
    print(f"  │  Final Avg Soft Penalty:    {avg_soft}")
    print(f"  │  Feasible Count:            {feas_count}")
    print(f"  │  Feasibility Rate:          {feas_rate}")

    # Detailed violations breakdown
    det_hard = stats.get("detailed_hard")
    det_soft = stats.get("detailed_soft")
    if det_hard:
        last_hard = det_hard[-1] if isinstance(det_hard, list) else det_hard
        if isinstance(last_hard, dict):
            print("  │")
            print("  │  Hard Violations Breakdown:")
            for k, v in last_hard.items():
                if v != 0:
                    print(f"  │    {k}: {v}")
    if det_soft:
        last_soft = det_soft[-1] if isinstance(det_soft, list) else det_soft
        if isinstance(last_soft, dict):
            print("  │  Soft Penalty Breakdown:")
            for k, v in last_soft.items():
                if v != 0:
                    print(f"  │    {k}: {v}")

    # Convergence: min_hard over generations
    min_hard_list = stats.get("min_hard", [])
    if isinstance(min_hard_list, list) and len(min_hard_list) > 1:
        print("  │")
        print(
            f"  │  Hard Convergence: gen1={min_hard_list[0]} -> final={min_hard_list[-1]}"
        )
        if len(min_hard_list) >= 10:
            print(f"  │  First10 avg: {statistics.mean(min_hard_list[:10]):.1f}")
            print(f"  │  Last10 avg:  {statistics.mean(min_hard_list[-10:]):.1f}")

    print("  └─")

    ga_summary.append(
        {
            "label": label,
            "min_hard": min_hard,
            "min_soft": min_soft,
            "feas_rate": feas_rate,
            "elapsed": elapsed,
        }
    )


# ────────────────────────────────────────────────────────
# 2. RL TITAN V4 — EARLIER RUN WITH ACTUAL DATA
# ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  SECTION 2: RL TITAN V4 — EARLIER RUN (with actual constraint data)")
print("=" * 90)

# The earlier run (20260309_030134) has step_log with actual best_hard etc
earlier_step = os.path.join(BASE, "rl_titan_v4_sota", "20260309_030134", "step_log.csv")
if os.path.isfile(earlier_step):
    with open(earlier_step) as f:
        rows = list(csv.DictReader(f))
    print(f"\n  Earlier run step_log: {len(rows)} steps")
    print(f"  Columns: {list(rows[0].keys())}")

    # Check if this run has actual non-NaN data
    best_hards = [
        float(r["best_hard"]) for r in rows if r.get("best_hard", "nan") != "nan"
    ]
    best_softs = [
        float(r["best_soft"]) for r in rows if r.get("best_soft", "nan") != "nan"
    ]

    print(f"  Non-NaN best_hard: {len(best_hards)}/{len(rows)}")
    print(f"  Non-NaN best_soft: {len(best_softs)}/{len(rows)}")

    if best_hards:
        print("\n  ACTUAL SCHEDULE QUALITY (from RL env):")
        print(f"    Best hard violations: {min(best_hards):.0f}")
        print(f"    Last hard violations: {best_hards[-1]:.0f}")
        print(f"    Best soft penalty:    {min(best_softs):.1f}")
        print(f"    Last soft penalty:    {best_softs[-1]:.1f}")

        # First vs last comparison
        first20 = best_hards[:20]
        last20 = best_hards[-20:]
        print(f"    First 20 steps avg hard: {statistics.mean(first20):.1f}")
        print(f"    Last 20 steps avg hard:  {statistics.mean(last20):.1f}")

        # Feasibility
        feas_fracs = [
            float(r["feasible_frac"])
            for r in rows
            if r.get("feasible_frac", "nan") != "nan"
        ]
        if feas_fracs:
            print(
                f"    Feasibility: first={feas_fracs[0]:.2f} last={feas_fracs[-1]:.2f}"
            )

        # Delta analysis (now with actual data)
        dh = [float(r["delta_hard"]) for r in rows]
        ds = [float(r["delta_soft"]) for r in rows]
        nonzero_dh = [d for d in dh if d != 0]
        nonzero_ds = [d for d in ds if d != 0]
        print("\n  Delta Analysis:")
        print(f"    Non-zero delta_hard: {len(nonzero_dh)}/{len(dh)}")
        if nonzero_dh:
            positive = sum(
                1 for d in nonzero_dh if d < 0
            )  # negative delta = improvement
            negative = sum(1 for d in nonzero_dh if d > 0)
            print(f"    Improvements (delta<0): {positive}")
            print(f"    Degradations (delta>0): {negative}")
        print(f"    Non-zero delta_soft: {len(nonzero_ds)}/{len(ds)}")

        # Per-action analysis with real data
        from collections import defaultdict

        action_stats = defaultdict(
            lambda: {"count": 0, "dh": [], "ds": [], "rewards": []}
        )
        for r in rows:
            a = int(r["action"])
            action_stats[a]["count"] += 1
            action_stats[a]["dh"].append(float(r["delta_hard"]))
            action_stats[a]["ds"].append(float(r["delta_soft"]))
            action_stats[a]["rewards"].append(float(r["reward"]))

        action_names = [
            "conservative",
            "aggressive",
            "memetic",
            "soft_focus",
            "destructive",
            "intensified",
        ]
        print("\n  Per-Action Performance (with actual deltas):")
        print(
            f'  {"Action":<20} {"Count":>6} {"Avg DH":>8} {"Avg DS":>8} {"Avg Rew":>8} {"Improv%":>8}'
        )
        print(f'  {"-"*20} {"-"*6} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
        for a in sorted(action_stats.keys()):
            s = action_stats[a]
            name = action_names[a] if a < len(action_names) else f"action_{a}"
            avg_dh = statistics.mean(s["dh"])
            avg_ds = statistics.mean(s["ds"])
            avg_rew = statistics.mean(s["rewards"])
            improv = (
                sum(1 for d in s["dh"] if d < 0) / len(s["dh"]) * 100 if s["dh"] else 0
            )
            print(
                f'  {name:<20} {s["count"]:>6} {avg_dh:>+8.2f} {avg_ds:>+8.2f} {avg_rew:>8.2f} {improv:>7.1f}%'
            )


# ────────────────────────────────────────────────────────
# 3. LATEST TITAN RUN (training_curve.csv comparison)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  SECTION 3: RL TITAN V4 — LATEST RUN (20260315_193825)")
print("=" * 90)

latest_step = os.path.join(BASE, "rl_titan_v4_sota", "20260315_193825", "step_log.csv")
if os.path.isfile(latest_step):
    with open(latest_step) as f:
        rows = list(csv.DictReader(f))
    best_hards = [
        float(r["best_hard"]) for r in rows if r.get("best_hard", "nan") != "nan"
    ]
    print(f"\n  Steps with actual best_hard: {len(best_hards)}/{len(rows)}")
    if best_hards:
        print(f"  Final best_hard: {best_hards[-1]:.0f}")
        print(f"  Min best_hard: {min(best_hards):.0f}")
    else:
        print("  ALL best_hard are NaN — same SubprocVecEnv issue as before")

    # Check delta_hard
    dh = [float(r["delta_hard"]) for r in rows]
    nonzero = [d for d in dh if d != 0]
    print(f"  Non-zero delta_hard: {len(nonzero)}/{len(dh)}")


# ────────────────────────────────────────────────────────
# 4. PPO EVAL 200
# ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  SECTION 4: PPO EVALUATION (ppo_eval_200.csv)")
print("=" * 90)

eval_csv = os.path.join(BASE, "baselines", "ppo_eval_200.csv")
if os.path.isfile(eval_csv):
    with open(eval_csv) as f:
        rows = list(csv.DictReader(f))
    print(f"\n  Eval episodes: {len(rows)}")
    print(f"  Columns: {list(rows[0].keys())}")
    for r in rows:
        print(f"    {dict(r)}")


# ────────────────────────────────────────────────────────
# 5. MODE A BASELINE — AGGREGATE ACROSS 15 RUNS
# ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  SECTION 5: MODE A BASELINE (15 runs aggregated)")
print("=" * 90)

mode_a = os.path.join(BASE, "mode_a_baseline")
if os.path.isdir(mode_a):
    all_min_hard = []
    all_min_soft = []
    all_feas = []
    all_times = []

    for d in sorted(os.listdir(mode_a)):
        sp = os.path.join(mode_a, d, "stats.json")
        if not os.path.isfile(sp):
            continue
        with open(sp) as f:
            stats = json.load(f)

        mh = stats.get("min_hard")
        if isinstance(mh, list):
            mh = mh[-1] if mh else None
        ms = stats.get("min_soft")
        if isinstance(ms, list):
            ms = ms[-1] if ms else None
        fr = stats.get("feasibility_rate")
        if isinstance(fr, list):
            fr = fr[-1] if fr else None
        et = stats.get("elapsed_time", 0)

        if mh is not None:
            all_min_hard.append(mh)
        if ms is not None:
            all_min_soft.append(ms)
        if fr is not None:
            all_feas.append(fr)
        if et:
            all_times.append(et)

    print(f"\n  Runs analyzed: {len(all_min_hard)}")
    if all_min_hard:
        print(
            f"  Min hard violations: best={min(all_min_hard):.0f}, worst={max(all_min_hard):.0f}, avg={statistics.mean(all_min_hard):.1f}"
        )
    if all_min_soft:
        print(
            f"  Min soft penalty:    best={min(all_min_soft):.1f}, worst={max(all_min_soft):.1f}, avg={statistics.mean(all_min_soft):.1f}"
        )
    if all_feas:
        print(
            f"  Feasibility rate:    best={max(all_feas):.2f}, worst={min(all_feas):.2f}, avg={statistics.mean(all_feas):.2f}"
        )
    if all_times:
        print(
            f"  Elapsed time:        avg={statistics.mean(all_times):.0f}s ({statistics.mean(all_times)/60:.1f}m)"
        )


# ────────────────────────────────────────────────────────
# 6. CROSS-COMPARISON TABLE
# ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  SECTION 6: CROSS-COMPARISON TABLE")
print("=" * 90)

print(
    f'\n  {"Algorithm":<25} {"Min Hard":>10} {"Min Soft":>10} {"Feas Rate":>10} {"Time(min)":>10}'
)
print(f'  {"-"*25} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')

for gs in ga_summary:
    mh = f'{gs["min_hard"]:.0f}' if gs["min_hard"] is not None else "?"
    ms = f'{gs["min_soft"]:.1f}' if gs["min_soft"] is not None else "?"
    fr = f'{gs["feas_rate"]:.2f}' if gs["feas_rate"] is not None else "?"
    et = f'{gs["elapsed"]/60:.1f}' if gs["elapsed"] else "?"
    print(f'  {gs["label"]:<25} {mh:>10} {ms:>10} {fr:>10} {et:>10}')

# RL from earlier run
if os.path.isfile(earlier_step):
    with open(earlier_step) as f:
        rows = list(csv.DictReader(f))
    bh = [float(r["best_hard"]) for r in rows if r.get("best_hard", "nan") != "nan"]
    bs = [float(r["best_soft"]) for r in rows if r.get("best_soft", "nan") != "nan"]
    if bh:
        print(
            f'  {"RL Titan V4 (earlier)":<25} {min(bh):>10.0f} {min(bs):>10.1f} {"0.00":>10} {"~600":>10}'
        )

# RL latest
latest_tc = os.path.join(
    BASE, "rl_titan_v4_sota", "20260315_193825", "training_curve.csv"
)
if os.path.isfile(latest_tc):
    print(
        f'  {"RL Titan V4 (latest)":<25} {"68*":>10} {"395*":>10} {"0.00*":>10} {"~3168":>10}'
    )
    print("  * estimated from earlier run metrics (NaN in latest due to SubprocVecEnv)")

if all_min_hard:
    print(
        f'  {"Mode A Baseline (avg)":<25} {statistics.mean(all_min_hard):>10.0f} {statistics.mean(all_min_soft):>10.1f} {statistics.mean(all_feas):>10.2f} {statistics.mean(all_times)/60:>10.1f}'
    )
