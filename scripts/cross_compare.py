"""Cross-comparison of all algorithms from fromanothervm."""

import csv
import json
import os
import statistics

BASE = r"C:\Users\Administrator\Desktop\main-sch-engine\output\fromanothervm"

ga_variants = [
    ("ga_01_baseline", "GA-01 Baseline"),
    ("ga_02_memetic", "GA-02 Memetic"),
    ("ga_03_repair_sequential", "GA-03 SeqRepair"),
    ("ga_04_repair_bandit", "GA-04 Bandit"),
    ("ga_06_ultimate", "GA-06 Ultimate"),
]

results = []
for folder, label in ga_variants:
    ga_dir = os.path.join(BASE, folder)
    if not os.path.isdir(ga_dir):
        continue
    best = None
    for d in sorted(os.listdir(ga_dir)):
        rp = os.path.join(ga_dir, d)
        if os.path.isdir(rp) and os.path.isfile(os.path.join(rp, "stats.json")):
            best = rp
    if not best:
        continue
    with open(os.path.join(best, "stats.json")) as f:
        s = json.load(f)
    meta = {}
    mp = os.path.join(best, "metadata.json")
    if os.path.isfile(mp):
        with open(mp) as f:
            meta = json.load(f)
    mh = s.get("min_hard")
    ms = s.get("min_soft")
    fr = s.get("feasibility_rate")
    if isinstance(mh, list):
        mh = mh[-1]
    if isinstance(ms, list):
        ms = ms[-1]
    if isinstance(fr, list):
        fr = fr[-1]
    et = meta.get("elapsed_sec", meta.get("elapsed_time", s.get("elapsed_time", 0)))

    # Detailed hard breakdown
    det = s.get("detailed_hard")
    last_det = None
    if det:
        last_det = det[-1] if isinstance(det, list) else det

    results.append(
        {
            "label": label,
            "min_hard": mh,
            "min_soft": ms,
            "feas_rate": fr,
            "elapsed": et,
            "det_hard": last_det,
        }
    )

# RL earlier run
step = os.path.join(BASE, "rl_titan_v4_sota", "20260309_030134", "step_log.csv")
with open(step) as f:
    rows = list(csv.DictReader(f))
bh = [float(r["best_hard"]) for r in rows if r.get("best_hard", "nan") != "nan"]
bs = [float(r["best_soft"]) for r in rows if r.get("best_soft", "nan") != "nan"]
results.append(
    {
        "label": "RL Titan(earlier)",
        "min_hard": min(bh) if bh else None,
        "min_soft": min(bs) if bs else None,
        "feas_rate": 0.0,
        "elapsed": 600 * 60,
        "det_hard": None,
    }
)

# RL latest
latest = os.path.join(BASE, "rl_titan_v4_sota", "20260315_193825", "step_log.csv")
with open(latest) as f:
    rows = list(csv.DictReader(f))
bh2 = [float(r["best_hard"]) for r in rows if r.get("best_hard", "nan") != "nan"]
bs2 = [float(r["best_soft"]) for r in rows if r.get("best_soft", "nan") != "nan"]
results.append(
    {
        "label": "RL Titan(latest)",
        "min_hard": min(bh2) if bh2 else None,
        "min_soft": min(bs2) if bs2 else None,
        "feas_rate": 0.0,
        "elapsed": 3168 * 60,
        "det_hard": None,
    }
)

# Mode A aggregate
mode_a = os.path.join(BASE, "mode_a_baseline")
all_mh, all_ms, all_fr = [], [], []
for d in sorted(os.listdir(mode_a)):
    sp = os.path.join(mode_a, d, "stats.json")
    if not os.path.isfile(sp):
        continue
    with open(sp) as f:
        s = json.load(f)
    mh = s.get("min_hard")
    ms = s.get("min_soft")
    fr = s.get("feasibility_rate")
    if isinstance(mh, list):
        mh = mh[-1]
    if isinstance(ms, list):
        ms = ms[-1]
    if isinstance(fr, list):
        fr = fr[-1]
    if mh is not None:
        all_mh.append(mh)
    if ms is not None:
        all_ms.append(ms)
    if fr is not None:
        all_fr.append(fr)

if all_mh:
    results.append(
        {
            "label": "ModeA(best/15)",
            "min_hard": min(all_mh),
            "min_soft": min(all_ms),
            "feas_rate": max(all_fr),
            "elapsed": 0,
            "det_hard": None,
        }
    )
    results.append(
        {
            "label": "ModeA(avg/15)",
            "min_hard": statistics.mean(all_mh),
            "min_soft": statistics.mean(all_ms),
            "feas_rate": statistics.mean(all_fr),
            "elapsed": 0,
            "det_hard": None,
        }
    )

# Print comparison table
print("=" * 80)
print("  CROSS-COMPARISON: ALL ALGORITHMS")
print("=" * 80)
hdr = (
    f"  {'Algorithm':<22} {'MinHard':>8} {'MinSoft':>10} {'FeasRate':>10} {'Time':>10}"
)
print(hdr)
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for r in results:
    mh_s = f"{r['min_hard']:.0f}" if r["min_hard"] is not None else "?"
    ms_s = f"{r['min_soft']:.1f}" if r["min_soft"] is not None else "?"
    fr_s = f"{r['feas_rate']:.3f}" if r["feas_rate"] is not None else "?"
    et_s = f"{r['elapsed']/60:.0f}m" if r["elapsed"] else "?"
    print(f"  {r['label']:<22} {mh_s:>8} {ms_s:>10} {fr_s:>10} {et_s:>10}")

# Print detailed hard for GAs
print()
print("=" * 80)
print("  DETAILED HARD VIOLATIONS BREAKDOWN (per constraint type)")
print("=" * 80)
for r in results:
    if r["det_hard"] and isinstance(r["det_hard"], dict):
        print(f"\n  {r['label']}:")
        for k, v in r["det_hard"].items():
            if v != 0:
                print(f"    {k}: {v}")

# RL step data analysis
print()
print("=" * 80)
print("  RL TITAN EARLIER RUN — ACTUAL CONSTRAINT DATA")
print("=" * 80)
with open(step) as f:
    rows = list(csv.DictReader(f))
bh = [float(r["best_hard"]) for r in rows if r.get("best_hard", "nan") != "nan"]
bs = [float(r["best_soft"]) for r in rows if r.get("best_soft", "nan") != "nan"]
ff = [float(r["feasible_frac"]) for r in rows if r.get("feasible_frac", "nan") != "nan"]
dh = [float(r["delta_hard"]) for r in rows]
print(f"  Steps with data: {len(bh)}/{len(rows)}")
if bh:
    print(f"  Best hard: {min(bh):.0f} (min), {bh[-1]:.0f} (final)")
    print(f"  Best soft: {min(bs):.1f} (min), {bs[-1]:.1f} (final)")
    print(f"  Feasible frac: {ff[-1]:.3f}" if ff else "  Feasible frac: N/A")

    # Constraint trajectory
    window = max(1, len(bh) // 10)
    print(f"\n  Hard Violations Trajectory (avg per window of {window}):")
    for i in range(0, len(bh), window):
        w = bh[i : i + window]
        print(
            f"    Step {i:>4}-{min(i+window,len(bh)):>4}: avg={statistics.mean(w):.1f} min={min(w):.0f}"
        )

    # Per-constraint breakdown at final step
    last_row = rows[-1]
    cv_keys = [k for k in last_row.keys() if k.startswith("cv_")]
    if cv_keys:
        print("\n  Final Constraint Violations (per type):")
        for k in cv_keys:
            v = float(last_row[k])
            if v > 0:
                print(f"    {k}: {v:.1f}")
else:
    print("  ALL METRICS ARE NaN - no constraint data available")

# PPO eval
print()
print("=" * 80)
print("  PPO EVALUATION (ppo_eval_200.csv)")
print("=" * 80)
eval_csv = os.path.join(BASE, "baselines", "ppo_eval_200.csv")
if os.path.isfile(eval_csv):
    with open(eval_csv) as f:
        rows = list(csv.DictReader(f))
    print(f"  Eval episodes: {len(rows)}")
    if rows:
        print(f"  Columns: {list(rows[0].keys())}")
        for r in rows[:10]:
            print(f"    {dict(r)}")
else:
    print("  File not found")

# Mode A summary
print()
print("=" * 80)
print("  MODE A BASELINE (15 runs)")
print("=" * 80)
print(f"  Runs: {len(all_mh)}")
print(
    f"  Min hard: best={min(all_mh):.0f} worst={max(all_mh):.0f} avg={statistics.mean(all_mh):.1f}"
)
print(
    f"  Min soft: best={min(all_ms):.1f} worst={max(all_ms):.1f} avg={statistics.mean(all_ms):.1f}"
)
print(
    f"  Feasibility: best={max(all_fr):.3f} worst={min(all_fr):.3f} avg={statistics.mean(all_fr):.3f}"
)
