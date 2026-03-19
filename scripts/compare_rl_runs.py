"""Compare all RL runs."""

import csv
import pathlib

base = pathlib.Path("output/rl_titan_v4_sota")
runs = sorted(base.iterdir())

print("=" * 70)
print("RL TITAN V4 SOTA - ALL RUNS COMPARISON")
print("=" * 70)

for rd in runs:
    tc = rd / "training_curve.csv"
    sl = rd / "step_log.csv"
    sb3 = rd / "sb3_training_metrics.csv"

    print(f"\nRun: {rd.name}")
    print("-" * 50)

    if tc.exists():
        rows = list(csv.DictReader(open(tc)))
        eps = len(rows)
        rewards = [float(r["episode_reward"]) for r in rows]
        print(f"  Episodes:     {eps}")
        print(f"  Reward first: {rewards[0]:.1f}")
        print(f"  Reward last:  {rewards[-1]:.1f}")
        print(f"  Reward max:   {max(rewards):.1f}")
        print(f"  Reward avg:   {sum(rewards)/len(rewards):.1f}")

    if sl.exists():
        rows = list(csv.DictReader(open(sl)))
        total = len(rows)
        valid_nums = []
        for r in rows:
            bh = r.get("best_hard", "nan")
            if bh not in ("nan", ""):
                try:
                    v = float(bh)
                    if v == v:  # not NaN
                        valid_nums.append(r)
                except:
                    pass
        print(f"  Step log:     {total} rows, {len(valid_nums)} with constraint data")
        if valid_nums:
            bh = [float(r["best_hard"]) for r in valid_nums]
            print(f"  Best hard:    min={min(bh):.0f}  last={bh[-1]:.0f}")
            bs = [float(r["best_soft"]) for r in valid_nums]
            print(f"  Best soft:    min={min(bs):.1f}  last={bs[-1]:.1f}")
            ff = [float(r["feasible_frac"]) for r in valid_nums]
            print(f"  Feasible:     max={max(ff):.1%}")

    if sb3.exists():
        rows = list(csv.DictReader(open(sb3)))
        if rows:
            ev = []
            for r in rows:
                val = r.get("explained_variance", "")
                if val:
                    try:
                        ev.append(float(val))
                    except:
                        pass
            if ev:
                print(f"  Expl var:     last={ev[-1]:.3f}  max={max(ev):.3f}")

    files = [f.name for f in rd.iterdir() if f.is_file()]
    has_model = any("ppo_titan" in f for f in files)
    has_figs = any("fig_" in f for f in files)
    has_ckpts = (rd / "checkpoints").exists()
    print(f"  Model zip:    {has_model}")
    print(f"  Figures:      {has_figs}")
    print(f"  Checkpoints:  {has_ckpts}")

print()
print("=" * 70)
print("PPO EVAL (baselines/ppo_eval_200.csv)")
print("=" * 70)
ppo = pathlib.Path("output/baselines/ppo_eval_200.csv")
if ppo.exists():
    rows = list(csv.DictReader(open(ppo)))
    last = rows[-1]
    print(f"  Steps:        {len(rows)}")
    print(f"  Best hard:    {float(last['best_hard']):.0f}")
    print(f"  Best soft:    {float(last['best_soft']):.1f}")
    print(f"  Mean hard:    {float(last['mean_hard']):.0f}")
    print(f"  Feasible:     {float(last['feasible_frac']):.1%}")

print()
print("=" * 70)
print("STATIC BASELINES (rl_phase54/static_baselines.csv)")
print("=" * 70)
sb = pathlib.Path("output/rl_phase54/static_baselines.csv")
if sb.exists():
    rows = list(csv.DictReader(open(sb)))
    print(f"  Heuristics:   {len(rows)}")
    for r in rows:
        name = r.get("heuristic", r.get("action_name", "?"))
        bh = r.get("best_hard", "?")
        bs = r.get("best_soft", "?")
        print(f"    {name:<30} hard={bh}  soft={bs}")
