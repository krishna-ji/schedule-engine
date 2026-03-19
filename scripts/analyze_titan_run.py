"""Deep analysis of Titan V4 SOTA run from VM."""

import csv
import math
import statistics
import sys

base = (
    sys.argv[1]
    if len(sys.argv) > 1
    else r"C:\Users\Administrator\Desktop\main-sch-engine\output\fromanothervm\rl_titan_v4_sota\20260315_193825"
)

# ===== TRAINING CURVE DEEP DIVE =====
print("=" * 80)
print("  1. TRAINING CURVE DEEP ANALYSIS")
print("=" * 80)
with open(f"{base}/training_curve.csv") as f:
    rows = list(csv.DictReader(f))

rewards = [float(r["episode_reward"]) for r in rows]
lengths = [int(r["episode_length"]) for r in rows]
episodes = [int(r["episode"]) for r in rows]

actions = {}
for i in range(6):
    key = f"action_{i}_count"
    actions[i] = [int(r[key]) for r in rows]

print(f"Episodes: {len(rows)}")
print(f"Timestep range: {rows[0]['timestep']} -> {rows[-1]['timestep']}")
unique_lens = set(lengths[:-1])
if len(unique_lens) == 1:
    print(f"Episode length: all={lengths[0]} (constant)")
else:
    print(f"Episode lengths vary: {unique_lens}")
if lengths[-1] < lengths[0]:
    print(f"Last episode length: {lengths[-1]} (TRUNCATED, normal={lengths[0]})")

# Reward analysis
print("\n--- Reward Statistics ---")
print(f"  Min:    {min(rewards):8.2f}  (ep {rewards.index(min(rewards))+1})")
print(f"  Max:    {max(rewards):8.2f}  (ep {rewards.index(max(rewards))+1})")
print(f"  Mean:   {statistics.mean(rewards):8.2f}")
print(f"  Median: {statistics.median(rewards):8.2f}")
print(f"  Stdev:  {statistics.stdev(rewards):8.2f}")
print(f"  CV:     {statistics.stdev(rewards)/statistics.mean(rewards)*100:.1f}%")

# Rolling windows of 10
print("\n--- 10-Episode Rolling Windows ---")
for i in range(0, len(rewards), 10):
    w = rewards[i : i + 10]
    avg = statistics.mean(w)
    bar = "#" * int(avg / 5)
    print(f"  Ep {i+1:3d}-{min(i+10,len(rewards)):3d}: avg={avg:7.1f} | {bar}")

# Monotonic improvement check
print("\n--- New Best Records ---")
best_so_far = float("-inf")
new_bests = []
for i, r in enumerate(rewards):
    if r > best_so_far:
        best_so_far = r
        new_bests.append((i + 1, r))
for ep, reward in new_bests:
    print(f"  Ep {ep:3d}: {reward:.2f}")

# Reward stability (last 20 eps)
if len(rewards) >= 20:
    last20 = rewards[-20:]
    print("\n--- Stability (Last 20 Episodes) ---")
    print(f"  Mean:  {statistics.mean(last20):.2f}")
    print(f"  Stdev: {statistics.stdev(last20):.2f}")
    print(f"  CV:    {statistics.stdev(last20)/statistics.mean(last20)*100:.1f}%")
    print(f"  Range: [{min(last20):.2f}, {max(last20):.2f}]")

# Learning rate (linear regression on rewards)
n = len(rewards)
x_mean = (n - 1) / 2
y_mean = statistics.mean(rewards)
ss_xx = sum((i - x_mean) ** 2 for i in range(n))
ss_xy = sum((i - x_mean) * (rewards[i] - y_mean) for i in range(n))
slope = ss_xy / ss_xx if ss_xx > 0 else 0
intercept = y_mean - slope * x_mean
ss_res = sum((rewards[i] - (intercept + slope * i)) ** 2 for i in range(n))
ss_tot = sum((rewards[i] - y_mean) ** 2 for i in range(n))
r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

print("\n--- Linear Reward Trend ---")
print(f"  Slope:     {slope:+.2f} reward/episode")
print(f"  R-squared: {r_squared:.3f}")
print(f"  Predicted ep 1:   {intercept:.1f}")
print(f"  Predicted ep 87:  {intercept + slope * 86:.1f}")

# Action preference evolution per 10-ep window
print("\n--- Action Preference Evolution (per 10-ep window) ---")
action_names = [
    "conservative",
    "aggressive",
    "memetic",
    "soft_focus",
    "destructive",
    "intensified",
]
header = (
    "Window     | " + " | ".join(f"{n[:5]:>5}" for n in action_names) + " | Dominant"
)
print(f"  {header}")
print(f"  {'-'*len(header)}")
for i in range(0, len(rows), 10):
    end = min(i + 10, len(rows))
    counts = [sum(actions[a][i:end]) for a in range(6)]
    total = sum(counts)
    pcts = [100 * c / total for c in counts]
    dominant = action_names[max(range(6), key=lambda a: counts[a])]
    pct_str = " | ".join(f"{p:5.1f}" for p in pcts)
    print(f"  Ep{i+1:3d}-{end:3d} | {pct_str} | {dominant}")


# ===== STEP LOG DEEP DIVE =====
print("\n" + "=" * 80)
print("  2. STEP LOG DEEP ANALYSIS")
print("=" * 80)
with open(f"{base}/step_log.csv") as f:
    srows = list(csv.DictReader(f))

print(f"Total steps logged: {len(srows)}")
print(f"Columns: {list(srows[0].keys())}")

step_rewards = [float(r["reward"]) for r in srows]
step_actions = [int(r["action"]) for r in srows]
delta_hard = [float(r["delta_hard"]) for r in srows]
delta_soft = [float(r["delta_soft"]) for r in srows]
rejected = [r["rejected"] == "True" for r in srows]

print("\n--- Step Reward ---")
print(f"  Min:  {min(step_rewards):8.2f}")
print(f"  Max:  {max(step_rewards):8.2f}")
print(f"  Mean: {statistics.mean(step_rewards):8.2f}")
print(f"  Stdev:{statistics.stdev(step_rewards):8.2f}")

# Reward distribution
bins = {}
for r in step_rewards:
    b = int(r // 5) * 5
    bins[b] = bins.get(b, 0) + 1
print("\n--- Step Reward Distribution ---")
for b in sorted(bins.keys()):
    bar = "#" * (bins[b] * 50 // max(bins.values()))
    print(f"  [{b:6.0f}, {b+5:6.0f}): {bins[b]:5d} {bar}")

# Delta analysis
print("\n--- Delta Hard (constraint improvements per step) ---")
non_zero_dh = [d for d in delta_hard if d != 0]
print(f"  All zero: {len(non_zero_dh) == 0}")
print(f"  Non-zero count: {len(non_zero_dh)} / {len(delta_hard)}")
if non_zero_dh:
    print(f"  Positive (improved): {sum(1 for d in non_zero_dh if d > 0)}")
    print(f"  Negative (worsened): {sum(1 for d in non_zero_dh if d < 0)}")

print("\n--- Delta Soft (soft penalty changes per step) ---")
non_zero_ds = [d for d in delta_soft if d != 0]
print(f"  All zero: {len(non_zero_ds) == 0}")
print(f"  Non-zero count: {len(non_zero_ds)} / {len(delta_soft)}")

# Rejection rate
print("\n--- Action Rejection ---")
rej_count = sum(rejected)
print(f"  Rejected: {rej_count} / {len(rejected)} ({100*rej_count/len(rejected):.1f}%)")

# Best_hard, best_soft, feasible_frac analysis
best_hard = [r["best_hard"] for r in srows]
best_soft = [r["best_soft"] for r in srows]
feas_frac = [r["feasible_frac"] for r in srows]

nan_bh = sum(1 for v in best_hard if v == "nan")
nan_bs = sum(1 for v in best_soft if v == "nan")
nan_ff = sum(1 for v in feas_frac if v == "nan")
print("\n--- Data Completeness ---")
print(
    f"  best_hard NaN:     {nan_bh}/{len(best_hard)} ({100*nan_bh/len(best_hard):.0f}%)"
)
print(
    f"  best_soft NaN:     {nan_bs}/{len(best_soft)} ({100*nan_bs/len(best_soft):.0f}%)"
)
print(
    f"  feasible_frac NaN: {nan_ff}/{len(feas_frac)} ({100*nan_ff/len(feas_frac):.0f}%)"
)

# Per-action reward analysis
print("\n--- Per-Action Step Rewards ---")
from collections import defaultdict

action_rewards = defaultdict(list)
for r, a in zip(step_rewards, step_actions, strict=False):
    action_rewards[a].append(r)

for a in sorted(action_rewards.keys()):
    rlist = action_rewards[a]
    print(
        f"  Action {a} ({action_names[a][:12]:>12}): n={len(rlist):4d}  "
        f"avg={statistics.mean(rlist):6.2f}  "
        f"min={min(rlist):7.2f}  max={max(rlist):6.2f}"
    )


# ===== SB3 TRAINING METRICS =====
print("\n" + "=" * 80)
print("  3. SB3 TRAINING METRICS DEEP ANALYSIS")
print("=" * 80)
with open(f"{base}/sb3_training_metrics.csv") as f:
    mrows = list(csv.DictReader(f))

# Deduplicate (SB3 repeats metrics per step within an iteration)
ts_seen = set()
unique_rows = []
for r in mrows:
    ts = r["timestep"]
    if ts not in ts_seen:
        ts_seen.add(ts)
        unique_rows.append(r)

print(f"Raw rows: {len(mrows)}, Unique iterations: {len(unique_rows)}")

# Extract metrics
pgl = [float(r["policy_gradient_loss"]) for r in unique_rows]
vl = [float(r["value_loss"]) for r in unique_rows]
el = [float(r["entropy_loss"]) for r in unique_rows]
kl = [float(r["approx_kl"]) for r in unique_rows]
cf = [float(r["clip_fraction"]) for r in unique_rows]
ev = [float(r["explained_variance"]) for r in unique_rows]
loss = [float(r["loss"]) for r in unique_rows]


def analyze_metric(name, vals, higher_is_better=None):
    print(f"\n  --- {name} ---")
    print(f"    First 5 avg: {statistics.mean(vals[:5]):.6f}")
    print(f"    Last 5 avg:  {statistics.mean(vals[-5:]):.6f}")
    print(f"    Min: {min(vals):.6f}  Max: {max(vals):.6f}")
    print(f"    Overall avg: {statistics.mean(vals):.6f}")
    trend = statistics.mean(vals[-5:]) - statistics.mean(vals[:5])
    arrow = "↑" if trend > 0 else "↓"
    if higher_is_better is not None:
        good = (trend > 0) == higher_is_better
        verdict = "GOOD" if good else "CONCERNING"
    else:
        verdict = ""
    print(f"    Trend: {trend:+.6f} {arrow} {verdict}")


analyze_metric("Policy Gradient Loss", pgl, higher_is_better=None)
analyze_metric("Value Loss", vl, higher_is_better=False)
analyze_metric("Entropy Loss", el, higher_is_better=None)
analyze_metric("Approx KL Divergence", kl, higher_is_better=False)
analyze_metric("Clip Fraction", cf, higher_is_better=False)
analyze_metric("Explained Variance", ev, higher_is_better=True)
analyze_metric("Total Loss", loss, higher_is_better=False)

# Explained variance trajectory
print("\n  --- Explained Variance Trajectory (value function quality) ---")
for i in range(0, len(ev), max(1, len(ev) // 8)):
    bar = "#" * int(max(0, ev[i]) * 40)
    print(f"    Iter {i+1:3d}: {ev[i]:+.4f} | {bar}")
print(f"    Iter {len(ev):3d}: {ev[-1]:+.4f} | {'#' * int(max(0, ev[-1]) * 40)}")


# ===== FINAL VERDICT =====
print("\n" + "=" * 80)
print("  4. OVERALL VERDICT")
print("=" * 80)

# Scoring
scores = {}
# Reward improvement
reward_improve = (
    (statistics.mean(rewards[-10:]) - statistics.mean(rewards[:10]))
    / statistics.mean(rewards[:10])
    * 100
)
scores["reward_improvement"] = min(100, reward_improve)
print(
    f"\n  Reward Improvement: {reward_improve:+.1f}% {'EXCELLENT' if reward_improve > 80 else 'GOOD' if reward_improve > 40 else 'MODERATE' if reward_improve > 20 else 'POOR'}"
)

# Learning stability
cv_last20 = statistics.stdev(rewards[-20:]) / statistics.mean(rewards[-20:]) * 100
scores["stability"] = max(0, 100 - cv_last20 * 2)
print(
    f"  Stability (last 20 CV): {cv_last20:.1f}% {'STABLE' if cv_last20 < 15 else 'MODERATE' if cv_last20 < 25 else 'UNSTABLE'}"
)

# Value function
scores["value_fn"] = min(100, max(0, ev[-1] * 100))
print(
    f"  Value Function (explained var): {ev[-1]:.3f} {'GOOD' if ev[-1] > 0.5 else 'MODERATE' if ev[-1] > 0.2 else 'POOR'}"
)

# Sample efficiency
eps_per_100k = len(rewards) / (float(rows[-1]["timestep"]) / 100000)
scores["sample_eff"] = min(100, eps_per_100k)
print(f"  Sample Efficiency: {eps_per_100k:.0f} ep/100k ts")

# Data quality
data_quality = 100 * (1 - nan_bh / len(best_hard))
scores["data_quality"] = data_quality
print(
    f"  Data Completeness: {data_quality:.0f}% {'GOOD' if data_quality > 80 else 'POOR - NaN metrics!'}"
)

# Heuristic differentiation
action_means = [statistics.mean(action_rewards[a]) for a in range(6)]
diff_range = max(action_means) - min(action_means)
scores["heuristic_diff"] = min(100, diff_range * 10)
print(
    f"  Heuristic Differentiation: range={diff_range:.2f} {'GOOD' if diff_range > 3 else 'LOW' if diff_range > 1 else 'NONE'}"
)

overall = statistics.mean(scores.values())
print(f"\n  === OVERALL SCORE: {overall:.0f}/100 ===")
grade = (
    "A"
    if overall > 80
    else "B" if overall > 65 else "C" if overall > 50 else "D" if overall > 35 else "F"
)
print(f"  === GRADE: {grade} ===")
