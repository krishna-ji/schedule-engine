"""Extract summary metrics from all experiment output folders."""

import csv
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv_last(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}, len(rows)


def read_json(path):
    with open(path) as f:
        return json.load(f)


print("=" * 70)
print("EXPERIMENT RESULTS SUMMARY")
print("=" * 70)

# ── GA Experiments ──
for name, folder in [
    ("GA Baseline", "ga_baseline/20260320_002246"),
    ("GA Memetic", "ga_memetic/20260320_002553"),
    ("GA Adaptive", "ga_adaptive/20260320_003837"),
]:
    r = read_json(os.path.join(BASE, "output", folder, "results.json"))
    print(f"\n── {name} ({r['mode']}) ──")
    print(f"  pop_size={r['config']['pop_size']}, ngen={r['config']['ngen']}")
    print(f"  best_hard={r['best_hard']}, best_soft={r['best_soft']:.1f}")
    print(f"  elapsed={r['elapsed_s']:.1f}s, sec/gen={r['sec_per_gen']:.3f}")

# ── LLH Differentiation ──
llh = read_json(
    os.path.join(BASE, "output/rl_llh_differentiation/llh_differentiation_results.json")
)
print(
    f"\n── LLH Differentiation (pop={llh['config']['pop_size']}, gens={llh['config']['max_gen']}) ──"
)
for aid, info in llh["actions"].items():
    print(
        f"  {info['name']:15s}: best_hard={info['best_hard']}, best_soft={info['best_soft']}, mean_hard={info['mean_hard']}, time={info['time_s']:.0f}s"
    )

# ── Static Baselines ──
last, n = read_csv_last(
    os.path.join(BASE, "output/rl_static_baselines/static_baselines.csv")
)
print(f"\n── RL Static Baselines (conservative_repair x {n} gens) ──")
print(f"  final: best_hard={last['best_hard']}, best_soft={last['best_soft']}")

# ── DQN Training ──
last_train, n_ep = read_csv_last(
    os.path.join(BASE, "output/rl_dqn/20260320_185249/training_curve.csv")
)
print(f"\n── DQN Training ({n_ep} episodes) ──")
print(f"  last_episode_reward={last_train['episode_reward']}")

# ── DQN Evaluation ──
last_eval, n_gen = read_csv_last(
    os.path.join(BASE, "output/rl_dqn/20260320_185249/evaluation_trajectory_200.csv")
)
print(f"\n── DQN Evaluation ({n_gen} gens) ──")
print(
    f"  final: best_hard={last_eval['best_hard']}, best_soft={last_eval['best_soft']}"
)
print(f"  action={last_eval['action_name']}")

# ── PPO Titan Training ──
last_ppo, n_ep_ppo = read_csv_last(
    os.path.join(BASE, "output/rl_titan_v4_sota/20260316_060537/training_curve.csv")
)
print(f"\n── PPO Titan v4 SOTA Training ({n_ep_ppo} episodes) ──")
print(f"  last_episode_reward={last_ppo['episode_reward']}")

# ── PPO Eval Baselines ──
last_baseline, n_bl = read_csv_last(
    os.path.join(BASE, "output/baselines/ppo_eval_200.csv")
)
print(f"\n── PPO Baselines Eval ({n_bl} gens) ──")
print(
    f"  final: best_hard={last_baseline['best_hard']}, best_soft={last_baseline['best_soft']}"
)
print(f"  action={last_baseline['action_name']}")

# ── Bench Compare ──
summary = read_json(os.path.join(BASE, "results/bench_compare/summary.json"))
verdict = read_json(os.path.join(BASE, "results/bench_compare/verdict.json"))
print("\n── Bench Compare (pymoo vs deap) ──")
for solver in ["pymoo", "deap"]:
    s = summary["aggregate"][solver]
    print(
        f"  {solver}: median_hard={s['median_best_hard']}, median_soft={s['median_best_soft']}, sec/gen={s['median_sec_per_gen']:.2f}"
    )
print(
    f"  verdict: hard_winner={verdict['hard_winner']}, speed_winner={verdict['speed_winner']}"
)

# ── Bench Eval ──
bench = read_json(os.path.join(BASE, "results/bench_eval.json"))
print("\n── Evaluator Benchmarks ──")
print(f"  repair_speedup={bench['repair_speedup']}x (bitset vs object)")
print(f"  eval_speedup={bench['evaluator_speedup']}x")

# ── Vectorized ──
vbench = read_json(os.path.join(BASE, "results/bench_eval_vectorized.json"))
print("\n── Vectorized Evaluator ──")
for b in vbench["benchmarks"]:
    print(
        f"  N={b['N']}: speedup={b['speedup']}x (batch={b['batch']['per_ind_mean_ms']:.3f}ms, vec={b['vectorized']['per_ind_mean_ms']:.3f}ms)"
    )

print("\n" + "=" * 70)
