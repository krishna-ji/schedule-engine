"""Detailed metrics extraction for thesis results chapter."""

import csv
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# DQN Training reward stats
rows = read_csv(os.path.join(BASE, "output/rl_dqn/20260320_185249/training_curve.csv"))
rewards = [float(r["episode_reward"]) for r in rows]
print("DQN Training Reward Stats:")
print(
    f"  episodes={len(rows)}, min={min(rewards):.2f}, max={max(rewards):.2f}, mean={sum(rewards)/len(rewards):.2f}"
)
print("  First 5:", [round(r, 1) for r in rewards[:5]])
print("  Last 5:", [round(r, 1) for r in rewards[-5:]])

# DQN last 3 eps action distribution
print("\nDQN Last 3 episodes action distribution:")
for r in rows[-3:]:
    ep = r["episode"]
    rew = float(r["episode_reward"])
    acts = ", ".join(f"a{i}={r.get('action_'+str(i)+'_count','?')}" for i in range(7))
    print(f"  ep {ep}: rew={rew:.1f}, {acts}")

# PPO Titan stats
rows_ppo = read_csv(
    os.path.join(BASE, "output/rl_titan_v4_sota/20260316_060537/training_curve.csv")
)
rewards_ppo = [float(r["episode_reward"]) for r in rows_ppo]
print("\nPPO Titan Reward Stats:")
print(
    f"  episodes={len(rows_ppo)}, min={min(rewards_ppo):.2f}, max={max(rewards_ppo):.2f}, mean={sum(rewards_ppo)/len(rewards_ppo):.2f}"
)
print("  First 5:", [round(r, 1) for r in rewards_ppo[:5]])
print("  Last 5:", [round(r, 1) for r in rewards_ppo[-5:]])

# PPO eval milestones
rows_eval = read_csv(os.path.join(BASE, "output/baselines/ppo_eval_200.csv"))
print("\nPPO Eval (200 gens) milestones:")
for idx in [0, 1, 4, 9, 19, 49, 99, 149, 199]:
    if idx < len(rows_eval):
        r = rows_eval[idx]
        print(
            f"  gen {r['generation']:>3s}: hard={r['best_hard']:>6s}, soft={r['best_soft'][:7]:>7s}, action={r['action_name']}"
        )

# DQN eval milestones
rows_dqn = read_csv(
    os.path.join(BASE, "output/rl_dqn/20260320_185249/evaluation_trajectory_200.csv")
)
print(f"\nDQN Eval ({len(rows_dqn)} gens) milestones:")
for idx in [0, 1, 4, 9, 14, 19, 24, 29]:
    if idx < len(rows_dqn):
        r = rows_dqn[idx]
        print(
            f"  gen {r['generation']:>3s}: hard={r['best_hard']:>6s}, soft={r['best_soft'][:7]:>7s}, action={r['action_name']}"
        )

# GA convergence milestones
for name, path in [
    ("GA Baseline", "output/ga_baseline/20260320_002246/convergence_history.csv"),
    ("GA Memetic", "output/ga_memetic/20260320_002553/convergence_history.csv"),
    ("GA Adaptive", "output/ga_adaptive/20260320_003837/convergence_history.csv"),
]:
    rows_ga = read_csv(os.path.join(BASE, path))
    print(f"\n{name} convergence (total {len(rows_ga)} gens):")
    for idx in [0, 4, 9, 24, 49, 99, 199, 299, 499]:
        if idx < len(rows_ga):
            r = rows_ga[idx]
            print(
                f"  gen {r['Gen']:>4s}: hard={r['Best_Hard']:>5s}, soft={r['Best_Soft']:>5s}"
            )
