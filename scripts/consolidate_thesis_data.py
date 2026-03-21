"""
Consolidate all experiment data into CSV/JSON for thesis results chapter.
Generates: results/thesis_consolidated/
  - master_comparison.csv
  - ga_convergence_overlay.csv
  - llh_action_profiles.csv
  - ppo_eval_milestones.csv
  - dqn_eval_milestones.csv
  - dqn_training_summary.json
  - ppo_training_summary.json
  - vectorized_speedups.csv
  - bench_solver_comparison.csv
"""

import csv
import json
import os
import statistics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "thesis_consolidated")
os.makedirs(OUT, exist_ok=True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path}")


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {path}")


# ═══════════════════════════════════════════════════════════════════
# 1. Master Comparison Table
# ═══════════════════════════════════════════════════════════════════
print("1. Master Comparison Table")

ga_base = read_json(
    os.path.join(BASE, "output/ga_baseline/20260320_002246/results.json")
)
ga_mem = read_json(os.path.join(BASE, "output/ga_memetic/20260320_002553/results.json"))
ga_adap = read_json(
    os.path.join(BASE, "output/ga_adaptive/20260320_003837/results.json")
)
llh = read_json(
    os.path.join(BASE, "output/rl_llh_differentiation/llh_differentiation_results.json")
)

# Static baselines final
static_rows = read_csv(
    os.path.join(BASE, "output/rl_static_baselines/static_baselines.csv")
)
static_last = static_rows[-1]

# PPO eval final
ppo_rows = read_csv(os.path.join(BASE, "output/baselines/ppo_eval_200.csv"))
ppo_last = ppo_rows[-1]

# DQN eval final
dqn_rows = read_csv(
    os.path.join(BASE, "output/rl_dqn/20260320_185249/evaluation_trajectory_200.csv")
)
dqn_last = dqn_rows[-1]

master = [
    {
        "method": "GA Baseline",
        "paradigm": "GA",
        "pop_size": 100,
        "generations": 200,
        "best_hard": ga_base["best_hard"],
        "best_soft": round(ga_base["best_soft"], 1),
        "elapsed_s": round(ga_base["elapsed_s"], 1),
        "sec_per_gen": round(ga_base["sec_per_gen"], 3),
        "uses_repair": "No",
    },
    {
        "method": "GA Memetic",
        "paradigm": "GA",
        "pop_size": 120,
        "generations": 500,
        "best_hard": ga_mem["best_hard"],
        "best_soft": round(ga_mem["best_soft"], 1),
        "elapsed_s": round(ga_mem["elapsed_s"], 1),
        "sec_per_gen": round(ga_mem["sec_per_gen"], 3),
        "uses_repair": "Yes",
    },
    {
        "method": "GA Adaptive",
        "paradigm": "GA",
        "pop_size": 100,
        "generations": 300,
        "best_hard": ga_adap["best_hard"],
        "best_soft": round(ga_adap["best_soft"], 1),
        "elapsed_s": round(ga_adap["elapsed_s"], 1),
        "sec_per_gen": round(ga_adap["sec_per_gen"], 3),
        "uses_repair": "Yes",
    },
    {
        "method": "RL Static (Conservative)",
        "paradigm": "RL",
        "pop_size": 120,
        "generations": 900,
        "best_hard": float(static_last["best_hard"]),
        "best_soft": round(float(static_last["best_soft"]), 1),
        "elapsed_s": "-",
        "sec_per_gen": "-",
        "uses_repair": "Yes",
    },
    {
        "method": "RL PPO (Eval 200)",
        "paradigm": "RL",
        "pop_size": 120,
        "generations": 200,
        "best_hard": float(ppo_last["best_hard"]),
        "best_soft": round(float(ppo_last["best_soft"]), 1),
        "elapsed_s": "-",
        "sec_per_gen": "-",
        "uses_repair": "Yes",
    },
    {
        "method": "RL DQN (Eval 30)",
        "paradigm": "RL",
        "pop_size": 120,
        "generations": 30,
        "best_hard": float(dqn_last["best_hard"]),
        "best_soft": round(float(dqn_last["best_soft"]), 1),
        "elapsed_s": "-",
        "sec_per_gen": "-",
        "uses_repair": "Yes",
    },
]

fields = [
    "method",
    "paradigm",
    "pop_size",
    "generations",
    "best_hard",
    "best_soft",
    "elapsed_s",
    "sec_per_gen",
    "uses_repair",
]
write_csv(os.path.join(OUT, "master_comparison.csv"), master, fields)

# ═══════════════════════════════════════════════════════════════════
# 2. GA Convergence Overlay (all 3 modes, aligned by generation)
# ═══════════════════════════════════════════════════════════════════
print("2. GA Convergence Overlay")

ga_base_conv = read_csv(
    os.path.join(BASE, "output/ga_baseline/20260320_002246/convergence_history.csv")
)
ga_mem_conv = read_csv(
    os.path.join(BASE, "output/ga_memetic/20260320_002553/convergence_history.csv")
)
ga_adap_conv = read_csv(
    os.path.join(BASE, "output/ga_adaptive/20260320_003837/convergence_history.csv")
)


# Build lookup maps
def conv_map(rows, hard_key="Best_Hard", soft_key="Best_Soft"):
    return {int(r["Gen"]): (float(r[hard_key]), float(r[soft_key])) for r in rows}


base_m, mem_m, adap_m = (
    conv_map(ga_base_conv),
    conv_map(ga_mem_conv),
    conv_map(ga_adap_conv),
)
all_gens = sorted(set(base_m.keys()) | set(mem_m.keys()) | set(adap_m.keys()))

overlay_rows = []
for g in all_gens:
    row = {"gen": g}
    if g in base_m:
        row["baseline_hard"], row["baseline_soft"] = base_m[g]
    if g in mem_m:
        row["memetic_hard"], row["memetic_soft"] = mem_m[g]
    if g in adap_m:
        row["adaptive_hard"], row["adaptive_soft"] = adap_m[g]
    overlay_rows.append(row)

write_csv(
    os.path.join(OUT, "ga_convergence_overlay.csv"),
    overlay_rows,
    [
        "gen",
        "baseline_hard",
        "baseline_soft",
        "memetic_hard",
        "memetic_soft",
        "adaptive_hard",
        "adaptive_soft",
    ],
)

# ═══════════════════════════════════════════════════════════════════
# 3. LLH Action Profiles
# ═══════════════════════════════════════════════════════════════════
print("3. LLH Action Profiles")

llh_rows = []
for aid, info in sorted(llh["actions"].items(), key=lambda x: int(x[0])):
    llh_rows.append(
        {
            "action_id": int(aid),
            "name": info["name"],
            "best_hard": info["best_hard"],
            "best_soft": info["best_soft"],
            "mean_hard": info["mean_hard"],
            "time_s": round(info["time_s"], 0),
        }
    )
write_csv(
    os.path.join(OUT, "llh_action_profiles.csv"),
    llh_rows,
    ["action_id", "name", "best_hard", "best_soft", "mean_hard", "time_s"],
)

# ═══════════════════════════════════════════════════════════════════
# 4. Eval Milestones (PPO + DQN)
# ═══════════════════════════════════════════════════════════════════
print("4. Eval Milestones")


def extract_milestones(rows, gen_key="generation"):
    milestones = []
    for r in rows:
        milestones.append(
            {
                "generation": int(float(r[gen_key])),
                "best_hard": float(r["best_hard"]),
                "best_soft": round(float(r["best_soft"]), 1),
                "action_name": r.get("action_name", ""),
            }
        )
    return milestones


write_csv(
    os.path.join(OUT, "ppo_eval_milestones.csv"),
    extract_milestones(ppo_rows),
    ["generation", "best_hard", "best_soft", "action_name"],
)

write_csv(
    os.path.join(OUT, "dqn_eval_milestones.csv"),
    extract_milestones(dqn_rows),
    ["generation", "best_hard", "best_soft", "action_name"],
)

# ═══════════════════════════════════════════════════════════════════
# 5. Training Summaries (DQN + PPO)
# ═══════════════════════════════════════════════════════════════════
print("5. Training Summaries")

dqn_train = read_csv(
    os.path.join(BASE, "output/rl_dqn/20260320_185249/training_curve.csv")
)
dqn_rewards = [float(r["episode_reward"]) for r in dqn_train]
write_json(
    os.path.join(OUT, "dqn_training_summary.json"),
    {
        "total_episodes": len(dqn_rewards),
        "min_reward": round(min(dqn_rewards), 2),
        "max_reward": round(max(dqn_rewards), 2),
        "mean_reward": round(statistics.mean(dqn_rewards), 2),
        "std_reward": round(statistics.stdev(dqn_rewards), 2),
        "rewards": [round(r, 2) for r in dqn_rewards],
    },
)

ppo_train = read_csv(
    os.path.join(BASE, "output/rl_titan_v4_sota/20260316_060537/training_curve.csv")
)
ppo_rewards = [float(r["episode_reward"]) for r in ppo_train]
write_json(
    os.path.join(OUT, "ppo_training_summary.json"),
    {
        "total_episodes": len(ppo_rewards),
        "min_reward": round(min(ppo_rewards), 2),
        "max_reward": round(max(ppo_rewards), 2),
        "mean_reward": round(statistics.mean(ppo_rewards), 2),
        "std_reward": round(statistics.stdev(ppo_rewards), 2),
        "rewards": [round(r, 2) for r in ppo_rewards],
    },
)

# ═══════════════════════════════════════════════════════════════════
# 6. Vectorized Speedups
# ═══════════════════════════════════════════════════════════════════
print("6. Vectorized Speedups")

vbench = read_json(os.path.join(BASE, "results/bench_eval_vectorized.json"))
speed_rows = []
for b in vbench["benchmarks"]:
    speed_rows.append(
        {
            "pop_size": b["N"],
            "batch_ms": round(b["batch"]["per_ind_mean_ms"], 3),
            "vectorized_ms": round(b["vectorized"]["per_ind_mean_ms"], 3),
            "speedup": b["speedup"],
        }
    )
write_csv(
    os.path.join(OUT, "vectorized_speedups.csv"),
    speed_rows,
    ["pop_size", "batch_ms", "vectorized_ms", "speedup"],
)

# ═══════════════════════════════════════════════════════════════════
# 7. Bench Solver Comparison
# ═══════════════════════════════════════════════════════════════════
print("7. Solver Comparison")

summary = read_json(os.path.join(BASE, "results/bench_compare/summary.json"))
solver_rows = []
for solver in ["pymoo", "deap"]:
    s = summary["aggregate"][solver]
    solver_rows.append(
        {
            "solver": solver,
            "n_runs": s["n_runs"],
            "median_best_hard": s["median_best_hard"],
            "median_best_soft": s["median_best_soft"],
            "mean_best_hard": round(s["mean_best_hard"], 1),
            "median_sec_per_gen": round(s["median_sec_per_gen"], 2),
            "mean_elapsed_s": round(s["mean_elapsed_s"], 1),
        }
    )
write_csv(
    os.path.join(OUT, "bench_solver_comparison.csv"),
    solver_rows,
    [
        "solver",
        "n_runs",
        "median_best_hard",
        "median_best_soft",
        "mean_best_hard",
        "median_sec_per_gen",
        "mean_elapsed_s",
    ],
)

print("\n✓ All consolidated files written to:", OUT)
