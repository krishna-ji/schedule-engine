#!/usr/bin/env python3
r"""Phase 59 — Stochastic Evaluation of Titan V3 PPO Agent.

The Phase 57 forensic audit proved that deterministic evaluation
(argmax) collapses a near-uniform policy to a single action.  This
script evaluates the Titan V3 agent with ``deterministic=False`` so
the policy's learned distribution is actually utilized — if the agent
outputs 32% for SoftFocus and 33% for Intensified, it will actually
*roll the dice* and try both proportionally.

Also runs:
  - Multiple stochastic seeds (3 runs) to measure variance
  - One deterministic run for comparison
  - All 6 static baselines (25 gens, pop=120, seed=42)
  - Full probability forensics at each step (like verify_rl_brain.py)
  - Comparison table + plots

Usage::

    python runs/eval_titan_v3_stochastic.py

Requires: output/models/ppo_titan_v3.zip (from rl_08_titan_v3_overclock.py)
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_titan_v3")

# ======================================================================
# Configuration
# ======================================================================
MODEL_PATH = PROJECT_ROOT / "output" / "models" / "ppo_titan_v3.zip"
PKL_PATH = ".cache/events_with_domains.pkl"
EVAL_POP_SIZE = 120
EVAL_MAX_GEN = 25
EVAL_SEED = 42

# Stochastic evaluation seeds (3 runs for variance measurement)
STOCHASTIC_SEEDS = [42, 123, 777]

ACTION_NAMES = {
    0: "Conservative",
    1: "Aggressive",
    2: "Memetic",
    3: "SoftFocus",
    4: "Destructive",
    5: "Intensified",
}
ACTION_SHORT = {
    0: "Con", 1: "Agg", 2: "Mem", 3: "Sft", 4: "Des", 5: "Int",
}


# ======================================================================
# Evaluation Helpers
# ======================================================================


def run_ppo_eval(
    model,
    seed: int,
    deterministic: bool,
    label: str,
    verbose: bool = True,
) -> dict:
    """Run one PPO evaluation episode. Returns trajectory + stats."""
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_MAX_GEN,
        pop_size=EVAL_POP_SIZE,
        algorithm_name="nsga2",
        seed=seed,
    )
    obs, info = env.reset()

    rows = []
    actions_taken = []
    all_probs = []
    cumulative_reward = 0.0

    if verbose:
        mode_str = "deterministic" if deterministic else "STOCHASTIC"
        print(f"\n  [{label}] {mode_str} eval (seed={seed})")
        header = (
            f"    {'Step':>4} │ {'Hard':>5} {'Soft':>5} │ {'Action':>12} │"
            f" {'p(Con)':>6} {'p(Agg)':>6} {'p(Mem)':>6} {'p(Sft)':>6} {'p(Des)':>6} {'p(Int)':>6}"
            f" │ {'R':>7}"
        )
        print(header)
        print("    " + "─" * (len(header) - 4))

    for step in range(1, EVAL_MAX_GEN + 1):
        # --- Extract probabilities ---
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]

        all_probs.append(probs.copy())

        # --- Get action (stochastic or deterministic) ---
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        actions_taken.append(action)

        # --- Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        best_hard = info.get("best_hard", 0)
        best_soft = info.get("best_soft", 0)

        rows.append({
            "gen": step + 1,
            "action": action,
            "best_hard": best_hard,
            "best_soft": best_soft,
            "reward": reward,
        })

        if verbose:
            prob_str = " ".join(f"{p:6.3f}" for p in probs)
            name = ACTION_SHORT.get(action, "?")
            print(f"    {step:>4} │ {best_hard:>5.0f} {best_soft:>5.0f} │ {name:>12} │ {prob_str} │ {reward:>7.3f}")

        if terminated or truncated:
            if verbose:
                print(f"    [Episode ended at step {step}]")
            break

    env.close()

    # --- Compute stats ---
    prob_matrix = np.array(all_probs)
    final = rows[-1] if rows else {"best_hard": 0, "best_soft": 0}
    best_hard_val = min(r["best_hard"] for r in rows) if rows else 0
    best_hard_gen = min(range(len(rows)), key=lambda i: rows[i]["best_hard"]) if rows else 0
    soft_at_best = rows[best_hard_gen]["best_soft"] if rows else 0

    action_freq = Counter(actions_taken)

    result = {
        "label": label,
        "seed": seed,
        "deterministic": deterministic,
        "final_hard": final["best_hard"],
        "final_soft": final["best_soft"],
        "best_hard": best_hard_val,
        "soft_at_best_hard": soft_at_best,
        "best_hard_gen": best_hard_gen + 2,  # 1-indexed, offset by init
        "cumulative_reward": cumulative_reward,
        "actions_taken": actions_taken,
        "action_freq": dict(action_freq),
        "prob_matrix": prob_matrix,
        "prob_variance": prob_matrix.var(axis=0).sum(),
        "rows": rows,
    }

    if verbose:
        print(f"    Final: hard={final['best_hard']:.0f}, soft={final['best_soft']:.0f}")
        print(f"    Best hard: {best_hard_val:.0f} at gen {best_hard_gen + 2} (soft={soft_at_best:.0f})")
        print(f"    Cumulative R: {cumulative_reward:.3f}")
        print(f"    Prob variance: {result['prob_variance']:.6f}")
        freq_str = ", ".join(f"{ACTION_SHORT[a]}={action_freq.get(a, 0)}" for a in range(6))
        print(f"    Actions: {freq_str}")

    return result


def run_static_baseline(action_id: int, seed: int) -> dict:
    """Run a single static baseline."""
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_MAX_GEN,
        pop_size=EVAL_POP_SIZE,
        algorithm_name="nsga2",
        seed=seed,
    )
    obs, info = env.reset()

    rows = [{"gen": 1, "best_hard": info["best_hard"], "best_soft": info["best_soft"]}]

    for g in range(EVAL_MAX_GEN - 1):
        obs, reward, done, trunc, info = env.step(action_id)
        rows.append({
            "gen": info["generation"],
            "best_hard": info["best_hard"],
            "best_soft": info["best_soft"],
        })
        if done or trunc:
            break

    env.close()

    final = rows[-1]
    best_hard_val = min(r["best_hard"] for r in rows)
    best_hard_idx = min(range(len(rows)), key=lambda i: rows[i]["best_hard"])
    soft_at_best = rows[best_hard_idx]["best_soft"]

    return {
        "action_id": action_id,
        "name": ACTION_NAMES[action_id],
        "final_hard": final["best_hard"],
        "final_soft": final["best_soft"],
        "best_hard": best_hard_val,
        "soft_at_best_hard": soft_at_best,
        "rows": rows,
    }


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Full Titan V3 stochastic evaluation pipeline."""
    from stable_baselines3 import PPO

    # -- Load model --------------------------------------------------------
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}\nRun rl_08_titan_v3_overclock.py first!"
    logger.info("Loading model from %s", MODEL_PATH)
    model = PPO.load(str(MODEL_PATH))
    logger.info("Model loaded: %s", model.policy.mlp_extractor)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_titan_v3_eval" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print("  TITAN V3 STOCHASTIC EVALUATION — Phase 59")
    print("=" * 100)
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  Eval:  pop={EVAL_POP_SIZE}, gens={EVAL_MAX_GEN}")
    print(f"  Stochastic seeds: {STOCHASTIC_SEEDS}")
    print("=" * 100)

    t_total = time.perf_counter()
    all_results = []

    # ------------------------------------------------------------------
    # 1. Deterministic evaluation (for comparison)
    # ------------------------------------------------------------------
    logger.info("Running deterministic evaluation...")
    det_result = run_ppo_eval(model, EVAL_SEED, deterministic=True, label="DET")
    all_results.append(det_result)

    # ------------------------------------------------------------------
    # 2. Stochastic evaluations (3 seeds)
    # ------------------------------------------------------------------
    stoch_results = []
    for i, seed in enumerate(STOCHASTIC_SEEDS):
        logger.info("Running stochastic evaluation (seed=%d)...", seed)
        result = run_ppo_eval(
            model, seed, deterministic=False,
            label=f"STOCH-{i+1}",
        )
        stoch_results.append(result)
        all_results.append(result)

    # ------------------------------------------------------------------
    # 3. Static baselines (seed=42)
    # ------------------------------------------------------------------
    logger.info("Running 6 static baselines (seed=%d)...", EVAL_SEED)
    baselines = {}
    for aid in range(6):
        logger.info("  Baseline %d (%s)...", aid, ACTION_NAMES[aid])
        t0 = time.perf_counter()
        bl = run_static_baseline(aid, EVAL_SEED)
        elapsed = time.perf_counter() - t0
        logger.info("    → hard=%d soft=%d best_hard=%d (%.1fs)",
                     bl["final_hard"], bl["final_soft"], bl["best_hard"], elapsed)
        baselines[aid] = bl

    # ------------------------------------------------------------------
    # 4. Grand Comparison Table
    # ------------------------------------------------------------------
    print()
    print()
    print("=" * 100)
    print("  GRAND COMPARISON TABLE — Titan V3 vs Static Baselines")
    print("=" * 100)

    print(f"\n  {'Method':<28s} {'Hard@25':>8s} {'Soft@25':>8s} {'BestH':>6s} {'Soft@BH':>8s} {'Actions':>30s}")
    print("  " + "─" * 92)

    # Deterministic PPO
    r = det_result
    freq_str = ", ".join(f"{ACTION_SHORT[a]}={r['action_freq'].get(a, 0)}" for a in range(6) if r['action_freq'].get(a, 0) > 0)
    print(f"  {'PPO Determ (seed=42)':<28s} {r['final_hard']:>8.0f} {r['final_soft']:>8.0f} "
          f"{r['best_hard']:>6.0f} {r['soft_at_best_hard']:>8.0f} {freq_str:>30s}")

    # Stochastic PPO runs
    for r in stoch_results:
        freq_str = ", ".join(f"{ACTION_SHORT[a]}={r['action_freq'].get(a, 0)}" for a in range(6) if r['action_freq'].get(a, 0) > 0)
        print(f"  {'PPO Stoch (seed=' + str(r['seed']) + ')':<28s} {r['final_hard']:>8.0f} {r['final_soft']:>8.0f} "
              f"{r['best_hard']:>6.0f} {r['soft_at_best_hard']:>8.0f} {freq_str:>30s}")

    # Stochastic mean
    stoch_hards = [r["best_hard"] for r in stoch_results]
    stoch_softs = [r["soft_at_best_hard"] for r in stoch_results]
    stoch_final_h = [r["final_hard"] for r in stoch_results]
    stoch_final_s = [r["final_soft"] for r in stoch_results]
    print(f"  {'PPO Stoch (mean±std)':<28s} "
          f"{np.mean(stoch_final_h):>7.1f}±{np.std(stoch_final_h):>3.1f} "
          f"{np.mean(stoch_final_s):>7.1f}±{np.std(stoch_final_s):>3.1f} "
          f"{np.mean(stoch_hards):>5.1f}±{np.std(stoch_hards):.1f} "
          f"{np.mean(stoch_softs):>7.1f}±{np.std(stoch_softs):.1f}")

    print("  " + "─" * 92)

    # Static baselines
    for aid in range(6):
        bl = baselines[aid]
        print(f"  Static {aid} ({bl['name']:<14s}) {bl['final_hard']:>8.0f} {bl['final_soft']:>8.0f} "
              f"{bl['best_hard']:>6.0f} {bl['soft_at_best_hard']:>8.0f}")

    print("  " + "─" * 92)

    # ------------------------------------------------------------------
    # 5. State-Dependency Verdict
    # ------------------------------------------------------------------
    print()
    print("  STATE-DEPENDENCY ANALYSIS:")
    for r in all_results:
        label = r["label"]
        pv = r["prob_variance"]
        if pv > 0.01:
            verdict = "STATE-DEPENDENT"
        elif pv > 0.001:
            verdict = "WEAK state-dep"
        else:
            verdict = "STATIC"
        print(f"    {label:<12s}: prob_variance={pv:.6f} → {verdict}")

    # ------------------------------------------------------------------
    # 6. Stochastic vs Deterministic Comparison
    # ------------------------------------------------------------------
    print()
    print("  STOCHASTIC vs DETERMINISTIC:")
    det_h = det_result["best_hard"]
    det_s = det_result["soft_at_best_hard"]
    mean_stoch_h = np.mean(stoch_hards)
    mean_stoch_s = np.mean(stoch_softs)
    print(f"    Deterministic: best_hard={det_h:.0f}, soft={det_s:.0f}")
    print(f"    Stochastic:    best_hard={mean_stoch_h:.1f}±{np.std(stoch_hards):.1f}, "
          f"soft={mean_stoch_s:.1f}±{np.std(stoch_softs):.1f}")
    if mean_stoch_h < det_h:
        print(f"    >>> STOCHASTIC IS BETTER by {det_h - mean_stoch_h:.1f} hard points!")
    elif mean_stoch_h > det_h:
        print(f"    >>> DETERMINISTIC IS BETTER by {mean_stoch_h - det_h:.1f} hard points")
    else:
        print(f"    >>> TIE on hard — check soft constraints")

    # ------------------------------------------------------------------
    # 7. Pareto Dominance Check
    # ------------------------------------------------------------------
    print()
    print("  PARETO DOMINANCE CHECK (best stochastic vs best static):")
    best_stoch = min(stoch_results, key=lambda r: (r["best_hard"], r["soft_at_best_hard"]))
    best_static = min(baselines.values(), key=lambda b: (b["best_hard"], b["soft_at_best_hard"]))
    print(f"    Best stochastic: hard={best_stoch['best_hard']:.0f}, soft={best_stoch['soft_at_best_hard']:.0f}")
    print(f"    Best static:     hard={best_static['best_hard']:.0f}, soft={best_static['soft_at_best_hard']:.0f} ({best_static['name']})")

    if best_stoch["best_hard"] < best_static["best_hard"]:
        print("    >>> PPO WINS on hard!")
    elif best_stoch["best_hard"] == best_static["best_hard"]:
        if best_stoch["soft_at_best_hard"] < best_static["soft_at_best_hard"]:
            print("    >>> PPO WINS — same hard, better soft!")
        else:
            print("    >>> TIE or static wins on soft at same hard level")
    else:
        print("    >>> STATIC WINS on hard. PPO needs more training.")

    # ------------------------------------------------------------------
    # 8. Save results CSV
    # ------------------------------------------------------------------
    csv_path = run_dir / "titan_v3_eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed", "deterministic", "final_hard", "final_soft",
                         "best_hard", "soft_at_best_hard", "cumulative_reward", "prob_variance"])
        for r in all_results:
            writer.writerow([
                r["label"], r["seed"], r["deterministic"],
                r["final_hard"], r["final_soft"],
                r["best_hard"], r["soft_at_best_hard"],
                r["cumulative_reward"], r["prob_variance"],
            ])
        for aid, bl in baselines.items():
            writer.writerow([
                f"Static_{aid}_{bl['name']}", EVAL_SEED, True,
                bl["final_hard"], bl["final_soft"],
                bl["best_hard"], bl["soft_at_best_hard"],
                0, 0,
            ])
    logger.info("Results saved: %s", csv_path)

    # ------------------------------------------------------------------
    # 9. Generate comparison plot
    # ------------------------------------------------------------------
    try:
        _generate_plot(det_result, stoch_results, baselines, run_dir)
    except Exception as e:
        logger.warning("Plot generation failed: %s", e)

    total_time = time.perf_counter() - t_total
    print()
    print("=" * 100)
    print(f"  Evaluation complete in {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Output: {run_dir}")
    print("=" * 100)


def _generate_plot(
    det_result: dict,
    stoch_results: list[dict],
    baselines: dict[int, dict],
    run_dir: Path,
) -> None:
    """Generate comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {
        0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c",
        3: "#d62728", 4: "#9467bd", 5: "#8c564b",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]

    # Static baselines
    for aid in range(6):
        bl = baselines[aid]
        gens = [r["gen"] for r in bl["rows"]]
        hard = [r["best_hard"] for r in bl["rows"]]
        ax.plot(gens, hard, color=colors[aid], alpha=0.4, linewidth=1.0,
                linestyle="--", label=f"Static {ACTION_SHORT[aid]}")

    # Deterministic PPO
    det_gens = [r["gen"] for r in det_result["rows"]]
    det_hard = [r["best_hard"] for r in det_result["rows"]]
    ax.plot(det_gens, det_hard, color="black", linewidth=2.0,
            label="PPO Deterministic", zorder=10)

    # Stochastic PPO runs
    stoch_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, sr in enumerate(stoch_results):
        s_gens = [r["gen"] for r in sr["rows"]]
        s_hard = [r["best_hard"] for r in sr["rows"]]
        ax.plot(s_gens, s_hard, color=stoch_colors[i % len(stoch_colors)],
                linewidth=1.5, alpha=0.8,
                label=f"PPO Stoch (seed={sr['seed']})", zorder=9)

    ax.set_ylabel("Best Hard Penalty")
    ax.set_title("Titan V3: PPO (Det + Stochastic) vs Static Baselines")
    ax.legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Bottom: action sequence from best stochastic run
    ax2 = axes[1]
    best_stoch = min(stoch_results, key=lambda r: r["best_hard"])
    action_gens = list(range(2, 2 + len(best_stoch["actions_taken"])))
    action_colors_list = [colors[a] for a in best_stoch["actions_taken"]]
    ax2.bar(action_gens, [1] * len(best_stoch["actions_taken"]),
            color=action_colors_list, width=1.0, edgecolor="none")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Action")
    ax2.set_yticks([])
    ax2.set_title(f"Best Stochastic Run Action Sequence (seed={best_stoch['seed']})", fontsize=10)
    legend_elements = [Patch(facecolor=colors[a], label=f"{a}: {ACTION_NAMES[a]}")
                       for a in range(6)]
    ax2.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=7)

    plt.tight_layout()
    pdf_path = run_dir / "fig_titan_v3_comparison.pdf"
    png_path = run_dir / "fig_titan_v3_comparison.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Plot saved: %s", pdf_path)


if __name__ == "__main__":
    main()
