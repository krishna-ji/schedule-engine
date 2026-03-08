#!/usr/bin/env python3
r"""Phase 57 — PPO Training on Validated 6-LLH Space.

Phase 56 proved LLH differentiation:
  - Hard range: 65–69 (4 points) — Destructive wins early, Conservative late
  - Soft range: 260–320 (60 points) — SoftFocus dominates
  - No single LLH dominates all 50 gens → RL can learn sequencing

This script:
  1. Trains PPO with phase-transition reward (amplified soft post-convergence)
  2. Evaluates deterministic policy against all 6 static baselines (pop=120, 25 gens)
  3. Prints the learned action sequence across generations
  4. Generates fig_04_baseline_comparison.pdf

Reduced episode config for tractable training:
  - max_generations=25 (differentiation is in gens 1–25)
  - pop_size=60 (half compute per gen)
  - total_timesteps=10_000 (200 episodes, ~8 hours)

Usage::

    python -m runs.rl_07_ppo_phase57

If first run shows no learning, rerun with TOTAL_TIMESTEPS=20_000.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_07_ppo_phase57")

# ======================================================================
# Configuration
# ======================================================================

# -- Training -----------------------------------------------------------
TRAIN_POP_SIZE = 40  # Minimal pop → fastest episodes (~3-5s/step)
TRAIN_MAX_GEN = 25  # Interesting differentiation in gens 1–25
TOTAL_TIMESTEPS = 2_500  # Budget (~104 episodes, ~6-8h wall-clock)
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
NET_ARCH = [64, 64]  # 2-layer MLP
N_STEPS = 24  # PPO rollout buffer = 1 episode (24 steps)
BATCH_SIZE = 24  # Full episode per mini-batch
N_EPOCHS = 10  # PPO update epochs per rollout
GAE_LAMBDA = 0.95
GAMMA = 0.99
ENT_COEF = 0.01  # Encourage exploration across 6 actions

# -- Evaluation (matches Phase 56 conditions) ---------------------------
EVAL_POP_SIZE = 120  # Full pop for fair comparison
EVAL_MAX_GEN = 25  # Same horizon as training
EVAL_SEED = 42  # Deterministic comparison
PKL_PATH = ".cache/events_with_domains.pkl"

# -- Static baseline reference from Phase 56 ----------------------------
# Best-ever hard per LLH (50 gens, pop=120, seed=42):
PHASE56_BASELINES = {
    0: {"name": "Conservative", "best_hard": 67, "soft_at_best": 308, "at_gen": 39},
    1: {"name": "Aggressive", "best_hard": 67, "soft_at_best": 309, "at_gen": 32},
    2: {"name": "Memetic", "best_hard": 66, "soft_at_best": 316, "at_gen": 42},
    3: {"name": "SoftFocus", "best_hard": 65, "soft_at_best": 247, "at_gen": 15},
    4: {"name": "Destructive", "best_hard": 65, "soft_at_best": 277, "at_gen": 6},
    5: {"name": "Intensified", "best_hard": 66, "soft_at_best": 310, "at_gen": 9},
}

ACTION_SHORT = {
    0: "Con",
    1: "Agg",
    2: "Mem",
    3: "Sft",
    4: "Des",
    5: "Int",
}


# ======================================================================
# 1. Training
# ======================================================================


def train(run_dir: Path) -> object:
    """Train PPO on the 6-LLH hyper-heuristic env."""
    from stable_baselines3 import PPO

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv
    from src.rl.training.thesis_callback import ThesisLoggingCallback

    logger.info("=" * 72)
    logger.info("Phase 57 — PPO Training on Validated 6-LLH Space")
    logger.info("  pop=%d  max_gen=%d  timesteps=%d  lr=%.0e",
                TRAIN_POP_SIZE, TRAIN_MAX_GEN, TOTAL_TIMESTEPS, LEARNING_RATE)
    logger.info("  n_steps=%d  batch=%d  epochs=%d  ent_coef=%.3f",
                N_STEPS, BATCH_SIZE, N_EPOCHS, ENT_COEF)
    logger.info("  run_dir: %s", run_dir)
    logger.info("=" * 72)

    # -- Environment (random seed per episode for diversity) ---------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=TRAIN_MAX_GEN,
        pop_size=TRAIN_POP_SIZE,
        algorithm_name="nsga2",
        seed=42,  # Will be varied by SB3 auto-reset
    )

    # -- PPO Agent --------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gae_lambda=GAE_LAMBDA,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=42,
        verbose=1,
    )

    # -- Callback ---------------------------------------------------------
    callback = ThesisLoggingCallback(run_dir=run_dir, verbose=1)

    # -- Train ------------------------------------------------------------
    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    train_time = time.perf_counter() - t0

    logger.info("Training complete in %.1fs (%.1f min)", train_time, train_time / 60)

    # -- Save model -------------------------------------------------------
    model_path = run_dir / "ppo_phase57.zip"
    model.save(str(model_path))
    logger.info("Model saved: %s", model_path)

    env.close()
    return model


# ======================================================================
# 2. Static Baseline Evaluation (25 gens, pop=120)
# ======================================================================


def run_static_baselines(run_dir: Path) -> dict[int, list[dict]]:
    """Run all 6 static baselines with eval config for fair comparison."""
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    logger.info("-" * 72)
    logger.info("Running 6 static baselines (pop=%d, gens=%d, seed=%d)",
                EVAL_POP_SIZE, EVAL_MAX_GEN, EVAL_SEED)
    logger.info("-" * 72)

    all_baselines: dict[int, list[dict]] = {}

    for action_id in range(6):
        name = PHASE56_BASELINES[action_id]["name"]
        logger.info("  Baseline %d (%s)...", action_id, name)

        env = PymooHyperHeuristicEnv(
            pkl_path=PKL_PATH,
            max_generations=EVAL_MAX_GEN,
            pop_size=EVAL_POP_SIZE,
            algorithm_name="nsga2",
            seed=EVAL_SEED,
        )
        obs, info = env.reset()
        t0 = time.perf_counter()

        rows = [{
            "gen": 1,
            "best_hard": info["best_hard"],
            "best_soft": info["best_soft"],
            "mean_hard": info["mean_hard"],
        }]

        for g in range(EVAL_MAX_GEN - 1):
            obs, reward, done, trunc, info = env.step(action_id)
            rows.append({
                "gen": info["generation"],
                "best_hard": info["best_hard"],
                "best_soft": info["best_soft"],
                "mean_hard": info["mean_hard"],
            })
            if done or trunc:
                break

        elapsed = time.perf_counter() - t0
        final = rows[-1]
        best_hard = min(r["best_hard"] for r in rows)
        logger.info("    → hard=%d soft=%d mean_hard=%d (%.1fs)",
                     final["best_hard"], final["best_soft"],
                     final["mean_hard"], elapsed)

        all_baselines[action_id] = rows
        env.close()

    # Save baselines CSV
    csv_path = run_dir / "static_baselines_25gen.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["action_id", "action_name", "gen", "best_hard", "best_soft", "mean_hard"])
        for aid, rows in all_baselines.items():
            name = PHASE56_BASELINES[aid]["name"]
            for r in rows:
                writer.writerow([aid, name, r["gen"], r["best_hard"], r["best_soft"], r["mean_hard"]])
    logger.info("Static baselines saved: %s", csv_path)

    return all_baselines


# ======================================================================
# 3. PPO Evaluation (deterministic, 25 gens, pop=120)
# ======================================================================


def evaluate_ppo(model, run_dir: Path) -> tuple[list[dict], list[int]]:
    """Run deterministic PPO evaluation and return trajectory + action sequence."""
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    logger.info("-" * 72)
    logger.info("PPO Deterministic Evaluation (pop=%d, gens=%d, seed=%d)",
                EVAL_POP_SIZE, EVAL_MAX_GEN, EVAL_SEED)
    logger.info("-" * 72)

    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_MAX_GEN,
        pop_size=EVAL_POP_SIZE,
        algorithm_name="nsga2",
        seed=EVAL_SEED,
    )
    obs, info = env.reset()

    rows = [{
        "gen": 1,
        "action": -1,
        "action_name": "init",
        "best_hard": info["best_hard"],
        "best_soft": info["best_soft"],
        "mean_hard": info["mean_hard"],
        "reward": 0.0,
    }]
    actions_taken: list[int] = []
    cumulative_reward = 0.0

    for g in range(EVAL_MAX_GEN - 1):
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        actions_taken.append(action)

        obs, reward, done, trunc, info = env.step(action)
        cumulative_reward += reward

        rows.append({
            "gen": info["generation"],
            "action": action,
            "action_name": ACTION_SHORT.get(action, f"a{action}"),
            "best_hard": info["best_hard"],
            "best_soft": info["best_soft"],
            "mean_hard": info["mean_hard"],
            "reward": reward,
        })

        logger.info(
            "  Gen %2d | act=%d (%s) | hard=%3.0f soft=%3.0f | R=%.4f",
            info["generation"], action, ACTION_SHORT.get(action, "?"),
            info["best_hard"], info["best_soft"], reward,
        )

        if done or trunc:
            break

    env.close()

    # Save evaluation CSV
    csv_path = run_dir / "ppo_evaluation_trajectory.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("PPO eval saved: %s (cumR=%.4f)", csv_path, cumulative_reward)

    return rows, actions_taken


# ======================================================================
# 4. Analysis & Comparison
# ======================================================================


def print_comparison(
    ppo_rows: list[dict],
    actions_taken: list[int],
    baselines: dict[int, list[dict]],
    run_dir: Path,
) -> None:
    """Print comprehensive comparison table and action sequence analysis."""

    print("\n")
    print("=" * 80)
    print("  PHASE 57 RESULTS: PPO vs Static Baselines (25 gens, pop=120, seed=42)")
    print("=" * 80)

    # -- Action Sequence --------------------------------------------------
    print("\n  LEARNED ACTION SEQUENCE")
    print("  " + "-" * 60)
    seq_str = " → ".join(ACTION_SHORT.get(a, f"a{a}") for a in actions_taken)
    print(f"  {seq_str}")

    # Action frequency
    freq = defaultdict(int)
    for a in actions_taken:
        freq[a] += 1
    print("\n  Action Frequency:")
    for aid in range(6):
        name = PHASE56_BASELINES[aid]["name"]
        cnt = freq.get(aid, 0)
        pct = 100.0 * cnt / len(actions_taken)
        bar = "█" * int(pct / 2)
        print(f"    {aid} ({name:15s}): {cnt:3d} ({pct:5.1f}%) {bar}")

    # Phase analysis
    n = len(actions_taken)
    early = actions_taken[:n // 3]
    mid = actions_taken[n // 3: 2 * n // 3]
    late = actions_taken[2 * n // 3:]

    def _mode(lst):
        if not lst:
            return -1
        c = defaultdict(int)
        for x in lst:
            c[x] += 1
        return max(c, key=c.get)

    print(f"\n  Phase Behavior:")
    print(f"    Early (gen 2-{n // 3 + 1}):  mode={_mode(early)} ({ACTION_SHORT.get(_mode(early), '?')})")
    print(f"    Mid   (gen {n // 3 + 2}-{2 * n // 3 + 1}):  mode={_mode(mid)} ({ACTION_SHORT.get(_mode(mid), '?')})")
    print(f"    Late  (gen {2 * n // 3 + 2}-{n + 1}): mode={_mode(late)} ({ACTION_SHORT.get(_mode(late), '?')})")

    # -- Comparison Table -------------------------------------------------
    ppo_final = ppo_rows[-1]
    ppo_best_hard = min(r["best_hard"] for r in ppo_rows)
    ppo_soft_at_best = None
    for r in ppo_rows:
        if r["best_hard"] == ppo_best_hard:
            ppo_soft_at_best = r["best_soft"]
            break

    print("\n  FINAL COMPARISON TABLE (gen 25)")
    print("  " + "-" * 76)
    print(f"  {'Method':<25s} {'Hard@25':>8s} {'Soft@25':>8s} {'BestHard':>9s} {'SoftAtBest':>11s} {'vs PPO':>7s}")
    print("  " + "-" * 76)
    print(f"  {'**PPO Adaptive**':<25s} {ppo_final['best_hard']:>8.0f} {ppo_final['best_soft']:>8.0f} "
          f"{ppo_best_hard:>9.0f} {ppo_soft_at_best:>11.0f} {'—':>7s}")

    for aid in range(6):
        name = PHASE56_BASELINES[aid]["name"]
        bl_rows = baselines[aid]
        bl_final = bl_rows[-1]
        bl_best_hard = min(r["best_hard"] for r in bl_rows)
        bl_soft_at_best = None
        for r in bl_rows:
            if r["best_hard"] == bl_best_hard:
                bl_soft_at_best = r["best_soft"]
                break

        delta = bl_final["best_hard"] - ppo_final["best_hard"]
        sign = "+" if delta > 0 else ""
        print(f"  Static {aid} ({name:<14s}) {bl_final['best_hard']:>8.0f} {bl_final['best_soft']:>8.0f} "
              f"{bl_best_hard:>9.0f} {bl_soft_at_best:>11.0f} {sign}{delta:>6.0f}")

    print("  " + "-" * 76)

    # -- Success Criterion ------------------------------------------------
    print("\n  SUCCESS CRITERION:")
    print(f"    PPO hard={ppo_final['best_hard']:.0f}, soft={ppo_final['best_soft']:.0f}")

    # Check Pareto dominance
    pareto_better = True
    for aid in range(6):
        bl = baselines[aid][-1]
        if ppo_final["best_hard"] <= bl["best_hard"] and ppo_final["best_soft"] < bl["best_soft"]:
            pass  # PPO dominates this baseline
        elif ppo_final["best_hard"] < bl["best_hard"] and ppo_final["best_soft"] <= bl["best_soft"]:
            pass  # PPO dominates this baseline
        else:
            pareto_better = False

    # Check if PPO is Pareto-non-dominated (no baseline is strictly better on BOTH)
    dominated = False
    for aid in range(6):
        bl = baselines[aid][-1]
        if bl["best_hard"] <= ppo_final["best_hard"] and bl["best_soft"] < ppo_final["best_soft"]:
            dominated = True
            break
        if bl["best_hard"] < ppo_final["best_hard"] and bl["best_soft"] <= ppo_final["best_soft"]:
            dominated = True
            break

    if pareto_better:
        print("    >>> PARETO DOMINANCE: PPO strictly dominates ALL static baselines!")
    elif not dominated:
        print("    >>> PARETO NON-DOMINATED: PPO is not dominated by any static baseline.")
        # Find where PPO sits in the hard-soft Pareto front
        best_soft_at_ppo_hard = float("inf")
        for aid in range(6):
            bl = baselines[aid][-1]
            if bl["best_hard"] <= ppo_final["best_hard"]:
                best_soft_at_ppo_hard = min(best_soft_at_ppo_hard, bl["best_soft"])
        if ppo_final["best_soft"] < best_soft_at_ppo_hard:
            print(f"    >>> PPO achieves BETTER soft ({ppo_final['best_soft']:.0f}) than best "
                  f"static at same hard level ({best_soft_at_ppo_hard:.0f})!")
    else:
        print("    >>> PPO is DOMINATED by at least one static baseline. More training needed.")

    # -- Per-gen trajectory comparison ------------------------------------
    print("\n  PER-GENERATION TRAJECTORY (best_hard)")
    print("  " + "-" * 70)
    header = f"  {'Gen':>4s}"
    for aid in range(6):
        header += f"  {ACTION_SHORT[aid]:>6s}"
    header += f"  {'PPO':>6s}  {'PPO_act':>7s}"
    print(header)
    print("  " + "-" * 70)

    max_gen = min(len(ppo_rows), min(len(baselines[a]) for a in range(6)))
    for g in range(max_gen):
        line = f"  {g + 1:>4d}"
        for aid in range(6):
            line += f"  {baselines[aid][g]['best_hard']:>6.0f}"
        line += f"  {ppo_rows[g]['best_hard']:>6.0f}"
        if g > 0 and g - 1 < len(actions_taken):
            line += f"  {ACTION_SHORT.get(actions_taken[g - 1], '?'):>7s}"
        else:
            line += f"  {'init':>7s}"
        print(line)

    print("  " + "-" * 70)


# ======================================================================
# 5. Plot Generation
# ======================================================================


def generate_comparison_plot(
    ppo_rows: list[dict],
    baselines: dict[int, list[dict]],
    actions_taken: list[int],
    run_dir: Path,
) -> Path:
    """Generate fig_04_baseline_comparison.pdf with all 7 convergence lines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        logger.warning("matplotlib not available — skipping plot generation")
        return run_dir / "fig_04_baseline_comparison.pdf"

    # -- Style setup (thesis-grade) ----------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # -- Top panel: Hard constraints ---------------------------------------
    ax1 = axes[0]
    colors = {
        0: "#1f77b4",  # Conservative - blue
        1: "#ff7f0e",  # Aggressive - orange
        2: "#2ca02c",  # Memetic - green
        3: "#d62728",  # SoftFocus - red
        4: "#9467bd",  # Destructive - purple
        5: "#8c564b",  # Intensified - brown
    }

    for aid in range(6):
        name = PHASE56_BASELINES[aid]["name"]
        gens = [r["gen"] for r in baselines[aid]]
        hard = [r["best_hard"] for r in baselines[aid]]
        ax1.plot(gens, hard, color=colors[aid], alpha=0.5, linewidth=1.0,
                 linestyle="--", label=f"Static {aid} ({name})")

    # PPO line (thick, black)
    ppo_gens = [r["gen"] for r in ppo_rows]
    ppo_hard = [r["best_hard"] for r in ppo_rows]
    ax1.plot(ppo_gens, ppo_hard, color="black", linewidth=2.5,
             label="PPO Adaptive", zorder=10)

    ax1.set_ylabel("Best Hard Penalty")
    ax1.set_title("Phase 57: PPO vs Static Baselines — Hard Constraint Convergence")
    ax1.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- Second y-axis for soft on top panel --------------------------------
    ax1b = ax1.twinx()
    ppo_soft = [r["best_soft"] for r in ppo_rows]
    ax1b.plot(ppo_gens, ppo_soft, color="black", linewidth=1.5,
              linestyle=":", alpha=0.6, label="PPO Soft")
    for aid in [3]:  # Just SoftFocus for comparison
        name = PHASE56_BASELINES[aid]["name"]
        soft = [r["best_soft"] for r in baselines[aid]]
        gens = [r["gen"] for r in baselines[aid]]
        ax1b.plot(gens, soft, color=colors[aid], alpha=0.4, linewidth=1.0,
                  linestyle=":", label=f"{name} Soft")
    ax1b.set_ylabel("Best Soft Penalty", color="gray")
    ax1b.tick_params(axis="y", labelcolor="gray")
    ax1b.legend(loc="lower left", framealpha=0.9)

    # -- Bottom panel: Action sequence heatmap -----------------------------
    ax2 = axes[1]
    action_gens = list(range(2, 2 + len(actions_taken)))
    action_colors = [colors[a] for a in actions_taken]
    ax2.bar(action_gens, [1] * len(actions_taken), color=action_colors, width=1.0, edgecolor="none")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Action")
    ax2.set_yticks([])
    ax2.set_title("Learned Action Sequence", fontsize=10)

    # Legend for action colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[a], label=f"{a}: {PHASE56_BASELINES[a]['name']}")
                       for a in range(6)]
    ax2.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=7, framealpha=0.9)

    plt.tight_layout()
    pdf_path = run_dir / "fig_04_baseline_comparison.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", pdf_path)

    # Also save PNG for quick viewing
    png_path = run_dir / "fig_04_baseline_comparison.png"
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    # Just re-save as PNG
    import shutil
    fig_copy = fig  # Already closed, regenerate
    # Simpler: just save from the same code
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close("all")

    return pdf_path


def generate_comparison_plot_v2(
    ppo_rows: list[dict],
    baselines: dict[int, list[dict]],
    actions_taken: list[int],
    run_dir: Path,
) -> Path:
    """Generate fig_04 — robust version that doesn't close prematurely."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        logger.warning("matplotlib not available — skipping plot generation")
        return run_dir / "fig_04_baseline_comparison.pdf"

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "figure.dpi": 300,
    })

    colors = {
        0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c",
        3: "#d62728", 4: "#9467bd", 5: "#8c564b",
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # -- Top: convergence curves ------------------------------------------
    ax = axes[0]
    for aid in range(6):
        name = PHASE56_BASELINES[aid]["name"]
        gens = [r["gen"] for r in baselines[aid]]
        hard = [r["best_hard"] for r in baselines[aid]]
        ax.plot(gens, hard, color=colors[aid], alpha=0.5, linewidth=1.0,
                linestyle="--", label=f"Static {aid} ({name})")

    ppo_gens = [r["gen"] for r in ppo_rows]
    ppo_hard = [r["best_hard"] for r in ppo_rows]
    ax.plot(ppo_gens, ppo_hard, color="black", linewidth=2.5,
            label="PPO Adaptive", zorder=10)

    ax.set_ylabel("Best Hard Penalty")
    ax.set_title("Phase 57: PPO vs Static Baselines — Hard Constraint Convergence")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- Bottom: action bar -----------------------------------------------
    ax2 = axes[1]
    action_gens = list(range(2, 2 + len(actions_taken)))
    action_colors_list = [colors[a] for a in actions_taken]
    ax2.bar(action_gens, [1] * len(actions_taken), color=action_colors_list,
            width=1.0, edgecolor="none")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Action")
    ax2.set_yticks([])
    ax2.set_title("Learned Action Sequence", fontsize=10)
    legend_elements = [Patch(facecolor=colors[a], label=f"{a}: {PHASE56_BASELINES[a]['name']}")
                       for a in range(6)]
    ax2.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=7, framealpha=0.9)

    plt.tight_layout()

    pdf_path = run_dir / "fig_04_baseline_comparison.pdf"
    png_path = run_dir / "fig_04_baseline_comparison.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s and .png", pdf_path)

    return pdf_path


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Full Phase 57 pipeline: train → baselines → evaluate → compare → plot."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_phase57" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    # -- 1. Train PPO -------------------------------------------------------
    model = train(run_dir)

    # -- 2. Run static baselines (same eval conditions) ----------------------
    baselines = run_static_baselines(run_dir)

    # -- 3. Evaluate trained PPO (deterministic) ----------------------------
    ppo_rows, actions_taken = evaluate_ppo(model, run_dir)

    # -- 4. Print comparison ------------------------------------------------
    print_comparison(ppo_rows, actions_taken, baselines, run_dir)

    # -- 5. Generate plot ---------------------------------------------------
    pdf = generate_comparison_plot_v2(ppo_rows, baselines, actions_taken, run_dir)

    # -- Summary ------------------------------------------------------------
    total_time = time.perf_counter() - t_total
    print(f"\n  Total Phase 57 time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Output: {run_dir}")
    print(f"  Model:  ppo_phase57.zip")
    print(f"  Plot:   {pdf.name}")
    print()

    logger.info("=" * 72)
    logger.info("Phase 57 complete in %.0fs", total_time)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
