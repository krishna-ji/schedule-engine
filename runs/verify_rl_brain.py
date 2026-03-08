#!/usr/bin/env python3
r"""Phase 57 Forensic Verification — PPO Brain Autopsy.

Loads the trained PPO model and runs a deterministic evaluation while
logging the **exact action probabilities** (softmax of logits) at every
generation.  This reveals whether the agent learned a state-dependent
policy or collapsed to a single static action.

Key question: does the probability distribution shift across generations,
or is it a flat "always pick action X" regardless of state?

Usage::

    python runs/verify_rl_brain.py
"""

from __future__ import annotations

import logging
import sys
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
logger = logging.getLogger("verify_rl_brain")

# ======================================================================
# Configuration
# ======================================================================
MODEL_PATH = PROJECT_ROOT / "output" / "rl_phase57" / "20260307_161124" / "ppo_phase57.zip"
PKL_PATH = ".cache/events_with_domains.pkl"
EVAL_POP_SIZE = 120
EVAL_MAX_GEN = 25
EVAL_SEED = 42

ACTION_NAMES = {
    0: "Conservative",
    1: "Aggressive",
    2: "Memetic",
    3: "SoftFocus",
    4: "Destructive",
    5: "Intensified",
}
ACTION_SHORT = {
    0: "Con",
    1: "Agg",
    2: "Mem",
    3: "Sft",
    4: "Des",
    5: "Int",
}


def main() -> None:
    from stable_baselines3 import PPO

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model from {MODEL_PATH}")
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    model = PPO.load(str(MODEL_PATH))
    logger.info("Model loaded successfully")

    # Print network architecture
    logger.info(f"Policy network: {model.policy.mlp_extractor}")

    # ------------------------------------------------------------------
    # 2. Create evaluation environment
    # ------------------------------------------------------------------
    logger.info(
        f"Creating eval env: pop={EVAL_POP_SIZE}, gens={EVAL_MAX_GEN}, seed={EVAL_SEED}"
    )
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_MAX_GEN,
        pop_size=EVAL_POP_SIZE,
        seed=EVAL_SEED,
    )

    obs, info = env.reset()
    init_hard = info.get("best_hard", "?")
    init_soft = info.get("best_soft", "?")
    logger.info(f"Reset complete. Initial hard={init_hard}, soft={init_soft}")

    # ------------------------------------------------------------------
    # 3. Header
    # ------------------------------------------------------------------
    print()
    print("=" * 120)
    print("  PHASE 57 FORENSIC VERIFICATION — PPO Brain Autopsy")
    print("=" * 120)
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  Eval:  pop={EVAL_POP_SIZE}, gens={EVAL_MAX_GEN}, seed={EVAL_SEED}")
    print(f"  Init:  hard={init_hard}, soft={init_soft}")
    print("=" * 120)
    print()

    # Column headers
    header = (
        f"{'Step':>6} │ {'Hard':>5} {'Soft':>5} │ {'Action':>14} │"
        f"  {'p(Con)':>7} {'p(Agg)':>7} {'p(Mem)':>7} {'p(Sft)':>7} {'p(Des)':>7} {'p(Int)':>7}"
        f" │ {'Reward':>8} │ {'Entropy':>8}"
    )
    separator = "─" * len(header)
    print(header)
    print(separator)

    # ------------------------------------------------------------------
    # 4. Evaluation loop with probability extraction
    # ------------------------------------------------------------------
    all_probs = []
    actions_taken = []
    rewards = []
    hard_trajectory = []
    soft_trajectory = []
    cumulative_reward = 0.0

    for step in range(1, EVAL_MAX_GEN + 1):
        # --- Extract action probabilities from the policy network ---
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
            # Also get log probs for entropy calculation
            log_probs = distribution.distribution.logits.detach().cpu().numpy()[0]

        # Entropy of the distribution: -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # --- Get deterministic action ---
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        best_hard = info.get("best_hard", "?")
        best_soft = info.get("best_soft", "?")
        cumulative_reward += reward

        # Store
        all_probs.append(probs.copy())
        actions_taken.append(action)
        rewards.append(reward)
        hard_trajectory.append(best_hard)
        soft_trajectory.append(best_soft)

        # --- Print formatted row ---
        prob_str = " ".join(f"{p:7.4f}" for p in probs)
        chosen_name = ACTION_NAMES.get(action, f"Unk({action})")

        # Highlight: bold the chosen action's probability
        row = (
            f"{step:>6} │ {best_hard:>5} {best_soft:>5} │ {chosen_name:>14} │"
            f"  {prob_str}"
            f" │ {reward:>8.3f} │ {entropy:>8.4f}"
        )
        print(row)

        if terminated or truncated:
            print(f"\n  [Episode ended at step {step}]")
            break

    print(separator)

    # ------------------------------------------------------------------
    # 5. Summary Statistics
    # ------------------------------------------------------------------
    print()
    print("=" * 120)
    print("  AUTOPSY SUMMARY")
    print("=" * 120)

    # --- Action frequency ---
    from collections import Counter

    action_counts = Counter(actions_taken)
    print("\n  ACTION FREQUENCY (evaluation):")
    for a in range(6):
        count = action_counts.get(a, 0)
        pct = 100.0 * count / len(actions_taken)
        bar = "█" * int(pct / 2)
        print(f"    {a} ({ACTION_NAMES[a]:>14}): {count:>3} ({pct:5.1f}%) {bar}")

    # --- Probability statistics ---
    prob_matrix = np.array(all_probs)  # shape (T, 6)
    print("\n  PROBABILITY STATISTICS (mean ± std across all steps):")
    for a in range(6):
        mean_p = prob_matrix[:, a].mean()
        std_p = prob_matrix[:, a].std()
        min_p = prob_matrix[:, a].min()
        max_p = prob_matrix[:, a].max()
        print(
            f"    {a} ({ACTION_NAMES[a]:>14}): "
            f"mean={mean_p:.4f} ± {std_p:.4f}  "
            f"[min={min_p:.4f}, max={max_p:.4f}]"
        )

    # --- Phase analysis ---
    n_steps = len(actions_taken)
    third = max(1, n_steps // 3)
    phases = {
        "Early (gen 1-8)": slice(0, min(8, n_steps)),
        "Mid   (gen 9-17)": slice(min(8, n_steps), min(17, n_steps)),
        "Late  (gen 18-25)": slice(min(17, n_steps), n_steps),
    }
    print("\n  PHASE-WISE MEAN PROBABILITIES:")
    phase_header = f"    {'Phase':<20} │ " + " ".join(f"{'p(' + ACTION_SHORT[a] + ')':>7}" for a in range(6))
    print(phase_header)
    print("    " + "─" * (len(phase_header) - 4))
    for phase_name, phase_slice in phases.items():
        phase_probs = prob_matrix[phase_slice]
        if len(phase_probs) == 0:
            continue
        means = phase_probs.mean(axis=0)
        row = f"    {phase_name:<20} │ " + " ".join(f"{m:>7.4f}" for m in means)
        print(row)

    # --- Entropy analysis ---
    entropies = [-np.sum(p * np.log(p + 1e-10)) for p in all_probs]
    max_entropy = np.log(6)  # uniform distribution
    print(f"\n  ENTROPY ANALYSIS:")
    print(f"    Mean entropy:    {np.mean(entropies):.4f} / {max_entropy:.4f} (max = uniform)")
    print(f"    Min entropy:     {np.min(entropies):.4f} (step {np.argmin(entropies) + 1})")
    print(f"    Max entropy:     {np.max(entropies):.4f} (step {np.argmax(entropies) + 1})")
    print(f"    Entropy trend:   first={entropies[0]:.4f} → last={entropies[-1]:.4f}")

    # --- State dependency check ---
    # If probabilities change significantly across steps, the policy IS state-dependent
    prob_variance = prob_matrix.var(axis=0).sum()  # total variance across all actions
    print(f"\n  STATE-DEPENDENCY METRIC:")
    print(f"    Total probability variance: {prob_variance:.6f}")
    if prob_variance > 0.01:
        print(f"    >>> VERDICT: Policy IS state-dependent (variance > 0.01)")
    elif prob_variance > 0.001:
        print(f"    >>> VERDICT: Policy shows WEAK state-dependency")
    else:
        print(f"    >>> VERDICT: Policy is STATIC — probabilities don't change with state")

    # --- Final scores ---
    print(f"\n  FINAL SCORES:")
    final_hard = hard_trajectory[-1] if hard_trajectory else "?"
    final_soft = soft_trajectory[-1] if soft_trajectory else "?"
    best_hard_gen = min(range(len(hard_trajectory)), key=lambda i: hard_trajectory[i]) if hard_trajectory else 0
    best_hard_val = min(hard_trajectory) if hard_trajectory else "?"
    best_soft_at_best_hard = soft_trajectory[best_hard_gen] if soft_trajectory else "?"
    print(f"    Final:         hard={final_hard}, soft={final_soft}")
    print(f"    Best hard:     {best_hard_val} at gen {best_hard_gen + 1} (soft={best_soft_at_best_hard})")
    print(f"    Cumulative R:  {cumulative_reward:.4f}")

    # --- Action sequence as string ---
    seq = " → ".join(ACTION_SHORT[a] for a in actions_taken)
    print(f"\n  ACTION SEQUENCE:")
    print(f"    {seq}")

    print()
    print("=" * 120)
    print("  Forensic verification complete.")
    print("=" * 120)

    env.close()


if __name__ == "__main__":
    main()
