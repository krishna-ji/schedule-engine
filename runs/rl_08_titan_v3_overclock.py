#!/usr/bin/env python3
r"""Phase 59 — Titan V3 Overclock: Properly Scaled PPO Training.

Forensic audit (verify_rl_brain.py) proved the Phase 57 PPO policy
collapsed into a near-uniform distribution because 2,500 timesteps was
severely under-trained for a 39-D observation space.

Titan V3 fixes:
  - 50,000 timesteps (20× Phase 57) — enough data for the value network
    to actually learn the 39-D state space
  - lr=5e-4 (aggressive learning) — extract more signal per sample
  - batch_size=128 — larger batches stabilise gradient estimates
  - ent_coef=0.05 (5× Phase 57) — prevent early policy collapse,
    force the agent to keep exploring all 6 LLHs

Episode config kept at pop=40, max_gen=25 (same as Phase 57) so each
episode is ~2-3 min.  50,000 steps / 24 steps/episode ≈ 2,083 episodes.

Wall-clock estimate: ~2,083 episodes × 2.5 min ≈ 87 hours (~3.6 days).

Usage::

    python runs/rl_08_titan_v3_overclock.py

Model saved to: output/models/ppo_titan_v3.zip
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
logger = logging.getLogger("rl_08_titan_v3")

# ======================================================================
# Configuration — Titan V3 Overclock
# ======================================================================

# -- Training -----------------------------------------------------------
TRAIN_POP_SIZE = 40        # Minimal pop → fastest episodes (~2-3 min)
TRAIN_MAX_GEN = 25         # 25 gens per episode (24 steps + init)
TOTAL_TIMESTEPS = 50_000   # 20× Phase 57 budget
LEARNING_RATE = 5e-4       # Aggressive learning rate
CLIP_RANGE = 0.2           # Standard PPO clip
NET_ARCH = [64, 64]        # 2-layer MLP (same architecture)
N_STEPS = 128              # Larger rollout buffer (multiple episodes)
BATCH_SIZE = 128            # Large batch for stable gradients
N_EPOCHS = 10              # PPO update epochs per rollout
GAE_LAMBDA = 0.95          # Standard GAE
GAMMA = 0.99               # Discount factor
ENT_COEF = 0.05            # 5× Phase 57 — prevent policy collapse

# -- Evaluation ---------------------------------------------------------
EVAL_POP_SIZE = 120         # Full pop for fair comparison
EVAL_MAX_GEN = 25           # Same horizon
EVAL_SEED = 42              # Deterministic comparison
PKL_PATH = ".cache/events_with_domains.pkl"

# -- Output -------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "output" / "models"
MODEL_PATH = MODEL_DIR / "ppo_titan_v3.zip"

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
# Training
# ======================================================================


def train(run_dir: Path) -> object:
    """Train PPO Titan V3 on the 6-LLH hyper-heuristic env."""
    from stable_baselines3 import PPO

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv
    from src.rl.training.thesis_callback import ThesisLoggingCallback

    print()
    print("=" * 80)
    print("  TITAN V3 OVERCLOCK — Phase 59 PPO Training")
    print("=" * 80)
    logger.info("  pop=%d  max_gen=%d  timesteps=%d  lr=%.1e",
                TRAIN_POP_SIZE, TRAIN_MAX_GEN, TOTAL_TIMESTEPS, LEARNING_RATE)
    logger.info("  n_steps=%d  batch=%d  epochs=%d  ent_coef=%.3f",
                N_STEPS, BATCH_SIZE, N_EPOCHS, ENT_COEF)
    logger.info("  model output: %s", MODEL_PATH)
    logger.info("  run_dir: %s", run_dir)
    print("=" * 80)

    # -- Environment -------------------------------------------------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=TRAIN_MAX_GEN,
        pop_size=TRAIN_POP_SIZE,
        algorithm_name="nsga2",
        seed=42,
    )

    # -- PPO Agent (Overclocked) -------------------------------------------
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

    # Print the network
    logger.info("Policy network:\n%s", model.policy.mlp_extractor)

    # -- Callback ----------------------------------------------------------
    callback = ThesisLoggingCallback(run_dir=run_dir, verbose=1)

    # -- Train! ------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("Starting training at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Estimated: ~2,083 episodes × ~2.5 min = ~87 hours")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    train_time = time.perf_counter() - t0
    logger.info("Training complete in %.1fs (%.1f min, %.1f hours)",
                train_time, train_time / 60, train_time / 3600)

    # -- Save model --------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info("Model saved: %s", MODEL_PATH)

    # Also save a copy in the run dir
    run_model_path = run_dir / "ppo_titan_v3.zip"
    model.save(str(run_model_path))
    logger.info("Model copy: %s", run_model_path)

    env.close()

    # -- Print summary -----------------------------------------------------
    print()
    print("=" * 80)
    print("  TITAN V3 TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Wall-clock:  {train_time:.0f}s ({train_time / 3600:.1f} hours)")
    print(f"  Timesteps:   {TOTAL_TIMESTEPS}")
    ep_count = callback._episode_count if hasattr(callback, '_episode_count') else '?'
    print(f"  Episodes:    {ep_count}")
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Run dir:     {run_dir}")
    print("=" * 80)

    return model


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Launch Titan V3 training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_titan_v3" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    model = train(run_dir)

    logger.info("Titan V3 training complete. Run eval_titan_v3_stochastic.py next.")


if __name__ == "__main__":
    main()
