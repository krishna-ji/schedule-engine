#!/usr/bin/env python3
r"""Phase 61 — Titan V3 Parallel: 24-Core Overclocked PPO Training.

Exploits the 32-core / 128GB machine by running 24 parallel Pymoo
environments via SubprocVecEnv.  Each subprocess independently loads
the scheduling problem and runs its own GA population, feeding
experience to a single MaskablePPO agent.

Math:
  - 24 envs × 128 n_steps = 3,072 steps per rollout
  - 50,000 total_timesteps / 3,072 = ~17 rollout updates
  - Each rollout: 128 steps × ~5s/step = ~640s wall-clock (parallel)
  - Total: 17 × 640s ≈ 3.0 hours

CRITICAL: Windows requires ``if __name__ == '__main__':`` guard to
prevent recursive subprocess spawning (spawn-based multiprocessing).

Usage::

    python runs/rl_08_titan_v3_parallel.py

Model saved to: output/models/ppo_titan_v3_parallel.zip
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_08_titan_v3_parallel")

# ======================================================================
# Configuration
# ======================================================================
NUM_CPU = 24  # 24 of 32 cores (leave 8 for OS + Titan V3 single)
TRAIN_POP_SIZE = 40  # Small pop → fast steps (~5s each)
TRAIN_MAX_GEN = 50  # 50 gens per episode (49 steps + init)
TOTAL_TIMESTEPS = 50_000  # Same budget as single-core Titan V3
LEARNING_RATE = 5e-4  # Aggressive learning
CLIP_RANGE = 0.2  # Standard PPO clip
NET_ARCH = [64, 64]  # 2-layer MLP
N_STEPS = 128  # Steps per env per rollout (128 × 24 = 3,072 buffer)
BATCH_SIZE = 512  # Mini-batch from the 3,072 buffer
N_EPOCHS = 10  # PPO update epochs per rollout
GAE_LAMBDA = 0.95
GAMMA = 0.99
ENT_COEF = 0.05  # 5× default — prevent policy collapse

PKL_PATH = ".cache/events_with_domains.pkl"
MODEL_DIR = PROJECT_ROOT / "output" / "models"
MODEL_PATH = MODEL_DIR / "ppo_titan_v3_parallel.zip"


# ======================================================================
# Environment Factory (must be top-level for pickling on Windows)
# ======================================================================


def make_env(rank: int, seed: int = 42):
    """Return a closure that creates a PymooHyperHeuristicEnv.

    Each subprocess gets a unique seed (seed + rank) so populations
    are diverse across workers.  The closure pattern is required by
    SubprocVecEnv.
    """

    def _init():
        from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

        env = PymooHyperHeuristicEnv(
            pkl_path=PKL_PATH,
            max_generations=TRAIN_MAX_GEN,
            pop_size=TRAIN_POP_SIZE,
            algorithm_name="nsga2",
            seed=seed + rank,
        )
        return env

    return _init


# ======================================================================
# Main — MUST be inside __name__ guard on Windows
# ======================================================================

if __name__ == "__main__":
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from src.rl.training.thesis_callback import ThesisLoggingCallback

    # -- Output directory --------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_titan_v3_parallel" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("  TITAN V3 PARALLEL — Phase 61: 24-Core Overclocked Training")
    print("=" * 80)
    logger.info(
        "  num_cpu=%d  pop=%d  max_gen=%d  timesteps=%d",
        NUM_CPU,
        TRAIN_POP_SIZE,
        TRAIN_MAX_GEN,
        TOTAL_TIMESTEPS,
    )
    logger.info(
        "  n_steps=%d  batch=%d  epochs=%d  ent_coef=%.3f  lr=%.1e",
        N_STEPS,
        BATCH_SIZE,
        N_EPOCHS,
        ENT_COEF,
        LEARNING_RATE,
    )
    logger.info(
        "  Rollout buffer: %d envs × %d steps = %d", NUM_CPU, N_STEPS, NUM_CPU * N_STEPS
    )
    logger.info(
        "  Rollout updates: ceil(%d / %d) = %d",
        TOTAL_TIMESTEPS,
        NUM_CPU * N_STEPS,
        -(-TOTAL_TIMESTEPS // (NUM_CPU * N_STEPS)),
    )
    logger.info("  Model output: %s", MODEL_PATH)
    logger.info("  Run dir: %s", run_dir)
    print("=" * 80)
    print()

    # -- Create 24 parallel environments -----------------------------------
    logger.info("Spawning %d subprocess environments (expect 30-60s delay)...", NUM_CPU)
    t_spawn = time.perf_counter()

    env = SubprocVecEnv(
        [make_env(i) for i in range(NUM_CPU)],
        start_method="spawn",  # Explicit spawn for Windows safety
    )

    spawn_time = time.perf_counter() - t_spawn
    logger.info("All %d environments spawned in %.1fs", NUM_CPU, spawn_time)

    # Verify action masks work through the vec env pipe
    masks = np.stack(env.env_method("action_masks"))
    logger.info("Action masks shape: %s (all True: %s)", masks.shape, np.all(masks))

    # -- MaskablePPO Agent -------------------------------------------------
    logger.info("Creating MaskablePPO agent...")

    model = MaskablePPO(
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

    logger.info("Policy network:\n%s", model.policy.mlp_extractor)
    total_params = sum(p.numel() for p in model.policy.parameters())
    logger.info("Total parameters: %d", total_params)

    # -- Callback ----------------------------------------------------------
    callback = ThesisLoggingCallback(run_dir=run_dir, verbose=1)

    # -- Train! ------------------------------------------------------------
    t0 = time.perf_counter()
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Training started at %s", start_str)
    logger.info("ETA: ~3 hours (17 rollout updates × ~640s each)")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    train_time = time.perf_counter() - t0
    logger.info(
        "Training complete in %.1fs (%.1f min, %.1f hours)",
        train_time,
        train_time / 60,
        train_time / 3600,
    )

    # -- Save model --------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info("Model saved: %s", MODEL_PATH)

    # Also save in run dir
    run_model = run_dir / "ppo_titan_v3_parallel.zip"
    model.save(str(run_model))
    logger.info("Model copy: %s", run_model)

    # -- Cleanup -----------------------------------------------------------
    env.close()

    # -- Summary -----------------------------------------------------------
    print()
    print("=" * 80)
    print("  TITAN V3 PARALLEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Wall-clock:  {train_time:.0f}s ({train_time / 3600:.1f} hours)")
    print(f"  Timesteps:   {TOTAL_TIMESTEPS}")
    ep_count = callback._episode_count if hasattr(callback, "_episode_count") else "?"
    print(f"  Episodes:    {ep_count} (logged from env 0 only)")
    print(f"  Workers:     {NUM_CPU}")
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Run dir:     {run_dir}")
    print()
    print("  Next: python runs/eval_titan_v3_stochastic.py")
    print("         (update MODEL_PATH to ppo_titan_v3_parallel.zip)")
    print("=" * 80)
