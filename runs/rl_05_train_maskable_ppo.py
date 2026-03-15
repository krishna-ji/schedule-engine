#!/usr/bin/env python3
r"""RL 05 -- Maskable PPO: 24-Core Parallel Training with Action Masking.

State-Conditioned Action Masking using MaskablePPO from sb3-contrib
with 24-core SubprocVecEnv parallelism.

Key Features:
  - MaskablePPO automatically calls env.action_masks() at each step
  - Actions 3 & 7 masked when hard constraints are violated
  - 24 parallel subprocess environments for fast training

CRITICAL: Windows requires ``if __name__ == '__main__':`` guard.

Usage::

    python runs/rl_05_train_maskable_ppo.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_05_train_maskable_ppo")

# Suppress noisy subprocess logs
logging.getLogger("src.pipeline.pymoo_operators").setLevel(logging.WARNING)
logging.getLogger("src.pipeline").setLevel(logging.WARNING)
logging.getLogger("src.rl.gym_env").setLevel(logging.WARNING)

# ======================================================================
# Configuration
# ======================================================================

NUM_CPU = 24  # 24 of 32 cores (28 causes BrokenPipe on Windows)
SEED = 42
POP_SIZE = 40  # Reduced from 50 -> ~5-7s/step
MAX_GENERATIONS = 50  # Increased from 25 (more steps/episode = richer training)
TOTAL_TIMESTEPS = 30_000  # Reduced from 50k
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
NET_ARCH = [64, 64]
N_STEPS = 128  # Steps per env per rollout (128 x 24 = 3,072 buffer)
BATCH_SIZE = 512  # Mini-batch from the 3,072 buffer
N_EPOCHS = 10
GAE_LAMBDA = 0.95
GAMMA = 0.99
ENT_COEF = 0.01
PKL_PATH = ".cache/events_with_domains.pkl"


# ======================================================================
# Environment Factory (must be top-level for pickling on Windows)
# ======================================================================


def make_env(rank: int, seed: int = SEED):
    """Return a closure that creates a training env for subprocess *rank*."""

    def _init():
        from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

        return PymooHyperHeuristicEnv(
            pkl_path=PKL_PATH,
            max_generations=MAX_GENERATIONS,
            pop_size=POP_SIZE,
            algorithm_name="nsga2",
            seed=seed + rank,
            run_preflight=False,
        )

    return _init


# ======================================================================
# Main — MUST be inside __name__ guard on Windows
# ======================================================================

if __name__ == "__main__":
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from src.pipeline.build_events import ensure_pkl
    from src.rl.training.thesis_callback import ThesisLoggingCallback

    # -- Pre-flight: build cache BEFORE spawning subprocesses
    ensure_pkl(PKL_PATH)

    logger.info("Running feasibility preflight (one-time)...")
    from src.pipeline.scheduling_problem import create_problem as _preflight_check

    _preflight_check(PKL_PATH, run_preflight=True)
    logger.info("Preflight PASSED — workers will skip redundant checks.")

    # -- Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "maskable_ppo" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    est_eps_per_worker = TOTAL_TIMESTEPS // (NUM_CPU * (MAX_GENERATIONS - 1))
    est_rollouts = -(-TOTAL_TIMESTEPS // (NUM_CPU * N_STEPS))

    print()
    print("=" * 80)
    print("  RL 05 — Maskable PPO: 24-Core Parallel Training")
    print("=" * 80)
    logger.info("  Hardware:    %d cores, SubprocVecEnv(spawn)", NUM_CPU)
    logger.info("  Population:  %d, max_gen=%d", POP_SIZE, MAX_GENERATIONS)
    logger.info("  Timesteps:   %d", TOTAL_TIMESTEPS)
    logger.info(
        "  Rollout:     %d envs x %d steps = %d buffer",
        NUM_CPU,
        N_STEPS,
        NUM_CPU * N_STEPS,
    )
    logger.info("  Updates:     %d rollouts", est_rollouts)
    logger.info("  Run dir:     %s", run_dir)
    print("=" * 80)
    print()

    # -- Create 12 parallel environments
    logger.info("Spawning %d subprocess environments...", NUM_CPU)
    t_spawn = time.perf_counter()

    env = SubprocVecEnv(
        [make_env(i) for i in range(NUM_CPU)],
        start_method="spawn",
    )

    spawn_time = time.perf_counter() - t_spawn
    logger.info("All %d environments spawned in %.1fs", NUM_CPU, spawn_time)

    # Verify action masks work through the vec env pipe
    masks = np.stack(env.env_method("action_masks"))
    logger.info("Action masks shape: %s (all True: %s)", masks.shape, np.all(masks))

    # -- MaskablePPO Agent
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
        seed=SEED,
        verbose=1,
    )

    logger.info("MaskablePPO model initialized")
    total_params = sum(p.numel() for p in model.policy.parameters())
    logger.info("Total parameters: %d", total_params)

    # -- Callback
    callback = ThesisLoggingCallback(run_dir=run_dir, verbose=1)

    # -- Train
    t0 = time.perf_counter()
    logger.info("Training started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    train_time = time.perf_counter() - t0
    logger.info(
        "Training complete in %.1fs (%.1f min, %.1f hours)",
        train_time,
        train_time / 60,
        train_time / 3600,
    )

    # -- Save model
    model_dir = PROJECT_ROOT / "output" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_run = run_dir / "maskable_ppo_final.zip"
    model.save(str(model_run))
    logger.info("Model saved (run): %s", model_run)

    canonical_path = model_dir / "maskable_ppo_final.zip"
    model.save(str(canonical_path))
    logger.info("Model saved (canonical): %s", canonical_path)

    # -- Cleanup
    env.close()

    # -- Summary
    print()
    print("=" * 80)
    print("  MASKABLE PPO TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Wall-clock:  {train_time:.0f}s ({train_time / 3600:.1f} hours)")
    print(f"  Timesteps:   {TOTAL_TIMESTEPS}")
    print(f"  Workers:     {NUM_CPU}")
    print(f"  Model:       {canonical_path}")
    print(f"  Run dir:     {run_dir}")
    print("=" * 80)

    # -- Generate thesis plots
    from src.rl.training.plot_thesis_figures import generate_plots

    generate_plots(run_dir)
