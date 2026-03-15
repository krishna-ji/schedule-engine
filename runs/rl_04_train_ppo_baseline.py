#!/usr/bin/env python3
r"""RL 04 -- PPO Baseline: Parallel Training + Evaluation.

12-core SubprocVecEnv PPO training with deterministic evaluation.

1. Train PPO for 50,000 timesteps across 12 parallel envs.
2. Run 50-gen strict evaluation (single env, deterministic).
3. Export evaluation CSV and thesis plots.

CRITICAL: Windows requires ``if __name__ == '__main__':`` guard.

Usage::

    python runs/rl_04_train_ppo_baseline.py
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
logger = logging.getLogger("rl_04_train_ppo_baseline")

# Suppress noisy subprocess logs
logging.getLogger("src.pipeline.pymoo_operators").setLevel(logging.WARNING)
logging.getLogger("src.pipeline").setLevel(logging.WARNING)
logging.getLogger("src.rl.gym_env").setLevel(logging.WARNING)

# ==================== CONFIGURATION ====================
NUM_CPU = 12  # 12 of 32 cores
TOTAL_TIMESTEPS = 50_000  # Reduced from 150k
TRAINING_GENERATIONS = 50
TRAINING_POP_SIZE = 40  # Reduced from 120 -> ~5-7s/step
N_STEPS = 128  # Steps per env per rollout (128 x 12 = 1,536 buffer)
BATCH_SIZE = 256  # Mini-batch from the 1,536 buffer
N_EPOCHS = 10

EVAL_GENERATIONS = 50  # Reduced from 200
EVAL_POP_SIZE = 40  # Reduced from 120

LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.95
GAMMA = 0.99
ENT_COEF = 0.01
NET_ARCH = [64, 64]
SEED = 42
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
            max_generations=TRAINING_GENERATIONS,
            pop_size=TRAINING_POP_SIZE,
            algorithm_name="nsga2",
            seed=seed + rank,
            run_preflight=False,
        )

    return _init


# ======================================================================
# Main — MUST be inside __name__ guard on Windows
# ======================================================================

if __name__ == "__main__":
    import pandas as pd
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from src.pipeline.build_events import ensure_pkl
    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv
    from src.rl.training.plot_thesis_figures import generate_plots
    from src.rl.training.thesis_callback import ThesisLoggingCallback

    # -- Pre-flight: build cache BEFORE spawning subprocesses
    ensure_pkl(PKL_PATH)

    logger.info("Running feasibility preflight (one-time)...")
    from src.pipeline.scheduling_problem import create_problem as _preflight_check

    _preflight_check(PKL_PATH, run_preflight=True)
    logger.info("Preflight PASSED — workers will skip redundant checks.")

    # -- Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_ppo_baseline" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    est_eps_per_worker = TOTAL_TIMESTEPS // (NUM_CPU * (TRAINING_GENERATIONS - 1))
    est_rollouts = -(-TOTAL_TIMESTEPS // (NUM_CPU * N_STEPS))

    print()
    print("=" * 80)
    print("  RL 04 — PPO Baseline: 12-Core Parallel Training")
    print("=" * 80)
    logger.info("  Hardware:    %d cores, SubprocVecEnv(spawn)", NUM_CPU)
    logger.info(
        "  Population:  %d, max_gen=%d", TRAINING_POP_SIZE, TRAINING_GENERATIONS
    )
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

    vec_env = SubprocVecEnv(
        [make_env(i) for i in range(NUM_CPU)],
        start_method="spawn",
    )

    spawn_time = time.perf_counter() - t_spawn
    logger.info("All %d environments spawned in %.1fs", NUM_CPU, spawn_time)

    # -- PPO Agent
    model = PPO(
        "MlpPolicy",
        vec_env,
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

    model_run = run_dir / "ppo_baseline_final.zip"
    model.save(str(model_run))
    logger.info("Model saved (run): %s", model_run)

    canonical_path = model_dir / "ppo_capstone_final.zip"
    model.save(str(canonical_path))
    logger.info("Model saved (canonical): %s", canonical_path)

    vec_env.close()

    # -- Evaluation (single env, deterministic)
    logger.info("-" * 60)
    logger.info(
        "Deterministic evaluation (%d gens, pop=%d)", EVAL_GENERATIONS, EVAL_POP_SIZE
    )
    logger.info("-" * 60)

    eval_env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_GENERATIONS,
        pop_size=EVAL_POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED + 1000,
    )

    obs, info = eval_env.reset(seed=SEED + 1000)
    generation_data = []

    t_eval = time.perf_counter()
    for step in range(EVAL_GENERATIONS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        best_hard = info.get("best_hard", 999999)
        best_soft = info.get("best_soft", 999999)
        current_gen = info.get("generation", step + 1)

        if (step + 1) % 10 == 0:
            logger.info(
                "  Gen %3d/%d | Hard=%.2f | Soft=%.2f | R=%.4f",
                current_gen,
                EVAL_GENERATIONS,
                best_hard,
                best_soft,
                reward,
            )

        generation_data.append(
            {
                "generation": current_gen,
                "best_hard": best_hard,
                "best_soft": best_soft,
                "reward": reward,
                "action": int(action),
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        if terminated or truncated:
            break

    eval_time = time.perf_counter() - t_eval
    logger.info("Evaluation complete in %.1fs", eval_time)

    eval_env.close()

    # -- Save evaluation CSV
    df = pd.DataFrame(generation_data)
    csv_path = run_dir / "evaluation_trajectory.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved evaluation trajectory: %s", csv_path)

    if generation_data:
        final_gen = generation_data[-1]
        print()
        print("=" * 80)
        print("  PPO BASELINE COMPLETE")
        print("=" * 80)
        print(f"  Training:   {train_time:.0f}s ({train_time / 3600:.1f} hours)")
        print(f"  Timesteps:  {TOTAL_TIMESTEPS}")
        print(f"  Workers:    {NUM_CPU}")
        print(f"  Final Hard: {final_gen['best_hard']:.2f}")
        print(f"  Final Soft: {final_gen['best_soft']:.2f}")
        print(f"  Model:      {canonical_path}")
        print(f"  Run dir:    {run_dir}")
        print("=" * 80)

    # -- Generate thesis plots
    generate_plots(run_dir)
