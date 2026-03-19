#!/usr/bin/env python3
r"""Phase 62 — Titan V4 SOTA: Algorithmic Overhaul with PBRS + Curriculum.

Builds on the Titan V3 parallel infrastructure (24-core SubprocVecEnv)
and adds two algorithmic upgrades:

1. **Potential-Based Reward Shaping (PBRS)**: Dense gradient signal via
   Bottleneck Density potential function.  Provides reward even when
   Δhard ≈ 0 in the convergence plateau.

2. **Constraint Curriculum**: Three-phase schedule that gradually
   increases reward complexity — spatial constraints first, then
   instructor, then full NP-hard.  Smooth blending at transitions.

Math — Reward Architecture
--------------------------

.. math::

    R_{\text{shaped}} = R_{\text{base}}
        + \underbrace{\gamma \Phi(s') - \Phi(s)}_{\text{PBRS}}
        + \underbrace{\sum_{c \in \text{active}} w_c \cdot \Delta_{cv_c}}_{\text{Curriculum}}

where :math:`\Phi(s) = -\text{Var}(\text{per-resource conflicts}) / \text{max\_var}`.

Curriculum Schedule (per-worker episodes):
  Phase 1 (0–21):   Spatial only    (SRE, FFC)
  Phase 2 (21–63):  + Instructor    (FTE, FPC, FCA)
  Phase 3 (63+):    Full complexity (+ CTE, CQF, ICTD)

Config: 24 cores, 100k timesteps, MaskablePPO, batch=512.

CRITICAL: Windows requires ``if __name__ == '__main__':`` guard.

Usage::

    python runs/rl_06_train_ppo_titan_v4_sota.py

Model saved to: output/models/ppo_titan_v4_sota.zip
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
logger = logging.getLogger("rl_06_train_ppo_titan_v4_sota")

# Suppress noisy per-reset/per-step logs from worker subprocesses.
# NOTE: This is cosmetic — SubprocVecEnv workers run in separate
# processes with independent GILs.  Logging is NOT the bottleneck
# (24 lines per episode boundary vs 13 min of GA computation).
logging.getLogger("src.pipeline.pymoo_operators").setLevel(logging.WARNING)
logging.getLogger("src.pipeline").setLevel(logging.WARNING)
logging.getLogger("src.rl.gym_env").setLevel(logging.WARNING)

# ======================================================================
# Configuration
# ======================================================================
NUM_CPU = 24  # 24 of 32 cores
TRAIN_POP_SIZE = 40  # Small pop → fast steps (~5s each)
TRAIN_MAX_GEN = 50  # 50 gens per episode (49 steps + init)
TOTAL_TIMESTEPS = 100_000  # 2× Titan V3 — curriculum speeds early convergence
LEARNING_RATE = 5e-4  # Aggressive learning
CLIP_RANGE = 0.2  # Standard PPO clip
NET_ARCH = [64, 64]  # 2-layer MLP
N_STEPS = 128  # Steps per env per rollout (128 × 24 = 3,072 buffer)
BATCH_SIZE = 512  # Mini-batch from the 3,072 buffer
N_EPOCHS = 10  # PPO update epochs per rollout
GAE_LAMBDA = 0.95
GAMMA = 0.99
ENT_COEF = 0.05  # 5× default → prevent policy collapse

# Curriculum thresholds (per-worker episodes)
# With 100k timesteps / 24 workers / 49 steps per episode ≈ 86 eps/worker
# Phase splits: ~25% / ~49% / ~26% of training
PHASE1_EPISODES = 21  # Phase 1: Spatial only (episodes 0–21)
PHASE2_EPISODES = 63  # Phase 2: + Instructor (episodes 21–63)
CURRICULUM_WEIGHT = 0.5  # Scaling factor for curriculum bonus

# PBRS
PBRS_GAMMA = 0.99
USE_CHROMOSOME_POTENTIAL = True  # Tier 2 per-resource variance

PKL_PATH = ".cache/events_with_domains.pkl"
MODEL_DIR = PROJECT_ROOT / "output" / "models"
MODEL_PATH = MODEL_DIR / "ppo_titan_v4_sota.zip"

MAX_CRASH_RETRIES = 10  # Max auto-restarts on BrokenPipeError
CHECKPOINT_INTERVAL = 5  # Save every 5 episodes (more frequent = less lost)


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the most recent checkpoint .zip in run_dir/checkpoints/, or None."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.zip"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


# ======================================================================
# Environment Factory (must be top-level for pickling on Windows)
# ======================================================================


def make_env(rank: int, seed: int = 42):
    """Return a closure that creates a curriculum-wrapped env.

    Each subprocess gets:
    - Unique seed (seed + rank) for diverse populations
    - ConstraintCurriculumWrapper with PBRS + curriculum shaping
    """

    def _init():
        from src.rl.gym_env.curriculum_wrapper import ConstraintCurriculumWrapper
        from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

        base_env = PymooHyperHeuristicEnv(
            pkl_path=PKL_PATH,
            max_generations=TRAIN_MAX_GEN,
            pop_size=TRAIN_POP_SIZE,
            algorithm_name="nsga2",
            seed=seed + rank,
            run_preflight=False,  # Skip redundant feasibility checks in workers
        )

        wrapped_env = ConstraintCurriculumWrapper(
            base_env,
            phase1_episodes=PHASE1_EPISODES,
            phase2_episodes=PHASE2_EPISODES,
            gamma=PBRS_GAMMA,
            curriculum_weight=CURRICULUM_WEIGHT,
            use_chromosome_potential=USE_CHROMOSOME_POTENTIAL,
        )

        return wrapped_env

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

    # -- Pre-flight: build cache BEFORE spawning subprocesses --------------
    ensure_pkl(PKL_PATH)

    # -- One-time feasibility check (in main process only) -----------------
    logger.info("Running feasibility preflight (one-time, main process only)...")
    from src.pipeline.scheduling_problem import create_problem as _preflight_check

    _preflight_check(PKL_PATH, run_preflight=True)
    logger.info("Preflight PASSED — workers will skip redundant checks.")

    # -- Output directory --------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "rl_titan_v4_sota" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    est_eps_per_worker = TOTAL_TIMESTEPS // (NUM_CPU * (TRAIN_MAX_GEN - 1))
    est_rollouts = -(-TOTAL_TIMESTEPS // (NUM_CPU * N_STEPS))

    print()
    print("=" * 80)
    print("  TITAN V4 SOTA — Phase 62: PBRS + Constraint Curriculum")
    print("=" * 80)
    logger.info("  Hardware:    %d cores, SubprocVecEnv(spawn)", NUM_CPU)
    logger.info("  Population:  %d, max_gen=%d", TRAIN_POP_SIZE, TRAIN_MAX_GEN)
    logger.info("  Timesteps:   %d (2× Titan V3)", TOTAL_TIMESTEPS)
    logger.info(
        "  Rollout:     %d envs × %d steps = %d buffer",
        NUM_CPU,
        N_STEPS,
        NUM_CPU * N_STEPS,
    )
    logger.info("  Updates:     %d rollouts", est_rollouts)
    logger.info(
        "  PPO:         lr=%.1e, clip=%.2f, ent=%.3f, epochs=%d, batch=%d",
        LEARNING_RATE,
        CLIP_RANGE,
        ENT_COEF,
        N_EPOCHS,
        BATCH_SIZE,
    )
    logger.info("")
    logger.info("  ── Algorithmic Upgrades ──")
    logger.info(
        "  PBRS:        γ=%.2f, Tier2=%s (per-resource variance)",
        PBRS_GAMMA,
        USE_CHROMOSOME_POTENTIAL,
    )
    logger.info(
        "  Curriculum:  Phase1=[0,%d] Spatial | Phase2=[%d,%d] +Inst | Phase3=[%d+] Full",
        PHASE1_EPISODES,
        PHASE1_EPISODES,
        PHASE2_EPISODES,
        PHASE2_EPISODES,
    )
    logger.info(
        "  Curriculum:  weight=%.2f, blend_window=5 eps",
        CURRICULUM_WEIGHT,
    )
    logger.info(
        "  Est. eps/worker: ~%d → Phase splits: ~25%%/49%%/26%%",
        est_eps_per_worker,
    )
    logger.info("")
    logger.info("  Model:       %s", MODEL_PATH)
    logger.info("  Run dir:     %s", run_dir)
    logger.info(
        "  Auto-restart: up to %d retries on BrokenPipeError", MAX_CRASH_RETRIES
    )
    print("=" * 80)
    print()

    # ==================================================================
    # Auto-restart training loop
    # ==================================================================
    t0 = time.perf_counter()
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Training started at %s", start_str)

    crash_count = 0
    training_complete = False

    while not training_complete and crash_count <= MAX_CRASH_RETRIES:
        env = None
        try:
            # -- Spawn environments ------------------------------------
            logger.info(
                "Spawning %d subprocess environments (attempt %d)...",
                NUM_CPU,
                crash_count + 1,
            )
            t_spawn = time.perf_counter()
            env = SubprocVecEnv(
                [make_env(i) for i in range(NUM_CPU)],
                start_method="spawn",
            )
            spawn_time = time.perf_counter() - t_spawn
            logger.info("All %d environments spawned in %.1fs", NUM_CPU, spawn_time)

            # Verify action masks
            masks = np.stack(env.env_method("action_masks"))
            logger.info(
                "Action masks shape: %s (all True: %s)", masks.shape, np.all(masks)
            )

            # -- Create or load model ----------------------------------
            resume_ckpt = _find_latest_checkpoint(run_dir)
            if resume_ckpt is not None:
                logger.info("RESUME: Loading checkpoint %s", resume_ckpt)
                model = MaskablePPO.load(str(resume_ckpt), env=env, verbose=1)
                reset_ts = False
                logger.info(
                    "Model loaded (ts=%d) — continuing training",
                    model.num_timesteps,
                )
            else:
                logger.info("Creating fresh MaskablePPO agent...")
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
                    policy_kwargs={"net_arch": NET_ARCH},
                    seed=42,
                    verbose=1,
                )
                reset_ts = True

            logger.info("Policy network:\n%s", model.policy.mlp_extractor)
            total_params = sum(p.numel() for p in model.policy.parameters())
            logger.info("Total parameters: %d", total_params)

            # -- Callback ----------------------------------------------
            callback = ThesisLoggingCallback(
                run_dir=run_dir,
                verbose=1,
                checkpoint_interval=CHECKPOINT_INTERVAL,
            )

            # -- Train! ------------------------------------------------
            logger.info(
                "ETA: ~%d hours (%d rollouts × ~36 min each)",
                est_rollouts * 36 // 60,
                est_rollouts,
            )
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callback,
                reset_num_timesteps=reset_ts,
            )

            training_complete = True
            logger.info("Training completed successfully!")

        except (BrokenPipeError, EOFError, ConnectionResetError) as exc:
            crash_count += 1
            logger.error("CRASH #%d: %s — %s", crash_count, type(exc).__name__, exc)

            # Save emergency checkpoint if model exists
            try:
                emergency_path = run_dir / "checkpoints" / "emergency_crash"
                (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                model.save(str(emergency_path))
                logger.info("Emergency checkpoint saved: %s", emergency_path)
            except Exception:
                logger.warning("Could not save emergency checkpoint")

            if crash_count > MAX_CRASH_RETRIES:
                logger.error("Max retries (%d) exceeded. Giving up.", MAX_CRASH_RETRIES)
                break

            logger.info(
                "Auto-restarting in 5 seconds... (attempt %d/%d)",
                crash_count + 1,
                MAX_CRASH_RETRIES + 1,
            )
            time.sleep(5)

        finally:
            # Always try to close the env to release subprocess resources
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    train_time = time.perf_counter() - t0
    logger.info(
        "Total wall-clock: %.1fs (%.1f min, %.1f hours), crashes: %d",
        train_time,
        train_time / 60,
        train_time / 3600,
        crash_count,
    )

    # -- Save final model --------------------------------------------------
    if training_complete:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_PATH))
        logger.info("Model saved: %s", MODEL_PATH)

        run_model = run_dir / "ppo_titan_v4_sota.zip"
        model.save(str(run_model))
        logger.info("Model copy: %s", run_model)

    # -- Summary -----------------------------------------------------------
    print()
    print("=" * 80)
    status = (
        "COMPLETE" if training_complete else f"FAILED (after {crash_count} crashes)"
    )
    print(f"  TITAN V4 SOTA TRAINING {status}")
    print("=" * 80)
    print(f"  Wall-clock:      {train_time:.0f}s ({train_time / 3600:.1f} hours)")
    print(f"  Timesteps:       {TOTAL_TIMESTEPS}")
    print(f"  Workers:         {NUM_CPU}")
    print(f"  Crashes/Restarts: {crash_count}")
    print(f"  PBRS:            γ={PBRS_GAMMA}, Tier2={USE_CHROMOSOME_POTENTIAL}")
    print(f"  Curriculum:      {PHASE1_EPISODES}/{PHASE2_EPISODES} episode thresholds")
    print(f"  Model:           {MODEL_PATH}")
    print(f"  Run dir:         {run_dir}")
    print()
    if training_complete:
        print("  Next: python runs/eval_titan_v3_stochastic.py")
        print("         (update MODEL_PATH to ppo_titan_v4_sota.zip)")
    print("=" * 80)

    if training_complete:
        from src.rl.training.plot_thesis_figures import generate_plots

        generate_plots(run_dir)
