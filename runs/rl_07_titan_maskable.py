#!/usr/bin/env python3
r"""RL 07 — The "Titan" SOTA Run: MaskablePPO + Micro-Memetic Elite 8.

Definitive 150,000-timestep training run combining:
  - **Single-environment** MaskablePPO (no vectorization overhead)
  - **State-conditioned action masking** (soft optimizers blocked when
    hard constraints are violated)
  - **Micro-memetic heuristics** restored:
      * SpatialResourceProjection: conflict-directed $k{=}5$ greedy bursts
      * UniversalFeasibilityProjection: bounded depth-3 ejection chains
      * MeridianCompactionHeuristic: feasibility-gated soft optimizer

Configuration (thesis parameters):
  pop_size = 120, max_generations = 50, timesteps = 150,000

Usage::

    python runs/rl_07_titan_maskable.py

Outputs (in ``output/titan/<timestamp>/``)::

    maskable_ppo_titan.zip     — final trained model
    titan_training_log.csv     — per-episode metrics
    titan_step_log.csv         — per-step metrics with mask stats
    titan_report.txt           — training summary report
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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
logger = logging.getLogger("rl_07_titan")

# ======================================================================
# TITAN CONFIGURATION — thesis-grade parameters
# ======================================================================

SEED = 42
POP_SIZE = 120  # Full population
MAX_GENERATIONS = 50  # Full episode length
TOTAL_TIMESTEPS = 100_000
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
NET_ARCH = [64, 64]
N_STEPS = 2048  # PPO rollout buffer
BATCH_SIZE = 64
N_EPOCHS = 10
GAE_LAMBDA = 0.95
GAMMA = 0.99
ENT_COEF = 0.01  # Encourage exploration
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
PKL_PATH = ".cache/events_with_domains.pkl"
ACCEPTANCE_TOLERANCE = 5.0  # Allow some degradation for exploration


# ======================================================================
# Titan Training
# ======================================================================


def train_titan():
    """Execute the Titan training run: MaskablePPO + Micro-Memetic Elite 8."""

    # -- Import MaskablePPO ------------------------------------------------
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.utils import set_random_seed
    except ImportError:
        logger.error("sb3-contrib not found! Install: pip install sb3-contrib")
        sys.exit(1)

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    # Set global seed
    set_random_seed(SEED)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "titan" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  RL 07 — THE TITAN SOTA RUN")
    logger.info("  MaskablePPO + Micro-Memetic Elite 8 Heuristics")
    logger.info("=" * 70)
    logger.info("  run_dir           : %s", run_dir)
    logger.info("  pop_size          : %d", POP_SIZE)
    logger.info("  max_generations   : %d", MAX_GENERATIONS)
    logger.info("  total_timesteps   : %d", TOTAL_TIMESTEPS)
    logger.info("  n_steps (rollout) : %d", N_STEPS)
    logger.info("  batch_size        : %d", BATCH_SIZE)
    logger.info("  net_arch          : %s", NET_ARCH)
    logger.info("  learning_rate     : %.1e", LEARNING_RATE)
    logger.info("  acceptance_tol    : %.1f", ACCEPTANCE_TOLERANCE)
    logger.info("=" * 70)

    # -- Create SINGLE Environment (no vectorization) ----------------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=ACCEPTANCE_TOLERANCE,
    )

    assert hasattr(env, "action_masks"), "Environment must support action_masks()"
    logger.info("Environment created with action masking support")

    # Verify Elite 8 action space is active
    from src.rl.actions.vectorized_ops import ACTION_NAMES, VECTORIZED_ACTION_SPACE

    logger.info("Action space (%d actions):", len(VECTORIZED_ACTION_SPACE))
    for aid, name in ACTION_NAMES.items():
        logger.info("  %d: %s", aid, name)

    # -- MaskablePPO Agent -------------------------------------------------
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=SEED,
        verbose=1,
        device="cpu",  # Force CPU — avoids vectorization overhead
    )

    logger.info("MaskablePPO initialized (CPU, single env)")

    # -- Titan Callback: full logging + mask statistics --------------------
    class TitanCallback(BaseCallback):
        """Comprehensive callback for Titan run logging."""

        def __init__(self, run_dir: Path):
            super().__init__(verbose=0)
            self.run_dir = run_dir
            self.episode_count = 0
            self.step_count = 0
            self.best_hard_ever = np.inf
            self.best_soft_ever = np.inf
            self.mask_blocked_total = 0
            self.mask_total = 0
            self.episode_rewards = []
            self.t_start = time.perf_counter()

            # Episode CSV
            self.ep_csv = run_dir / "titan_training_log.csv"
            with open(self.ep_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "episode",
                        "timestep",
                        "ep_reward",
                        "ep_length",
                        "best_hard",
                        "best_soft",
                        "feasible_frac",
                        "mask_blocked_pct",
                        "wall_time_s",
                        "best_hard_ever",
                        "best_soft_ever",
                    ]
                )

            # Step CSV
            self.step_csv = run_dir / "titan_step_log.csv"
            with open(self.step_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "timestep",
                        "action",
                        "action_name",
                        "reward",
                        "best_hard",
                        "best_soft",
                        "rejected",
                        "delta_hard",
                        "mask_3_blocked",
                        "mask_7_blocked",
                    ]
                )

        def _on_step(self) -> bool:
            self.step_count += 1

            # Track mask usage
            try:
                env_inner = self.training_env.envs[0]
                masks = env_inner.action_masks()
                self.mask_total += 1
                if not masks[3] or not masks[7]:
                    self.mask_blocked_total += 1
            except Exception:
                pass

            # Detect episode end via infos
            infos = self.locals.get("infos", [{}])
            if infos and "episode" in infos[0]:
                info = infos[0]
                ep = info["episode"]
                self.episode_count += 1
                self.episode_rewards.append(ep["r"])

                best_h = info.get("best_hard", np.inf)
                best_s = info.get("best_soft", np.inf)
                self.best_hard_ever = min(self.best_hard_ever, best_h)
                self.best_soft_ever = min(self.best_soft_ever, best_s)

                mask_pct = (self.mask_blocked_total / max(self.mask_total, 1)) * 100
                wall_t = time.perf_counter() - self.t_start

                with open(self.ep_csv, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            self.episode_count,
                            self.num_timesteps,
                            f"{ep['r']:.6f}",
                            ep["l"],
                            best_h,
                            best_s,
                            info.get("feasible_frac", 0.0),
                            f"{mask_pct:.1f}",
                            f"{wall_t:.1f}",
                            self.best_hard_ever,
                            self.best_soft_ever,
                        ]
                    )

                if self.episode_count % 25 == 0:
                    fps = self.num_timesteps / wall_t if wall_t > 0 else 0
                    logger.info(
                        "EP %4d | ts=%6d | R=%+.3f | hard=%7.1f | "
                        "best_ever=%7.1f | soft=%7.1f | mask%%=%.0f | "
                        "FPS=%.0f | wall=%.0fs",
                        self.episode_count,
                        self.num_timesteps,
                        ep["r"],
                        best_h,
                        self.best_hard_ever,
                        best_s,
                        mask_pct,
                        fps,
                        wall_t,
                    )

            return True

    callback = TitanCallback(run_dir)

    # -- FPS Estimate ------------------------------------------------------
    # Quick 3-step benchmark to estimate total training time
    logger.info("Running FPS benchmark (3 steps)...")
    obs, info = env.reset()
    t_bench = time.perf_counter()
    for _ in range(3):
        masks = env.action_masks()
        valid = np.where(masks)[0]
        obs, reward, term, trunc, info = env.step(np.random.choice(valid))
        if term or trunc:
            obs, info = env.reset()
    fps_estimate = 3.0 / (time.perf_counter() - t_bench)
    env.close()

    eta_seconds = TOTAL_TIMESTEPS / fps_estimate
    eta_minutes = eta_seconds / 60
    logger.info(
        "FPS estimate: %.1f steps/s | ETA: %.0f seconds (%.1f minutes)",
        fps_estimate,
        eta_seconds,
        eta_minutes,
    )

    # Re-create environment (reset after benchmark)
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=ACCEPTANCE_TOLERANCE,
    )
    model.set_env(env)

    # -- TRAIN -------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("  STARTING TITAN TRAINING: %d timesteps", TOTAL_TIMESTEPS)
    logger.info("  Estimated completion: %.1f minutes", eta_minutes)
    logger.info("=" * 70)

    t0 = time.perf_counter()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        log_interval=10,
        progress_bar=True,
    )

    train_time = time.perf_counter() - t0
    actual_fps = TOTAL_TIMESTEPS / train_time

    logger.info("=" * 70)
    logger.info("  TITAN TRAINING COMPLETE")
    logger.info("  Wall time: %.1fs (%.1f min)", train_time, train_time / 60)
    logger.info("  Actual FPS: %.1f", actual_fps)
    logger.info("  Episodes: %d", callback.episode_count)
    logger.info("  Best Hard (ever): %.1f", callback.best_hard_ever)
    logger.info("  Best Soft (ever): %.1f", callback.best_soft_ever)
    logger.info("=" * 70)

    # -- Save Model --------------------------------------------------------
    # Save to run directory
    model_run = run_dir / "maskable_ppo_titan.zip"
    model.save(str(model_run))
    logger.info("Model saved (run): %s", model_run)

    # Save to canonical path
    canonical_dir = PROJECT_ROOT / "output" / "models"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / "maskable_ppo_titan.zip"
    model.save(str(canonical_path))
    logger.info("Model saved (canonical): %s", canonical_path)

    # -- Generate Report ---------------------------------------------------
    ep_rewards = callback.episode_rewards
    report_path = run_dir / "titan_report.txt"
    with open(report_path, "w") as f:
        f.write("TITAN SOTA RUN — TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("CONFIGURATION\n")
        f.write(f"  Population Size    : {POP_SIZE}\n")
        f.write(f"  Max Generations    : {MAX_GENERATIONS}\n")
        f.write(f"  Total Timesteps    : {TOTAL_TIMESTEPS:,}\n")
        f.write(f"  Learning Rate      : {LEARNING_RATE}\n")
        f.write(f"  Network Arch       : {NET_ARCH}\n")
        f.write(f"  N_Steps (rollout)  : {N_STEPS}\n")
        f.write(f"  Batch Size         : {BATCH_SIZE}\n")
        f.write(f"  Acceptance Tol     : {ACCEPTANCE_TOLERANCE}\n")
        f.write(f"  Seed               : {SEED}\n\n")
        f.write("RESULTS\n")
        f.write(f"  Training Time      : {train_time:.1f}s ({train_time/60:.1f} min)\n")
        f.write(f"  Actual FPS         : {actual_fps:.1f}\n")
        f.write(f"  Episodes Completed : {callback.episode_count}\n")
        f.write(f"  Steps Completed    : {callback.step_count}\n")
        f.write(f"  Best Hard (ever)   : {callback.best_hard_ever:.1f}\n")
        f.write(f"  Best Soft (ever)   : {callback.best_soft_ever:.1f}\n\n")
        f.write("REWARD STATISTICS\n")
        if ep_rewards:
            f.write(f"  Mean Episode R     : {np.mean(ep_rewards):.4f}\n")
            f.write(f"  Std Episode R      : {np.std(ep_rewards):.4f}\n")
            f.write(f"  Min Episode R      : {np.min(ep_rewards):.4f}\n")
            f.write(f"  Max Episode R      : {np.max(ep_rewards):.4f}\n\n")
        f.write("ACTION MASKING\n")
        mask_pct = (callback.mask_blocked_total / max(callback.mask_total, 1)) * 100
        f.write(f"  Total Steps Tracked: {callback.mask_total}\n")
        f.write(f"  Steps w/ Masking   : {callback.mask_blocked_total}\n")
        f.write(f"  Mask Rate          : {mask_pct:.1f}%\n\n")
        f.write("ELITE 8 ACTION SPACE (Micro-Memetic)\n")
        for aid, name in ACTION_NAMES.items():
            f.write(f"  {aid}: {name}\n")
        f.write("\nMICRO-MEMETIC UPGRADES ACTIVE\n")
        f.write(
            "  0: SpatialResourceProjection  — conflict-directed k=5 greedy bursts\n"
        )
        f.write("  4: UniversalFeasibility       — bounded depth-3 ejection chains\n")
        f.write("  7: MeridianCompaction         — feasibility-gated soft optimizer\n")

    logger.info("Report saved: %s", report_path)

    env.close()
    return model, run_dir, callback


# ======================================================================
# Main
# ======================================================================


def main():
    try:
        model, run_dir, callback = train_titan()

        print("\n" + "=" * 70)
        print("  THE TITAN RUN IS COMPLETE")
        print("  Model:    output/models/maskable_ppo_titan.zip")
        print(f"  Logs:     {run_dir}")
        print(f"  Best Hard: {callback.best_hard_ever:.1f}")
        print(f"  Best Soft: {callback.best_soft_ever:.1f}")
        print(f"  Episodes:  {callback.episode_count}")
        print("=" * 70)

    except KeyboardInterrupt:
        logger.warning("Titan run interrupted by user")
    except Exception as e:
        logger.error("Titan run failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
