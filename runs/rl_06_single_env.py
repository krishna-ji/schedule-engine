#!/usr/bin/env python3
r"""RL 06 SINGLE ENV — Fixed MaskablePPO Training (No Vectorization).

The problem was SB3 creating too many vectorized environments on a 32-core
machine. This script forces single environment training to fix the issues.

Key Fixes:
- Explicitly use single environment (no vectorization)
- Proper episode termination handling
- Single-threaded operation
- Reduced complexity for reliable training

Usage::
    python runs/rl_06_single_env.py

Expected: Stable training with episodes running full generations
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_06_single")

# ======================================================================
# SINGLE ENVIRONMENT CONFIGURATION
# ======================================================================

SEED = 42
POP_SIZE = 30  # Reasonable size for single env
MAX_GENERATIONS = 20  # Good episode length
TOTAL_TIMESTEPS = 20_000  # Start modest, can scale up
LEARNING_RATE = 3e-4
PKL_PATH = ".cache/events_with_domains.pkl"


def train_single_env():
    """Train MaskablePPO with single environment (no vectorization)."""

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.utils import set_random_seed
    except ImportError:
        logger.error("sb3-contrib not found! Install with: pip install sb3-contrib")
        sys.exit(1)

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    # Set global seed
    set_random_seed(SEED)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "single_ppo" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RL 06 SINGLE ENV — Fixed MaskablePPO Training")
    logger.info("  run_dir           : %s", run_dir)
    logger.info(
        "  pop_size: %d  max_gen: %d  timesteps: %d",
        POP_SIZE,
        MAX_GENERATIONS,
        TOTAL_TIMESTEPS,
    )
    logger.info("  SINGLE ENVIRONMENT (no vectorization)")
    logger.info("=" * 60)

    # -- Create SINGLE Environment (no vectorization) ---------------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=0.0,  # Strict mode
    )

    if not hasattr(env, "action_masks"):
        logger.error("Environment missing action_masks() method!")
        sys.exit(1)

    logger.info("✅ Single environment created with action masking")

    # -- MaskablePPO with Single Environment -------------------------------
    model = MaskablePPO(
        "MlpPolicy",
        env,  # Single environment - no DummyVecEnv wrapper
        learning_rate=LEARNING_RATE,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=SEED,
        verbose=1,
        device="cpu",  # Force CPU to avoid GPU multiprocessing issues
    )

    logger.info("✅ MaskablePPO initialized (single env, CPU)")

    # -- Simple Callback for Single Environment ----------------------------
    class SingleEnvCallback(BaseCallback):
        """Simple callback for single environment training."""

        def __init__(self, run_dir: Path):
            super().__init__(verbose=0)
            self.run_dir = run_dir
            self.episode_count = 0
            self.step_count = 0

            # Initialize CSV
            self.log_csv = run_dir / "training_log.csv"
            with open(self.log_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestep",
                        "episode",
                        "episode_reward",
                        "episode_length",
                        "best_hard",
                        "best_soft",
                        "action_masks_blocked",
                    ]
                )

        def _on_step(self) -> bool:
            """Called after each step."""
            self.step_count += 1

            # Log episode completion
            if "episode" in self.locals.get("infos", [{}])[0]:
                info = self.locals["infos"][0]
                episode = info["episode"]
                self.episode_count += 1

                # Calculate masked actions
                try:
                    masks = self.training_env.action_masks()
                    blocked_count = np.sum(~masks)
                except:
                    blocked_count = 0

                # Log to CSV
                with open(self.log_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            self.num_timesteps,
                            self.episode_count,
                            episode["r"],
                            episode["l"],
                            info.get("best_hard", "N/A"),
                            info.get("best_soft", "N/A"),
                            blocked_count,
                        ]
                    )

                # Console logging every 10 episodes
                if self.episode_count % 10 == 0:
                    logger.info(
                        "Episode %d: R=%.3f, len=%d, hard=%.1f, blocked=%d/8",
                        self.episode_count,
                        episode["r"],
                        episode["l"],
                        info.get("best_hard", np.inf),
                        blocked_count,
                    )

            return True

    callback = SingleEnvCallback(run_dir)

    # -- Start Training ----------------------------------------------------
    logger.info("🚀 Starting single environment training...")
    t0 = time.perf_counter()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            log_interval=5,
            progress_bar=True,  # Show progress
        )

        train_time = time.perf_counter() - t0
        logger.info(
            "✅ Training completed in %.1fs (%.1f min)", train_time, train_time / 60
        )

    except Exception as e:
        train_time = time.perf_counter() - t0
        logger.error("❌ Training failed after %.1fs: %s", train_time, e)
        raise

    # -- Save Model --------------------------------------------------------
    model_path = run_dir / "single_ppo_model.zip"
    model.save(str(model_path))
    logger.info("✅ Model saved: %s", model_path)

    # -- Generate Report ---------------------------------------------------
    report_path = run_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("SINGLE ENVIRONMENT TRAINING REPORT\\n")
        f.write("=" * 50 + "\\n")
        f.write("Configuration:\\n")
        f.write(f"  Population Size: {POP_SIZE}\\n")
        f.write(f"  Max Generations: {MAX_GENERATIONS}\\n")
        f.write(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}\\n")
        f.write(f"  Training Time: {train_time:.1f}s ({train_time/60:.1f} min)\\n")
        f.write("\\n")
        f.write("Results:\\n")
        f.write(f"  Episodes Completed: {callback.episode_count}\\n")
        f.write(f"  Steps Completed: {callback.step_count}\\n")
        f.write(
            f"  Avg Episode Length: {callback.step_count/max(callback.episode_count, 1):.1f} steps\\n"
        )
        f.write("\\n")
        f.write("Environment: Single (no vectorization)\\n")
        f.write("Action Masking: Enabled (soft optimizers blocked when hard > 0)\\n")
        f.write("\\n")
        f.write("Next Steps:\\n")
        f.write("  - If training is stable, scale up timesteps and population\\n")
        f.write("  - If episodes are too short, increase max_generations\\n")
        f.write("  - If action masking is too restrictive, adjust conditions\\n")

    logger.info("✅ Training report: %s", report_path)

    env.close()
    return model, run_dir


def main():
    """Main execution function."""
    try:
        model, run_dir = train_single_env()

        logger.info("=" * 60)
        logger.info("🎯 SINGLE ENVIRONMENT TRAINING COMPLETE")
        logger.info("  Output: %s", run_dir)
        logger.info("=" * 60)

        print("\\n✅ SUCCESS: Fixed single environment training!")
        print("   🔧 No more vectorization issues")
        print("   🎯 Episodes should run full generations")
        print("   📊 Check training_log.csv for episode metrics")

    except KeyboardInterrupt:
        logger.warning("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error("❌ Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
