#!/usr/bin/env python3
r"""RL 06 FAST — Emergency Speed-Optimized Training Script.

Uses simplified random repair operators instead of expensive micro-memetic
optimizers to get training working first. Once this works, we can gradually
add complexity back.

Key Changes:
- Disabled micro-memetic optimizers
- Using simple random domain sampling
- Reduced population size (30 individuals)
- Shorter episodes (15 generations)
- Focus on getting PPO learning working correctly

Usage::
    python runs/rl_06_fast_training.py

Expected: ~2-5 minutes per episode instead of 30 seconds
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
logger = logging.getLogger("rl_06_fast")

# ======================================================================
# EMERGENCY FAST CONFIGURATION
# ======================================================================

SEED = 42
POP_SIZE = 30  # Very small for speed
MAX_GENERATIONS = 15  # Short episodes
TOTAL_TIMESTEPS = 10_000  # Quick test first
LEARNING_RATE = 3e-4
PKL_PATH = ".cache/events_with_domains.pkl"


def train_fast():
    """Fast training with simplified operators."""

    # Import dependencies
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        logger.error("sb3-contrib not found! Install it with: pip install sb3-contrib")
        sys.exit(1)

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "fast_ppo" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RL 06 FAST — Speed-Optimized Emergency Training")
    logger.info("  run_dir           : %s", run_dir)
    logger.info(
        "  pop_size: %d  max_gen: %d  timesteps: %d",
        POP_SIZE,
        MAX_GENERATIONS,
        TOTAL_TIMESTEPS,
    )
    logger.info("=" * 60)

    # -- FAST Environment with Simple Operators ---------------------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=0.0,  # No tolerance - strict evaluation
    )

    # Verify action masking
    if not hasattr(env, "action_masks"):
        logger.error("Environment missing action_masks() method!")
        sys.exit(1)

    logger.info("Environment configured with action masking")

    # -- Simple MaskablePPO Agent ------------------------------------------
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        policy_kwargs=dict(net_arch=[32, 32]),  # Smaller network
        seed=SEED,
        verbose=1,
    )

    logger.info("MaskablePPO initialized with simplified config")

    # -- Basic Callback ----------------------------------------------------
    class FastCallback(BaseCallback):
        def __init__(self, run_dir: Path):
            super().__init__(verbose=0)
            self.run_dir = run_dir
            self.episode_count = 0

        def _on_rollout_end(self) -> None:
            if hasattr(self.locals, "infos") and self.locals["infos"]:
                for info in self.locals["infos"]:
                    if info and "episode" in info:
                        self.episode_count += 1
                        if self.episode_count % 10 == 0:
                            logger.info(
                                "Episode %d: R=%.3f, hard=%.1f",
                                self.episode_count,
                                info.get("episode", {}).get("r", 0),
                                info.get("best_hard", np.inf),
                            )

        def _on_step(self) -> bool:
            return True

    callback = FastCallback(run_dir)

    # -- FAST TRAINING -----------------------------------------------------
    logger.info("Starting FAST training...")
    t0 = time.perf_counter()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        log_interval=5,
    )

    train_time = time.perf_counter() - t0
    logger.info(
        "FAST training complete in %.1fs (%.1f min)", train_time, train_time / 60
    )

    # -- Save Model --------------------------------------------------------
    model_path = run_dir / "fast_ppo_model.zip"
    model.save(str(model_path))
    logger.info("Model saved: %s", model_path)

    env.close()
    return model, run_dir


def main():
    try:
        model, run_dir = train_fast()

        logger.info("=" * 60)
        logger.info("✅ FAST TRAINING COMPLETE")
        logger.info("  Output: %s", run_dir)
        logger.info("=" * 60)

        print("\\n🚀 SUCCESS: Fast training completed!")
        print("   Episodes should now run for full", MAX_GENERATIONS, "generations")
        print("   Next: Scale up to full micro-memetic training")

    except Exception as e:
        logger.error("FAST training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
