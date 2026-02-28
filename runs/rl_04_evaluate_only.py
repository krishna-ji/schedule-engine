#!/usr/bin/env python3
"""
Evaluation-only script for the 200-generation deterministic run.
Use this if you have a pre-trained PPO model ready.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

# ==================== CONFIGURATION ====================
EVAL_GENERATIONS = 200
EVAL_POP_SIZE = 120
EVAL_ACCEPTANCE_TOLERANCE = 0.0  # Strict exploitation
SEED = 42
PKL_PATH = ".cache/events_with_domains.pkl"

# Directory setup
OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_capstone_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create run-specific subdirectory
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_dir = OUTPUT_DIR / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_trained_model(model_path: Path) -> None:
    """Run 200-generation evaluation with a trained PPO model."""

    logger.info("Starting 200-generation evaluation")
    logger.info("Model path: %s", model_path)
    logger.info("Run directory: %s", run_dir)

    # ==================== ENVIRONMENT SETUP ====================
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_GENERATIONS,
        pop_size=EVAL_POP_SIZE,
        algorithm_name="nsga2",
        acceptance_tolerance=EVAL_ACCEPTANCE_TOLERANCE,
        seed=SEED,
    )

    # Load the trained model
    logger.info("Loading trained PPO model...")
    model = PPO.load(str(model_path), env=env)

    # ==================== EVALUATION LOOP ====================
    logger.info("Beginning deterministic evaluation...")

    # Reset environment
    obs, info = env.reset(seed=SEED)
    generation_data = []

    t0 = time.perf_counter()

    for step in range(EVAL_GENERATIONS):
        # Deterministic action (no randomness)
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # Extract metrics from the environment
        best_hard = info.get("best_hard", 999999)
        best_soft = info.get("best_soft", 999999)
        current_gen = info.get("generation", step + 1)

        # Log progress every 20 generations
        if (step + 1) % 20 == 0:
            logger.info(
                "Generation %d/%d | Best_Hard=%.2f | Best_Soft=%.2f | Reward=%.4f",
                current_gen,
                EVAL_GENERATIONS,
                best_hard,
                best_soft,
                reward,
            )

        # Store data for CSV
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

    eval_time = time.perf_counter() - t0
    logger.info("Evaluation complete in %.1fs", eval_time)

    # ==================== SAVE RESULTS ====================
    # Save generation trajectory
    df = pd.DataFrame(generation_data)
    csv_path = run_dir / "evaluation_trajectory_200.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved evaluation trajectory: %s", csv_path)

    # Report final results
    if generation_data:
        final_gen = generation_data[-1]
        logger.info("=" * 60)
        logger.info("FINAL RESULTS (Generation %d):", final_gen["generation"])
        logger.info("Best_Hard: %.2f", final_gen["best_hard"])
        logger.info("Best_Soft: %.2f", final_gen["best_soft"])
        logger.info("Final Reward: %.4f", final_gen["reward"])
        logger.info("=" * 60)

        print("\n🎉 CAPSTONE EVALUATION COMPLETE! 🎉")
        print(f"Final Best_Hard: {final_gen['best_hard']:.2f}")
        print(f"Final Best_Soft: {final_gen['best_soft']:.2f}")
        print(f"Results saved to: {csv_path}")

    env.close()


if __name__ == "__main__":
    # Look for the canonical model or ask user to specify
    canonical_model = PROJECT_ROOT / "output" / "models" / "ppo_capstone_final.zip"

    if canonical_model.exists():
        logger.info("Found canonical model: %s", canonical_model)
        evaluate_trained_model(canonical_model)
    else:
        print("❌ No trained model found at:", canonical_model)
        print("\nPlease:")
        print("1. Re-run the training with the fixed script, OR")
        print("2. Specify the path to your trained model")
        print("\nExpected model location:", canonical_model)
        sys.exit(1)
