#!/usr/bin/env python3
"""
FIXED CAPSTONE SCRIPT - Phase 40: Capstone Training Run with Tolerance Exploration and Extended Evaluation Horizons
FIXES: Model saved before efficacy matrix printing, encoding-safe matrix display
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
from src.rl.training.thesis_callback import ThesisLoggingCallback

# ==================== CONFIGURATION ====================
TOTAL_TIMESTEPS = 150_000
TRAINING_GENERATIONS = 50
TRAINING_POP_SIZE = 120
TRAINING_ACCEPTANCE_TOLERANCE = 10.0  # Exploration phase

EVAL_GENERATIONS = 200
EVAL_POP_SIZE = 120
EVAL_ACCEPTANCE_TOLERANCE = 0.0  # Strict exploitation

LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
NET_ARCH = [64, 64]
SEED = 42
PKL_PATH = ".cache/events_with_domains.pkl"

# Directory setup
OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_capstone_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create run-specific subdirectory
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_dir = OUTPUT_DIR / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
log_path = run_dir / "capstone_fixed.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train() -> PPO:
    """Train the PPO agent for exactly 150,000 timesteps with tolerance exploration."""

    logger.info("Starting PPO training with tolerance exploration")
    logger.info("Training timesteps: %d", TOTAL_TIMESTEPS)
    logger.info("Training tolerance: %.1f (exploration)", TRAINING_ACCEPTANCE_TOLERANCE)
    logger.info("Run directory: %s", run_dir)

    # ==================== ENVIRONMENT SETUP ====================
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=TRAINING_GENERATIONS,
        pop_size=TRAINING_POP_SIZE,
        algorithm_name="nsga2",
        acceptance_tolerance=TRAINING_ACCEPTANCE_TOLERANCE,
        seed=SEED,
    )

    # ==================== MODEL SETUP ====================
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE,
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=SEED,
        verbose=1,
    )

    # ==================== CALLBACK ====================
    callback = ThesisLoggingCallback(run_dir=run_dir, verbose=1)

    # ==================== TRAINING ====================
    try:
        t0 = time.perf_counter()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        train_time = time.perf_counter() - t0
        logger.info(
            "Training complete in %.1fs (%.1f min)", train_time, train_time / 60
        )
    except Exception as e:
        logger.error("Training failed: %s", e)
        # Still try to save model if training completed but callback failed
        if hasattr(model, "_last_obs"):
            logger.info("Attempting to save model despite training error...")

    # ==================== SAVE MODEL (CRITICAL - DO THIS FIRST!) ====================
    try:
        # Save to run_dir
        model_run = run_dir / "ppo_capstone_final.zip"
        model.save(str(model_run))
        logger.info("Model saved (run): %s", model_run)

        # Save to canonical output/models/ path
        canonical_dir = PROJECT_ROOT / "output" / "models"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = canonical_dir / "ppo_capstone_final.zip"
        model.save(str(canonical_path))
        logger.info("Model saved (canonical): %s", canonical_path)
    except Exception as e:
        logger.error("Model saving failed: %s", e)

    # ==================== SAFE EFFICACY MATRIX PRINTING ====================
    try:
        # Try to print the efficacy matrix safely
        if hasattr(callback, "_print_efficacy_matrix"):
            logger.info("Attempting to print efficacy matrix...")
            callback._print_efficacy_matrix()
        else:
            logger.warning("No efficacy matrix method available in callback")
    except UnicodeEncodeError as e:
        logger.warning("Efficacy matrix display failed due to encoding: %s", e)
        logger.info("Training data and model were still saved successfully!")
    except Exception as e:
        logger.error("Efficacy matrix display failed: %s", e)

    env.close()
    return model


def evaluate(model: PPO) -> None:
    """Run the 200-generation deterministic evaluation phase."""

    logger.info("Starting 200-generation evaluation phase")
    logger.info(
        "Evaluation tolerance: %.1f (strict exploitation)", EVAL_ACCEPTANCE_TOLERANCE
    )

    # ==================== EVALUATION ENVIRONMENT ====================
    eval_env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=EVAL_GENERATIONS,
        pop_size=EVAL_POP_SIZE,
        algorithm_name="nsga2",
        acceptance_tolerance=EVAL_ACCEPTANCE_TOLERANCE,
        seed=SEED,
    )

    # Update model's environment
    model.set_env(eval_env)

    # ==================== EVALUATION LOOP ====================
    logger.info("Beginning deterministic evaluation...")

    # Reset environment
    obs, info = eval_env.reset(seed=SEED)
    generation_data = []

    t0 = time.perf_counter()

    for step in range(EVAL_GENERATIONS):
        # Deterministic action (no randomness)
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = eval_env.step(action)

        # Extract metrics
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

    # ==================== SAVE EVALUATION RESULTS ====================
    # Save generation trajectory
    df = pd.DataFrame(generation_data)
    csv_path = run_dir / "evaluation_trajectory_200.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved evaluation trajectory: %s", csv_path)

    # Report final results
    if generation_data:
        final_gen = generation_data[-1]
        logger.info("=" * 80)
        logger.info("CAPSTONE EVALUATION COMPLETE!")
        logger.info("FINAL RESULTS (Generation %d):", final_gen["generation"])
        logger.info("Best_Hard: %.2f", final_gen["best_hard"])
        logger.info("Best_Soft: %.2f", final_gen["best_soft"])
        logger.info("Final Reward: %.4f", final_gen["reward"])
        logger.info("Results saved to: %s", csv_path)
        logger.info("=" * 80)

        # Also print to console for user
        print("\nCAPSTONE RESULTS")
        print(f"Final Best_Hard: {final_gen['best_hard']:.2f}")
        print(f"Final Best_Soft: {final_gen['best_soft']:.2f}")
        print(f"CSV: {csv_path}")

    eval_env.close()


def main():
    logger.info("Starting FIXED Capstone Run - Phase 40")
    logger.info("Tolerance Exploration + Extended Evaluation Horizons")

    # Phase 1: Train with tolerance exploration
    trained_model = train()

    # Phase 2: Evaluate with strict tolerance
    evaluate(trained_model)

    logger.info("CAPSTONE RUN COMPLETE!")


if __name__ == "__main__":
    main()
