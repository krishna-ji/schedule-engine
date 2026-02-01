#!/usr/bin/env python3
"""
RL Experiment 05: Reward Shaping

Compare scalar vs hypervolume reward calculations for RL training.

Usage:
    python runs/rl_05_reward_shaping.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.notebooks import (
    build_notebook_config,
    create_env,
    load_context,
    set_global_seed,
)
from schedule_engine.rl.gym_env.reward_calculator import RewardCalculator


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "rl_05_reward_shaping.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_05_reward_shaping")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 05: Reward Shaping."""

    # CONFIGURATION

    SEED = 42
    POP_SIZE = 20
    MAX_GENERATIONS = 30
    MAX_STEPS = 10
    NUM_TRANSITIONS = 10  # Number of transitions to sample for comparison

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_05_reward_shaping" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 05: REWARD SHAPING")
    logger.info("=" * 60)
    logger.info(
        f"Config: pop={POP_SIZE}, ngen={MAX_GENERATIONS}, transitions={NUM_TRANSITIONS}"
    )
    logger.info(f"Output: {OUTPUT_DIR}")

    # LOAD DATA & CREATE ENVIRONMENT

    logger.info("Loading data and creating environment...")
    set_global_seed(SEED)

    config = build_notebook_config(seed=SEED, overrides={"pop_size": POP_SIZE})
    _, context = load_context(DATA_DIR, config)

    env = create_env(
        context=context,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        max_steps=MAX_STEPS,
    )

    logger.info(
        f"Environment created with population of {len(env.population)} individuals"
    )

    # COMPARE REWARD CALCULATION METHODS

    logger.info("Comparing reward calculation methods...")
    scalar_calc = RewardCalculator(use_hypervolume=False)
    hv_calc = RewardCalculator(use_hypervolume=True)

    population = env.population
    comparison_results: list[dict[str, object]] = []

    for i in range(min(NUM_TRANSITIONS, len(population) - 1)):
        prev_ind = population[i]
        new_ind = population[i + 1]

        # Calculate rewards using both methods
        scalar_reward, scalar_components = scalar_calc.calculate_reward(
            prev_individual=prev_ind,
            new_individual=new_ind,
            population_diversity=0.1,
            generation=i,
            population=population,
        )

        hv_reward, hv_components = hv_calc.calculate_reward(
            prev_individual=prev_ind,
            new_individual=new_ind,
            population_diversity=0.1,
            generation=i,
            population=population,
        )

        comparison_results.append(
            {
                "transition": i,
                "scalar_reward": scalar_reward,
                "scalar_components": scalar_components,
                "hv_reward": hv_reward,
                "hv_components": hv_components,
            }
        )

        logger.info(
            f"Transition {i}: Scalar={scalar_reward:.4f}, Hypervolume={hv_reward:.4f}"
        )

    # RESULTS SUMMARY

    scalar_rewards = [r["scalar_reward"] for r in comparison_results]
    hv_rewards = [r["hv_reward"] for r in comparison_results]

    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 05: REWARD SHAPING COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info("Scalar Reward Statistics:")
    logger.info(f"  Mean: {np.mean(scalar_rewards):.4f}")
    logger.info(f"  Std:  {np.std(scalar_rewards):.4f}")
    logger.info(f"  Min:  {np.min(scalar_rewards):.4f}")
    logger.info(f"  Max:  {np.max(scalar_rewards):.4f}")

    logger.info("Hypervolume Reward Statistics:")
    logger.info(f"  Mean: {np.mean(hv_rewards):.4f}")
    logger.info(f"  Std:  {np.std(hv_rewards):.4f}")
    logger.info(f"  Min:  {np.min(hv_rewards):.4f}")
    logger.info(f"  Max:  {np.max(hv_rewards):.4f}")

    correlation = float(np.corrcoef(scalar_rewards, hv_rewards)[0, 1])
    logger.info(f"Correlation between methods: {correlation:.4f}")
    logger.info("=" * 60)

    # SAVE RESULTS

    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_05_reward_shaping",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "max_generations": MAX_GENERATIONS,
            "max_steps": MAX_STEPS,
            "num_transitions": NUM_TRANSITIONS,
        },
        "results": {
            "scalar": {
                "mean": float(np.mean(scalar_rewards)),
                "std": float(np.std(scalar_rewards)),
                "min": float(np.min(scalar_rewards)),
                "max": float(np.max(scalar_rewards)),
            },
            "hypervolume": {
                "mean": float(np.mean(hv_rewards)),
                "std": float(np.std(hv_rewards)),
                "min": float(np.min(hv_rewards)),
                "max": float(np.max(hv_rewards)),
            },
            "correlation": correlation,
            "transition_details": comparison_results,
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 05 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
