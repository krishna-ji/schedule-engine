#!/usr/bin/env python3
"""
RL Experiment 01: PPO Baseline

Train and evaluate PPO agent on ScheduleEnv to establish baseline RL performance.

Usage:
    python runs/rl_01_ppo_baseline.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.notebooks import (
    build_notebook_config,
    create_env,
    evaluate_agent,
    load_context,
    set_global_seed,
    train_agent,
)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "rl_01_ppo_baseline.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_01_ppo_baseline")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 01: PPO Baseline."""

    # CONFIGURATION

    SEED = 42
    POP_SIZE = 20
    MAX_GENERATIONS = 50
    MAX_STEPS = 20
    TIMESTEPS = 5000  # Training timesteps for PPO

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_01_ppo_baseline" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 01: PPO BASELINE")
    logger.info("=" * 60)
    logger.info(
        f"Config: pop={POP_SIZE}, ngen={MAX_GENERATIONS}, steps={MAX_STEPS}, timesteps={TIMESTEPS}"
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
        f"Environment created: obs_space={env.observation_space.shape}, action_space={env.action_space.n}"
    )

    # TRAIN PPO AGENT

    logger.info("Training PPO agent...")
    agent, train_time = train_agent(
        agent_type="ppo",
        env=env,
        timesteps=TIMESTEPS,
        seed=SEED,
    )
    logger.info(f"PPO agent trained in {train_time:.2f}s")

    # EVALUATE AGENT

    logger.info("Evaluating agent...")
    result = evaluate_agent(agent, env, max_generations=MAX_GENERATIONS)

    logger.info("=" * 50)
    logger.info("RL EXPERIMENT 01: PPO BASELINE RESULTS")
    logger.info("=" * 50)
    logger.info(f"Training time: {train_time:.2f}s")
    logger.info(f"Best fitness:  {result.best_fitness}")
    logger.info(f"Convergence:   Generation {result.convergence_gen}/{MAX_GENERATIONS}")
    logger.info("=" * 50)

    # SAVE RESULTS

    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_01_ppo_baseline",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "max_generations": MAX_GENERATIONS,
            "max_steps": MAX_STEPS,
            "timesteps": TIMESTEPS,
            "agent_type": "ppo",
        },
        "results": {
            "train_time_seconds": train_time,
            "best_fitness": result.best_fitness,
            "convergence_gen": result.convergence_gen,
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 01 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
