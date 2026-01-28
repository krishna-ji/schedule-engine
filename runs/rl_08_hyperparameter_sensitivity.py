#!/usr/bin/env python3
"""
RL Experiment 08: Hyperparameter Sensitivity

Analyze sensitivity to learning rate hyperparameter for PPO.

Usage:
    python runs/rl_08_hyperparameter_sensitivity.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
    log_file = output_dir / "rl_08_hyperparameter_sensitivity.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_08_hyperparameter_sensitivity")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 08: Hyperparameter Sensitivity."""
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    SEED = 42
    POP_SIZE = 20
    MAX_GENERATIONS = 40
    MAX_STEPS = 15
    TIMESTEPS = 3000

    # Learning rate sweep
    LEARNING_RATES = [1e-4, 3e-4, 1e-3]

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = (
        PROJECT_ROOT / "output" / "rl_08_hyperparameter_sensitivity" / TIMESTAMP
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 08: HYPERPARAMETER SENSITIVITY")
    logger.info("=" * 60)
    logger.info(
        f"Config: pop={POP_SIZE}, ngen={MAX_GENERATIONS}, timesteps={TIMESTEPS}"
    )
    logger.info(f"Learning rates to test: {LEARNING_RATES}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    logger.info("Loading data...")
    set_global_seed(SEED)

    config = build_notebook_config(seed=SEED, overrides={"pop_size": POP_SIZE})
    _, context = load_context(DATA_DIR, config)
    logger.info("Scheduling context loaded")

    # ==========================================================================
    # LEARNING RATE SWEEP
    # ==========================================================================
    logger.info("Running learning rate sweep...")
    sweep_results: list[dict[str, Any]] = []

    for lr in LEARNING_RATES:
        logger.info(f"Testing learning_rate={lr:.0e}...")

        # Create fresh environment for each run
        env = create_env(
            context=context,
            pop_size=POP_SIZE,
            max_generations=MAX_GENERATIONS,
            max_steps=MAX_STEPS,
        )

        # Train with this learning rate
        agent, train_time = train_agent(
            agent_type="ppo",
            env=env,
            timesteps=TIMESTEPS,
            seed=SEED,
            learning_rate=lr,
        )

        # Evaluate
        result = evaluate_agent(agent, env, max_generations=MAX_GENERATIONS)

        sweep_results.append(
            {
                "learning_rate": lr,
                "train_time": train_time,
                "best_fitness": result.best_fitness,
                "convergence_gen": result.convergence_gen,
            }
        )

        logger.info(
            f"  lr={lr:.0e}: best={result.best_fitness}, conv={result.convergence_gen}"
        )

    # ==========================================================================
    # RESULTS SUMMARY
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 08: HYPERPARAMETER SENSITIVITY RESULTS")
    logger.info("=" * 60)
    logger.info("Learning Rate Sweep Results:")
    logger.info(
        f"{'lr':>10s} | {'fitness':>10s} | {'conv_gen':>10s} | {'train_time':>10s}"
    )
    logger.info("-" * 45)
    for r in sweep_results:
        logger.info(
            f"{r['learning_rate']:.0e} | {r['best_fitness']:>10.2f} | {r['convergence_gen']:>10d} | {r['train_time']:>10.2f}s"
        )

    # Find best learning rate
    best_result = min(sweep_results, key=lambda x: x["best_fitness"])
    logger.info(
        f"Best learning rate: {best_result['learning_rate']:.0e} (fitness={best_result['best_fitness']})"
    )
    logger.info("=" * 60)

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_08_hyperparameter_sensitivity",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "max_generations": MAX_GENERATIONS,
            "max_steps": MAX_STEPS,
            "timesteps": TIMESTEPS,
            "learning_rates": LEARNING_RATES,
        },
        "results": {
            "sweep_results": sweep_results,
            "best_learning_rate": best_result["learning_rate"],
            "best_fitness": best_result["best_fitness"],
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 08 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
