#!/usr/bin/env python3
"""
RL Ablation: Systematic Method Comparison

Compare repair selection strategies:
  - Random (baseline)
  - PPO (policy gradient)
  - DQN (value-based)

Usage:
    python runs/rl_07_ablation.py
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

from src.rl.helpers import run_ablation


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "rl_07_ablation.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_07_ablation")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 07: Full Ablation Study."""

    # CONFIGURATION

    SEED = 42
    POP_SIZE = 20
    MAX_GENERATIONS = 50
    MAX_STEPS = 20
    TIMESTEPS = 3000  # Per method
    TRIALS = 5  # Statistical significance

    # Methods to compare
    METHODS = {
        "random": {"agent_type": "random"},
        "ppo": {"agent_type": "ppo"},
        "dqn": {"agent_type": "dqn"},
    }

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_07_ablation" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 07: FULL ABLATION STUDY")
    logger.info("=" * 60)
    logger.info(f"Config: methods={list(METHODS.keys())}, trials={TRIALS}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # RUN ABLATION STUDY

    logger.info(f"Running ablation study: {len(METHODS)} methods × {TRIALS} trials...")
    logger.info("This may take several minutes...")

    results = run_ablation(
        methods=METHODS,
        data_dir=DATA_DIR,
        trials=TRIALS,
        timesteps=TIMESTEPS,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        max_steps=MAX_STEPS,
        seed=SEED,
    )

    logger.info("Ablation study completed")

    # RESULTS SUMMARY

    logger.info("=" * 70)
    logger.info("RL EXPERIMENT 07: FULL ABLATION STUDY RESULTS")
    logger.info("=" * 70)

    # Compute statistics per method
    method_stats: dict[str, dict[str, object]] = {}
    for method_key, runs in results.items():
        best_fitness_vals = [r.best_fitness for r in runs]
        conv_vals = [r.convergence_gen for r in runs]

        method_stats[method_key] = {
            "best_fitness_mean": float(np.mean(best_fitness_vals)),
            "best_fitness_std": float(np.std(best_fitness_vals)),
            "convergence_mean": float(np.mean(conv_vals)),
            "convergence_std": float(np.std(conv_vals)),
            "best_fitness_all": best_fitness_vals,
            "convergence_all": conv_vals,
        }

        logger.info(f"{method_key.upper()}:")
        logger.info(
            f"  Best Fitness: {np.mean(best_fitness_vals):.2f} ± {np.std(best_fitness_vals):.2f}"
        )
        logger.info(
            f"  Convergence:  {np.mean(conv_vals):.1f} ± {np.std(conv_vals):.1f} generations"
        )
        logger.info(f"  Raw values:   fitness={best_fitness_vals}, conv={conv_vals}")

    logger.info("=" * 70)

    # SAVE RESULTS

    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_07_ablation",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "max_generations": MAX_GENERATIONS,
            "max_steps": MAX_STEPS,
            "timesteps": TIMESTEPS,
            "trials": TRIALS,
            "methods": list(METHODS.keys()),
        },
        "results": {
            method: {
                "best_fitness_mean": stats["best_fitness_mean"],
                "best_fitness_std": stats["best_fitness_std"],
                "convergence_mean": stats["convergence_mean"],
                "convergence_std": stats["convergence_std"],
                "best_fitness_all": stats["best_fitness_all"],
                "convergence_all": stats["convergence_all"],
            }
            for method, stats in method_stats.items()
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 07 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
