#!/usr/bin/env python3
"""
RL Train Curriculum: Staged Difficulty Training

Train PPO with curriculum learning - progressive phases:
  Phase 1: Easy (few constraints)
  Phase 2: Medium (more constraints)
  Phase 3: Full (all constraints)

Usage:
    python runs/rl_03_train_curriculum.py
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

from src.rl.helpers import (
    build_notebook_config,
    create_env,
    evaluate_agent,
    load_context,
    set_global_seed,
    train_agent,
)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "rl_03_train_curriculum.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_03_train_curriculum")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 03: Curriculum Learning."""

    # CONFIGURATION

    SEED = 42
    POP_SIZE = 20

    # Curriculum stages: gradually increase difficulty
    STAGES = [
        {"name": "easy", "max_generations": 30, "max_steps": 10, "timesteps": 3000},
        {"name": "medium", "max_generations": 50, "max_steps": 15, "timesteps": 4000},
        {"name": "hard", "max_generations": 80, "max_steps": 20, "timesteps": 5000},
    ]

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_03_train_curriculum" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 03: CURRICULUM LEARNING")
    logger.info("=" * 60)
    logger.info(f"Config: pop={POP_SIZE}, stages={len(STAGES)}")
    logger.info(f"Stages: {[s['name'] for s in STAGES]}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # LOAD DATA

    logger.info("Loading data...")
    set_global_seed(SEED)

    config = build_notebook_config(seed=SEED, overrides={"pop_size": POP_SIZE})
    _, context = load_context(DATA_DIR, config)
    logger.info("Scheduling context loaded")

    # CURRICULUM TRAINING LOOP

    logger.info("Starting curriculum training...")
    agent = None
    stage_results: list[dict[str, object]] = []
    total_train_time = 0.0

    for i, stage in enumerate(STAGES):
        logger.info("=" * 50)
        logger.info(f"STAGE {i+1}/{len(STAGES)}: {stage['name'].upper()}")
        logger.info("=" * 50)

        # Create environment for this stage
        env = create_env(
            context=context,
            pop_size=POP_SIZE,
            max_generations=stage["max_generations"],
            max_steps=stage["max_steps"],
        )

        # Train or continue training
        if agent is None:
            agent, train_time = train_agent(
                agent_type="ppo",
                env=env,
                timesteps=stage["timesteps"],
                seed=SEED,
            )
        else:
            agent.set_env(env)
            start = time.time()
            agent.learn(total_timesteps=stage["timesteps"], progress_bar=False)
            train_time = time.time() - start

        total_train_time += train_time

        # Evaluate at this stage
        result = evaluate_agent(agent, env, max_generations=stage["max_generations"])
        stage_results.append(
            {
                "stage": stage["name"],
                "train_time": train_time,
                "best_fitness": result.best_fitness,
                "convergence_gen": result.convergence_gen,
            }
        )

        logger.info(
            f"Stage {stage['name']}: best={result.best_fitness}, conv={result.convergence_gen} (train={train_time:.2f}s)"
        )

    # RESULTS SUMMARY

    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 03: CURRICULUM LEARNING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total training time: {total_train_time:.2f}s")
    logger.info("Per-stage results:")
    for sr in stage_results:
        logger.info(
            f"  {sr['stage']:8s}: fitness={sr['best_fitness']}, conv={sr['convergence_gen']}, time={sr['train_time']:.2f}s"
        )
    logger.info("=" * 60)

    # SAVE RESULTS

    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_03_train_curriculum",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "stages": STAGES,
        },
        "results": {
            "total_train_time_seconds": total_train_time,
            "stage_results": stage_results,
            "final_best_fitness": stage_results[-1]["best_fitness"],
            "final_convergence_gen": stage_results[-1]["convergence_gen"],
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 03 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
