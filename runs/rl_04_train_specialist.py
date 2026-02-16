#!/usr/bin/env python3
"""
RL Train Specialist: Multi-Agent with Specialists

Train specialized agents for different constraint types:
  - Hard constraint specialist
  - Soft constraint specialist
  - Coordinator selects appropriate agent based on state

Usage:
    python runs/rl_04_train_specialist.py
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

from src.rl.helpers import (
    build_notebook_config,
    create_env,
    load_context,
    set_global_seed,
)
from src.rl.multi_agent.agent_coordinator import AgentCoordinator


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "rl_04_train_specialist.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_04_train_specialist")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 04: Specialist Agents."""

    # CONFIGURATION

    SEED = 42
    POP_SIZE = 20
    MAX_GENERATIONS = 50
    MAX_STEPS = 15
    NUM_EPISODES = 5  # Number of episodes to run for selection pattern analysis

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_04_train_specialist" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 04: SPECIALIST AGENTS")
    logger.info("=" * 60)
    logger.info(
        f"Config: pop={POP_SIZE}, ngen={MAX_GENERATIONS}, episodes={NUM_EPISODES}"
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

    # RUN AGENT SELECTION LOOP

    logger.info("Running agent selection analysis...")
    coordinator = AgentCoordinator(strategy="state_based")
    selection_history: list[list[str]] = []

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        episode_selections: list[str] = []

        for step in range(MAX_STEPS):
            # Get current state for agent selection
            state = {
                "generation": info.get("generation", 0),
                "generations_without_improvement": info.get(
                    "generations_without_improvement", 0
                ),
            }

            # Select specialist agent based on state
            agent = coordinator.select_agent(env.population, state, obs)
            episode_selections.append(agent.name)

            # Take action (random for demonstration)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        selection_history.append(episode_selections)
        logger.info(
            f"Episode {episode+1}: Selected agents: {episode_selections[:5]}..."
        )

    logger.info(f"Completed {NUM_EPISODES} episodes")

    # ANALYZE SELECTION PATTERNS

    logger.info("Analyzing selection patterns...")
    all_selections = [sel for episode in selection_history for sel in episode]
    selection_counts = Counter(all_selections)

    logger.info("=" * 50)
    logger.info("RL EXPERIMENT 04: SPECIALIST AGENTS RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total selections: {len(all_selections)}")
    logger.info("Agent selection distribution:")
    for agent_name, count in selection_counts.most_common():
        pct = 100 * count / len(all_selections)
        logger.info(f"  {agent_name}: {count} ({pct:.1f}%)")
    logger.info("=" * 50)

    # SAVE RESULTS

    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_04_train_specialist",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "max_generations": MAX_GENERATIONS,
            "max_steps": MAX_STEPS,
            "num_episodes": NUM_EPISODES,
            "strategy": "state_based",
        },
        "results": {
            "total_selections": len(all_selections),
            "selection_distribution": dict(selection_counts),
            "selection_history": selection_history,
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 04 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
