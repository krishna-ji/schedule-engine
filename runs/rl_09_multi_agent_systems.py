#!/usr/bin/env python3
"""
RL Experiment 09: Multi-Agent Systems

Analyze agent selection dynamics across multiple episodes with multi-agent coordinator.

Usage:
    python runs/rl_09_multi_agent_systems.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schedule_engine.rl.helpers import (
    build_notebook_config,
    create_env,
    load_context,
    set_global_seed,
)
from schedule_engine.rl.multi_agent.agent_coordinator import AgentCoordinator


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "rl_09_multi_agent_systems.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_09_multi_agent_systems")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 09: Multi-Agent Systems."""

    # CONFIGURATION

    SEED = 42
    POP_SIZE = 20
    MAX_GENERATIONS = 50
    MAX_STEPS = 15
    NUM_EPISODES = 10  # Episodes to run for analysis

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_09_multi_agent_systems" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 09: MULTI-AGENT SYSTEMS")
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

    # RUN MULTI-EPISODE SELECTION ANALYSIS

    logger.info("Running multi-episode selection analysis...")
    coordinator = AgentCoordinator(strategy="state_based")

    all_selections: list[str] = []
    episode_data: list[dict[str, object]] = []

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        episode_selections: list[str] = []
        episode_states: list[dict[str, object]] = []

        for step in range(MAX_STEPS):
            # Get current state for agent selection
            state = {
                "generation": info.get("generation", 0),
                "generations_without_improvement": info.get(
                    "generations_without_improvement", 0
                ),
            }

            # Select specialist agent
            agent = coordinator.select_agent(env.population, state, obs)
            episode_selections.append(agent.name)
            episode_states.append(state.copy())

            # Take random action (for demonstration)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        all_selections.extend(episode_selections)
        episode_data.append(
            {
                "episode": episode,
                "selections": episode_selections,
                "states": episode_states,
                "num_steps": len(episode_selections),
            }
        )

        logger.info(
            f"Episode {episode+1}/{NUM_EPISODES}: {len(episode_selections)} steps, agents={episode_selections[:3]}..."
        )

    logger.info(
        f"Completed {NUM_EPISODES} episodes, {len(all_selections)} total selections"
    )

    # RESULTS SUMMARY

    selection_counts = Counter(all_selections)

    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 09: MULTI-AGENT SYSTEMS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total selections: {len(all_selections)}")
    logger.info(f"Episodes: {NUM_EPISODES}")
    logger.info("Agent Selection Distribution:")
    for agent_name, count in selection_counts.most_common():
        pct = 100 * count / len(all_selections)
        logger.info(f"  {agent_name:20s}: {count:4d} ({pct:5.1f}%)")

    # Episode statistics
    steps_per_episode = [ed["num_steps"] for ed in episode_data]
    logger.info("Steps per Episode:")
    logger.info(f"  Mean: {sum(steps_per_episode)/len(steps_per_episode):.1f}")  # type: ignore[arg-type]
    logger.info(f"  Min:  {min(steps_per_episode)}")  # type: ignore[type-var]
    logger.info(f"  Max:  {max(steps_per_episode)}")  # type: ignore[type-var]
    logger.info("=" * 60)

    # SAVE RESULTS

    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_09_multi_agent_systems",
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
            "steps_per_episode": steps_per_episode,
            "episode_selections": [ed["selections"] for ed in episode_data],
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 09 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
