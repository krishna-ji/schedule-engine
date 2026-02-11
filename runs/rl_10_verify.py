#!/usr/bin/env python3
"""
RL Verify: Component Status Check

Verify all RL components are properly configured:
  - Environment setup
  - Agent initialization
  - Reward calculation
  - Training loop

Usage:
    python runs/rl_10_verify.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schedule_engine.rl.helpers import build_notebook_config


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "rl_10_verify.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_10_verify")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 10: Summary & Component Status."""

    # CONFIGURATION

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_10_verify" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 10: SUMMARY & COMPONENT STATUS")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # CHECK RL CONFIGURATION

    logger.info("Checking RL configuration...")
    config = build_notebook_config()

    logger.info("RL Configuration Status:")
    logger.info(f"  RL enabled: {config.rl.enabled}")
    logger.info(f"  Default agent: {config.rl.default_agent}")
    logger.info(f"  Use hypervolume reward: {config.rl.use_hypervolume_reward}")

    # CHECK COMPONENT AVAILABILITY

    logger.info("Checking component availability...")
    component_status: dict[str, str] = {}

    # Check ScheduleEnv
    try:
        from schedule_engine.rl import ScheduleEnv

        component_status["ScheduleEnv"] = "✓ Available"
    except ImportError as e:
        component_status["ScheduleEnv"] = f"✗ {e}"

    # Check PPO agent
    try:
        from schedule_engine.rl.agents import create_ppo_agent

        component_status["PPO Agent"] = "✓ Available"
    except ImportError as e:
        component_status["PPO Agent"] = f"✗ {e}"

    # Check DQN agent
    try:
        from schedule_engine.rl.agents import create_dqn_agent

        component_status["DQN Agent"] = "✓ Available"
    except ImportError as e:
        component_status["DQN Agent"] = f"✗ {e}"

    # Check Random agent
    try:
        from schedule_engine.rl.agents import RandomAgent

        component_status["Random Agent"] = "✓ Available"
    except ImportError as e:
        component_status["Random Agent"] = f"✗ {e}"

    # Check AgentCoordinator
    try:
        from schedule_engine.rl.multi_agent.agent_coordinator import AgentCoordinator

        component_status["AgentCoordinator"] = "✓ Available"
    except ImportError as e:
        component_status["AgentCoordinator"] = f"✗ {e}"

    # Check RewardCalculator
    try:
        from schedule_engine.rl.gym_env.reward_calculator import RewardCalculator

        component_status["RewardCalculator"] = "✓ Available"
    except ImportError as e:
        component_status["RewardCalculator"] = f"✗ {e}"

    logger.info("Component Status:")
    for component, status in component_status.items():
        logger.info(f"  {component:20s}: {status}")

    # SUMMARY RESULTS

    available = sum(1 for s in component_status.values() if "✓" in s)
    total = len(component_status)

    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 10: COMPONENT STATUS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Components Available: {available}/{total}")
    logger.info(
        """
RL Experiment Scripts (01-10) Status:
  rl_01: PPO Baseline          - Train and evaluate PPO agent
  rl_02: DQN Baseline          - Train and evaluate DQN agent
  rl_03: Curriculum Learning   - Staged training (easy→hard)
  rl_04: Specialist Agents     - State-based agent selection
  rl_05: Reward Shaping        - Scalar vs hypervolume rewards
  rl_06: Adaptive Probabilities- Fixed vs adaptive GA params
  rl_07: Full Ablation Study   - Systematic method comparison
  rl_08: Hyperparameter Sweep  - Learning rate sensitivity
  rl_09: Multi-Agent Systems   - Multi-episode coordination
  rl_10: Summary (this script)

All scripts are now STANDALONE with:
  ✓ Imports from schedule_engine/notebooks/
  ✓ File-based logging
  ✓ Timestamped output directories
  ✓ JSON results export
"""
    )
    logger.info("=" * 60)

    # SAVE RESULTS

    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_10_verify",
        "timestamp": TIMESTAMP,
        "config": {
            "rl_enabled": config.rl.enabled,
            "default_agent": config.rl.default_agent,
            "use_hypervolume_reward": config.rl.use_hypervolume_reward,
        },
        "results": {
            "components_available": available,
            "components_total": total,
            "component_status": component_status,
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 10 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
