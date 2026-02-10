#!/usr/bin/env python3
"""
RL Experiment 06: Adaptive Probabilities

Demonstrate adaptive GA probability configuration (fixed vs adaptive).

Usage:
    python runs/rl_06_adaptive_probabilities.py
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
    log_file = output_dir / "rl_06_adaptive_probabilities.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("rl_06_adaptive_probabilities")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run RL Experiment 06: Adaptive Probabilities."""

    # CONFIGURATION

    SEED = 42

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = PROJECT_ROOT / "output" / "rl_06_adaptive_probabilities" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 06: ADAPTIVE PROBABILITIES")
    logger.info("=" * 60)
    logger.info(f"Config: seed={SEED}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # COMPARE FIXED VS ADAPTIVE CONFIGURATIONS

    logger.info("Comparing fixed vs adaptive configurations...")

    # Create config with fixed probabilities (default)
    fixed_config = build_notebook_config(
        seed=SEED, overrides={"use_adaptive_probabilities": False}
    )

    # Create config with adaptive probabilities
    adaptive_config = build_notebook_config(
        seed=SEED, overrides={"use_adaptive_probabilities": True}
    )

    logger.info("=" * 60)
    logger.info("CONFIGURATION COMPARISON")
    logger.info("=" * 60)
    logger.info("Fixed Probabilities Config:")
    logger.info(
        f"  use_adaptive_probabilities: {fixed_config.ga.use_adaptive_probabilities}"
    )
    logger.info(f"  cxpb: {fixed_config.ga.cxpb}")
    logger.info(f"  mutpb: {fixed_config.ga.mutpb}")

    logger.info("Adaptive Probabilities Config:")
    logger.info(
        f"  use_adaptive_probabilities: {adaptive_config.ga.use_adaptive_probabilities}"
    )
    logger.info(f"  cxpb: {adaptive_config.ga.cxpb}")
    logger.info(f"  mutpb: {adaptive_config.ga.mutpb}")
    logger.info("=" * 60)

    # RESULTS SUMMARY

    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 06: ADAPTIVE PROBABILITIES RESULTS")
    logger.info("=" * 60)
    logger.info(
        """
Key Difference:
- Fixed: cxpb/mutpb remain constant throughout evolution
- Adaptive: Probabilities adjust based on population diversity
  and fitness improvement rate

When adaptive_probabilities=True:
- If diversity is low: increase mutation probability
- If improvement stagnates: adjust crossover/mutation balance
- Self-tunes based on search progress
"""
    )
    logger.info("=" * 60)

    # SAVE RESULTS

    logger.info("Saving results...")
    results_data = {
        "experiment": "rl_06_adaptive_probabilities",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
        },
        "results": {
            "fixed_config": {
                "use_adaptive_probabilities": (
                    fixed_config.ga.use_adaptive_probabilities
                ),
                "cxpb": fixed_config.ga.cxpb,
                "mutpb": fixed_config.ga.mutpb,
            },
            "adaptive_config": {
                "use_adaptive_probabilities": (
                    adaptive_config.ga.use_adaptive_probabilities
                ),
                "cxpb": adaptive_config.ga.cxpb,
                "mutpb": adaptive_config.ga.mutpb,
            },
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("RL EXPERIMENT 06 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
