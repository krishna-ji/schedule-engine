#!/usr/bin/env python3
"""
Lightweight wrapper that sets RL_DEFAULT_PROFILE=prod and passes through args
so `uv run train-prod -- --timesteps 10000` works as expected.
"""
from __future__ import annotations

import os
import sys


def main():
    os.environ["RL_DEFAULT_PROFILE"] = "prod"
    # Import training module and call its main() with current argv
    try:
        from src.rl.training.train_script import main as train_main

        train_main()
    except Exception as e:
        # Print error nicely and re-raise
        print(f"Failed to start prod training: {e}")
        raise


if __name__ == "__main__":
    main()
