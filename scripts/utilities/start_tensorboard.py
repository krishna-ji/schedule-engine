#!/usr/bin/env python3
"""
Start TensorBoard for RL training logs.

Cross-platform Python replacement for start_tensorboard.ps1/.sh
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Start TensorBoard server for training logs."""
    log_dir = Path("logs/tensorboard/train")

    # Check if log directory exists
    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist")
        print(f"Please train an RL agent first to generate logs")
        sys.exit(1)

    # Check if there are any log files
    log_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not log_files:
        print(f"Warning: No TensorBoard log files found in '{log_dir}'")
        print(f"TensorBoard will start but may show empty dashboards")

    port = 6006
    print("=" * 60)
    print("TENSORBOARD - RL Training Dashboard")
    print("=" * 60)
    print(f"Log directory: {log_dir.absolute()}")
    print(f"Port: {port}")
    print(f"URL: http://localhost:{port}/")
    print("\nPress Ctrl+C to stop TensorBoard")
    print("=" * 60)
    print()

    try:
        # Start TensorBoard using uv run
        subprocess.run(
            [
                "uv",
                "run",
                "tensorboard",
                "--logdir",
                str(log_dir),
                "--port",
                str(port),
                "--bind_all",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n\nTensorBoard stopped")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting TensorBoard: {e}")
        print("\nMake sure tensorboard is installed:")
        print("  uv add tensorboard")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'uv' command not found")
        print("Please install UV package manager first")
        sys.exit(1)


if __name__ == "__main__":
    main()
