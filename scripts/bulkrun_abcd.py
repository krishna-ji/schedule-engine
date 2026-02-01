#!/usr/bin/env python3
"""
Bulk runner for modes A, B, C, D.

Usage:
    python3 scripts/bulkrun_abcd.py

Runs in order:
  - runs/mode_a_baseline.py
  - runs/mode_b_memetic.py
  - runs/mode_c_roundrobin.py
  - runs/mode_d_adaptive.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "runs"

    run_files = [
        runs_dir / "mode_a_baseline.py",
        runs_dir / "mode_b_memetic.py",
        runs_dir / "mode_c_roundrobin.py",
        runs_dir / "mode_d_adaptive.py",
    ]

    for run_file in run_files:
        if not run_file.exists():
            print(f"[error] Missing file: {run_file}")
            return 1

    for run_file in run_files:
        print(f"\n[run] {run_file.name}")
        result = subprocess.run(
            [sys.executable, str(run_file)],
            cwd=str(project_root),
            check=False,
        )
        if result.returncode != 0:
            print(f"[error] {run_file.name} failed with code {result.returncode}")
            return result.returncode

    print("\n[ok] All A/B/C/D runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
