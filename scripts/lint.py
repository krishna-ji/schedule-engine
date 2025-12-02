"""Helper wrappers for running Ruff via ``uv run lint`` commands."""

from __future__ import annotations

import subprocess
import sys


def _run_ruff(args: list[str]) -> int:
    """Invoke Ruff with the provided arguments and propagate its exit code."""

    result = subprocess.run([sys.executable, "-m", "ruff", *args], check=False)
    return result.returncode


def main_check() -> int:
    """Execute ``ruff check .``"""

    return _run_ruff(["check", "."])


def main_fix() -> int:
    """Execute ``ruff check . --fix``"""

    return _run_ruff(["check", ".", "--fix"])
