"""Helper entry point for running mypy via ``uv run typecheck``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run MyPy using the settings defined in ``pyproject.toml``."""

    mypy_args = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file=pyproject.toml",
        "--no-incremental",
    ]

    cache_dir = PROJECT_ROOT / ".mypy_cache"
    if cache_dir.exists():
        # Remove stale cache to avoid upstream mypy bug with built-in modules
        import shutil

        shutil.rmtree(cache_dir, ignore_errors=True)

    result = subprocess.run(mypy_args, cwd=PROJECT_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
