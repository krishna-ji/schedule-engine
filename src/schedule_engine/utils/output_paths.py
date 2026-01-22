"""Helpers for consistent output directory structure."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path


def _ensure_dir(path: Path) -> Path:
    """Ensure ``path`` exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_csv_dir(output_dir: str | Path) -> Path:
    """Return the canonical csv/ directory for a run (creating it if needed)."""

    return _ensure_dir(Path(output_dir) / "csv")


def get_nsga_plot_dir(output_dir: str | Path) -> Path:
    """Return the plots/nsga directory (creating it if needed)."""

    return _ensure_dir(Path(output_dir) / "plots" / "nsga")


def get_constraint_plot_dir(output_dir: str | Path) -> Path:
    """Return the plots/constraints directory (creating it if needed)."""

    return _ensure_dir(Path(output_dir) / "plots" / "constraints")


def move_console_log_to_run(
    log_path: str | Path,
    output_dir: str | Path,
    target_name: str = "log_console.log",
) -> Path | None:
    """Move the structured console log into the final run directory.

    The StructuredLogger writes to an absolute path before the ExperimentManager
    knows which output folder will be used. This helper closes the logging
    subsystem, moves the file, and returns the new path so callers can surface
    it in summaries.
    """

    source = Path(log_path)
    if not source.exists():
        return None

    dest_dir = _ensure_dir(Path(output_dir))
    dest = dest_dir / target_name

    # Flush and close all handlers before moving to avoid Windows file locks.
    logging.shutdown()

    if source.resolve() == dest.resolve():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), dest)
    return dest
