"""
Utilities for handling experiment names and output directories.
"""

from __future__ import annotations

import re


def sanitize_experiment_name(name: str | None) -> str | None:
    """Sanitize an experiment name into a safe filesystem-friendly string.

    Returns None for empty or whitespace-only input.
    Keeps lowercase letters, digits, underscores, hyphens and dots. Turns other
    characters into hyphens and truncates to 50 characters.
    """
    if not name:
        return None
    s = str(name).strip()
    if not s:
        return None

    # Normalize to lowercase
    s = s.lower()
    # Replace invalid characters with '-'
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    # Strip left/right dots and dashes
    s = s.strip(".-_")
    if not s:
        return None
    return s[:50]
