"""JSON serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _json_key(key: Any) -> Any:
    """Ensure dict keys are JSON-serializable."""
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    if isinstance(key, (np.integer, np.floating, np.bool_)):
        return key.item()
    if isinstance(key, Path):
        return str(key)
    return str(key)


def to_jsonable(value: Any) -> Any:
    """Convert common non-JSON types into JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {_json_key(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value
