"""Utility helpers for loading RL training configuration profiles."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.rl.training.presets import TRAINING_BASE_DEFAULTS, TRAINING_PROFILE_OVERRIDES

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_PROFILE = "test"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries and return the merged result."""

    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_profile_overrides(
    profile: str, stack: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Resolve profile overrides with optional inheritance."""

    stack = stack or ()
    if profile in stack:
        cycle = " -> ".join((*stack, profile))
        raise ValueError(f"Circular training profile inheritance detected: {cycle}")

    try:
        raw = deepcopy(TRAINING_PROFILE_OVERRIDES[profile])
    except KeyError as exc:
        raise ValueError(f"Unknown training profile: {profile}") from exc

    parent = raw.pop("inherits", raw.pop("base_profile", None))
    if parent:
        parent_overrides = _resolve_profile_overrides(parent, (*stack, profile))
        return _deep_merge(parent_overrides, raw)

    return raw


def _load_custom_override(path: Path) -> dict[str, Any]:
    """Load custom override from supported file formats (currently JSON)."""

    if not path.exists():
        raise FileNotFoundError(f"Override file not found: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError("Custom training overrides must be JSON files")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Custom override file must contain a JSON object")

    return data


def list_training_profiles() -> Iterable[str]:
    """Return the set of available training profiles."""

    return sorted(TRAINING_PROFILE_OVERRIDES.keys())


def load_training_config(
    profile: str | None = None,
    custom_path: str | None = None,
) -> dict[str, Any]:
    """Build RL training configuration from Python presets."""

    selected_profile = profile or DEFAULT_PROFILE
    config = deepcopy(TRAINING_BASE_DEFAULTS)
    overrides = _resolve_profile_overrides(selected_profile)
    config = _deep_merge(config, overrides)

    if custom_path:
        custom_data = _load_custom_override(Path(custom_path))
        config = _deep_merge(config, custom_data)

    config.setdefault("profile", selected_profile)
    return config
