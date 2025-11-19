"""Utility helpers for loading RL training configuration profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_CONFIG_DIR = PROJECT_ROOT / "configs" / "training"
BASE_CONFIG_PATH = TRAIN_CONFIG_DIR / "base.yaml"
DEFAULT_PROFILE = "test"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file and return its content as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def list_training_profiles() -> Iterable[str]:
    """Return the set of available training profiles (excluding base)."""
    if not TRAIN_CONFIG_DIR.exists():
        return []

    return sorted(
        path.stem
        for path in TRAIN_CONFIG_DIR.glob("*.yaml")
        if path.name not in {"base.yaml"}
    )


def load_training_config(
    profile: str | None = None,
    custom_path: str | None = None,
) -> Dict[str, Any]:
    """
    Load RL training configuration by merging base + profile + optional custom file.
    """
    selected_profile = profile or DEFAULT_PROFILE
    profile_path = TRAIN_CONFIG_DIR / f"{selected_profile}.yaml"

    config = _load_yaml(BASE_CONFIG_PATH)
    config = _deep_merge(config, _load_yaml(profile_path))

    if custom_path:
        config = _deep_merge(config, _load_yaml(Path(custom_path)))

    config.setdefault("profile", selected_profile)
    return config
