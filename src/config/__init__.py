"""Configuration module: thin wrapper over the loader + Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.config.loader import load_config
from src.config.models import Config

# Global config object (set during CLI bootstrap)
_config: Config | None = None


def init_config(
    config_path: str | Path | None = None, config_obj: Config | None = None
) -> Config:
    """Initialize the global config once and return it."""

    global _config
    if config_obj is not None:
        _config = config_obj
    else:
        _config = load_config(str(config_path) if config_path else None)
    return _config


def get_config() -> Config:
    """Return the cached config, loading from disk if necessary."""

    global _config
    if _config is None:
        _config = load_config()
    return _config


__all__: Final[list[str]] = ["get_config", "init_config", "Config"]
