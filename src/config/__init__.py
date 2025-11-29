"""Configuration module: thin wrapper over the loader + Pydantic models."""

from __future__ import annotations

from typing import Final

from src.config.loader import load_config
from src.config.models import Config
from src.config.presets.base import ConfigBlueprint
from src.config.presets.profiles import Profile

# Global config object (set during CLI bootstrap)
_config: Config | None = None


def init_config(
    profile: Profile | str | None = None,
    config_obj: Config | None = None,
    blueprint: ConfigBlueprint | None = None,
) -> Config:
    """Initialize the global config once and return it."""

    global _config
    if config_obj is not None:
        _config = config_obj
    else:
        if blueprint is None:
            raise ValueError("Either config_obj or blueprint must be provided")
        _config = load_config(blueprint=blueprint, profile=profile)
    return _config


def get_config() -> Config:
    """Return the cached config (must be initialized first)."""

    global _config
    if _config is None:
        raise RuntimeError(
            "Config not initialized! Call init_config() first or provide blueprint."
        )
    return _config


__all__: Final[list[str]] = ["get_config", "init_config", "Config"]
