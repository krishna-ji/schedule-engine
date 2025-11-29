"""Configuration module: thin wrapper over the loader + Pydantic models."""

from __future__ import annotations

from typing import Final

from src.config.loader import load_config
from src.config.models import Config
from src.config.presets.base import ConfigBlueprint
from src.config.presets.profiles import Profile
from src.config.runtime_mode import RuntimeMode

# Global config object (set during CLI bootstrap)
_config: Config | None = None


def init_config(
    runtime_mode: RuntimeMode | None = None,
    profile: Profile | str | None = None,
    config_obj: Config | None = None,
    blueprint: ConfigBlueprint | None = None,
) -> Config:
    """Initialize the global config once and return it."""

    global _config
    if config_obj is not None:
        _config = config_obj
    else:
        _config = load_config(
            runtime_mode=runtime_mode, profile=profile, blueprint=blueprint
        )
    return _config


def get_config() -> Config:
    """Return the cached config, loading from disk if necessary."""

    global _config
    if _config is None:
        _config = load_config()
    return _config


__all__: Final[list[str]] = ["get_config", "init_config", "Config"]
