"""Configuration module: thin wrapper over the loader + Pydantic models."""

from __future__ import annotations

from typing import Final

from src.config.models import Config

# Global config object (set during CLI bootstrap)
_config: Config | None = None


def init_config(
    config_obj: Config,
) -> Config:
    """Initialize the global config once and return it."""

    global _config
    _config = config_obj
    return _config


def get_config() -> Config:
    """Return the cached config (must be initialized first)."""

    global _config
    if _config is None:
        # Lazy initialize a default Config so unit tests and utilities can
        # use get_config() without explicit bootstrap. Build the default
        # Pydantic Config from the dataclass TestConfig to ensure all
        # nested fields are populated and validated.
        from configs.profiles import TestConfig

        _config = TestConfig().to_pydantic()
    return _config


__all__: Final[list[str]] = ["get_config", "init_config", "Config"]
