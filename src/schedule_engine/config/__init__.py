"""Configuration module for Schedule Engine.

Simple global config: set it once, access it anywhere.
No Pydantic, no loaders, no transformation layers.
"""

from __future__ import annotations

from schedule_engine.config.models import Config

# Global config (set by run files via init_config)
_config: Config | None = None


def init_config(config_obj: Config) -> Config:
    """Set the global config and return it."""
    global _config
    _config = config_obj
    return _config


def get_config() -> Config:
    """Return the global config. Must call init_config() first."""
    global _config
    if _config is None:
        raise RuntimeError(
            "Config not initialized. Call init_config(Config(...)) first."
        )
    return _config


def get_config_or_default() -> Config:
    """Return global config, or a fresh default Config if not initialized."""
    global _config
    if _config is None:
        return Config()
    return _config


__all__ = [
    "Config",
    "get_config",
    "get_config_or_default",
    "init_config",
]
