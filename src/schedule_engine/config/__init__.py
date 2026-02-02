"""Configuration module: thin wrapper over the loader + Pydantic models."""

from __future__ import annotations

from typing import Final

from schedule_engine.config.models import Config

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
    """Return the cached config (must be initialized first).

    For notebook workflows, use init_config() explicitly before calling this.
    For CLI workflows, the launcher initializes config automatically.

    Raises:
        RuntimeError: If config has not been initialized via init_config()
    """

    global _config
    if _config is None:
        raise RuntimeError(
            "Config not initialized. Call init_config() before get_config(). "
            "For notebooks: from schedule_engine.config.loader import load_from_dataclass; "
            "config = load_from_dataclass(your_config_dataclass); init_config(config)"
        )
    return _config


__all__: Final[list[str]] = ["get_config", "init_config", "Config"]
