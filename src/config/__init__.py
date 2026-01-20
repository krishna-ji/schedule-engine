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
    """Return the cached config (must be initialized first).

    For notebook workflows, use init_config() explicitly.
    For standalone scripts/tests, a default test config is created.
    """

    global _config
    if _config is None:
        # Create a default test config for standalone usage
        from src.config.loader import dict_to_pydantic

        default_dict = {
            "experiment_name": "Default Test Config",
            "environment": "test",
            "ngen": 30,
            "pop_size": 10,
            "cxpb": 0.70,
            "mutpb": 0.20,
        }
        _config = dict_to_pydantic(default_dict)
    return _config


__all__: Final[list[str]] = ["get_config", "init_config", "Config"]
