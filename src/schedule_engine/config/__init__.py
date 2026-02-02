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


def get_config_or_default() -> Config:
    """Return cached config, or create a default test config for standalone usage.

    Use this in utility functions that may be called from standalone scripts
    (e.g., runs/*.py) where full config initialization isn't required.

    For main application code, prefer get_config() for fail-fast behavior.
    """

    global _config
    if _config is None:
        from schedule_engine.config.loader import dict_to_pydantic

        default_dict = {
            "experiment_name": "Default Test Config",
            "environment": "test",
            "ngen": 30,
            "pop_size": 10,
            "cxpb": 0.70,
            "mutpb": 0.20,
        }
        # Don't cache - each call returns fresh default (avoids side effects)
        return dict_to_pydantic(default_dict)
    return _config


__all__: Final[list[str]] = [
    "get_config",
    "get_config_or_default",
    "init_config",
    "Config",
]
