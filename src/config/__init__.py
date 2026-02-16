"""Configuration for Schedule Engine.

Config is a recursive namespace with dot-access.
No Pydantic, no dataclasses, no centralized defaults.

Each entry point (run file, helper) defines its own config values::

    config = Config(
        ga=dict(ngen=200, pop_size=100),
        soft_constraints=dict(...)
    )
    init_config(config)

Internal code reads via dot-access::

    config.ga.ngen        # 200
    config.repair.enabled # True
"""

from __future__ import annotations

import copy
from typing import Any


# UTILITIES
def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a copy of *base*."""
    merged = base.copy()
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


# CONFIG CLASS
class Config:
    """Recursive namespace with dot-access over a dict tree.

    Create by passing all values you need::

        config = Config(ga=dict(ngen=200), repair=dict(enabled=True))
        config.ga.ngen         # -> 200
        config.repair.enabled  # -> True
    """

    def __init__(self, **kwargs: Any) -> None:
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, Config(**val))
            else:
                setattr(self, key, val)

    def __getattr__(self, name: str) -> Any:
        """Allow dynamic attribute access for mypy compatibility."""
        raise AttributeError(f"Config has no attribute {name!r}")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __bool__(self) -> bool:
        return bool(self.__dict__)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __iter__(self) -> Any:
        return iter(self.__dict__)

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain nested dict (for pickling / JSON / workers)."""
        out: dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if isinstance(val, Config):
                out[key] = val.to_dict()
            else:
                out[key] = copy.deepcopy(val)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        """Reconstruct a Config from a dict (inverse of to_dict())."""
        return cls(**d)

    def __repr__(self) -> str:
        items = ", ".join(
            f"{k}=..." if isinstance(v, Config) else f"{k}={v!r}"
            for k, v in self.__dict__.items()
        )
        return f"Config({items})"


# GLOBAL SINGLETON
_config: Config | None = None


def init_config(config_obj: Config) -> Config:
    """Set the global config and return it."""
    global _config
    _config = config_obj
    return _config


def get_config() -> Config:
    """Return the global config. Must call init_config() first."""
    global _config  # noqa: PLW0602
    if _config is None:
        raise RuntimeError(
            "Config not initialized. Call init_config(Config(...)) first."
        )
    return _config


def get_config_or_default() -> Config:
    """Return global config, or empty Config if not initialized."""
    global _config  # noqa: PLW0602
    if _config is None:
        return Config()
    return _config


__all__ = [
    "Config",
    "_deep_merge",
    "get_config",
    "get_config_or_default",
    "init_config",
]
